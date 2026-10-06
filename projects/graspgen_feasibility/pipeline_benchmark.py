"""Single-view integration smoke test: SAM2 -> official XYZ -> GraspGen -> collision filter."""
from pathlib import Path
import gc
import json
import sys
import time

from benchmark import ROOT, build_sampler, measure, sample_all, seed
import numpy as np
import torch
from PIL import Image, ImageDraw
from scipy.spatial import cKDTree

OLD = ROOT.parent/'goldilocs_reproduction'
sys.path.insert(0,str(OLD/'third_party/sam2'))

def main():
    started=time.perf_counter()
    x=json.loads((ROOT/'data/sample_data/real_scene_pc/1745619598_467125.json').read_text())
    rgb=np.asarray(x['scene_info']['img_color'],dtype=np.uint8)
    xyz=np.asarray(x['scene_info']['pc_color'],dtype=np.float32).reshape(*rgb.shape[:2],3)
    depth=np.asarray(x['scene_info']['img_depth'])
    valid=np.isfinite(xyz).all(-1)&np.isfinite(depth)&(depth>0)
    reference=np.asarray(x['object_info']['pc'],dtype=np.float32)
    distances,indices=cKDTree(xyz.reshape(-1,3)).query(reference)
    if distances.max()>1e-5:raise RuntimeError('Reference point/pixel correspondence is not exact')
    v,u=np.unravel_index(indices,rgb.shape[:2])
    box=np.array([u.min(),v.min(),u.max(),v.max()],dtype=np.float32)
    reference_pixels=np.zeros(rgb.shape[:2],dtype=bool);reference_pixels[v,u]=True
    io_seconds=time.perf_counter()-started
    from sam2.build_sam import build_sam2
    from sam2.sam2_image_predictor import SAM2ImagePredictor
    sam, sam_load=measure(lambda:build_sam2('configs/sam2/sam2_hiera_l.yaml',
                         str(OLD/'checkpoints/sam2_hiera_large.pt'),device='cuda',apply_postprocessing=False))
    predictor=SAM2ImagePredictor(sam)
    sampler, grasp_load=measure(build_sampler)
    from grasp_gen.robot import get_gripper_info
    from grasp_gen.utils.point_cloud_utils import filter_colliding_grasps_fast
    gripper=get_gripper_info('robotiq_2f_140').collision_mesh
    stages={}
    seed(1000)
    def segment():
        with torch.inference_mode(), torch.autocast('cuda',dtype=torch.bfloat16):
            predictor.set_image(rgb)
            masks, scores, logits = predictor.predict(box=box,multimask_output=True)
        best=int(np.argmax(scores))
        return masks[best].astype(bool),float(scores[best])
    def full():
        (mask,confidence),stages['segmentation']=measure(segment)
        def extract():
            target=xyz[mask&valid]
            background=xyz[(~mask)&valid]
            if len(target)<100:raise RuntimeError('Too few target points')
            # Same scene budget used by the author's demo; fixed seed.
            if len(background)>8192:
                background=background[np.random.choice(len(background),8192,replace=False)]
            return torch.as_tensor(target,dtype=torch.float32,device='cuda'),background
        (target,background),stages['pointcloud_extraction']=measure(extract)
        (poses,scores),stages['grasp_generation']=measure(lambda:sample_all(sampler,target,100))
        keep=scores>=.8
        selected=poses[keep].cpu().numpy(); selected_scores=scores[keep].cpu().numpy()
        free,stages['collision_check']=measure(lambda:filter_colliding_grasps_fast(
            scene_pc=background,grasp_poses=selected,gripper_collision_mesh=gripper,
            collision_threshold=.02,num_collision_samples=2000,batch_size=4))
        return mask,confidence,target.cpu().numpy(),background,selected,selected_scores,free,int(len(poses))
    # Outer measure owns end-to-end memory reset; nested measures reset counters.
    # Preserve a full-run memory high-water mark using stage maxima below.
    output,total=measure(full)
    mask,confidence,target,background,poses,scores,free,raw_count=output
    total['peak_allocated_mib']=max(s['peak_allocated_mib'] for s in stages.values())
    total['peak_reserved_mib']=max(s['peak_reserved_mib'] for s in stages.values())
    total['device_peak_used_mib']=max([total['device_peak_used_mib']]+[s['device_peak_used_mib'] for s in stages.values()])
    overlay=rgb.copy();overlay[mask]=(overlay[mask]*.45+np.array([0,255,0])*.55).astype(np.uint8)
    im=Image.fromarray(overlay);ImageDraw.Draw(im).rectangle(box.tolist(),outline='red',width=3)
    im.save(ROOT/'runs/pipeline_segmentation.png')
    np.savez_compressed(ROOT/'runs/pipeline_output.npz',target=target,background=background,
                        poses=poses,scores=scores,collision_free=free,mask=mask,prompt_box=box)
    ref_valid=reference_pixels&valid;pred_valid=mask&valid
    report={'kind':'single_view_integration_smoke_test_not_real_robot_grasp_success',
            'source':'official GraspGen real_scene_pc/1745619598_467125.json',
            'image_shape':list(rgb.shape),'prompt_box':box.tolist(),
            'prompt_origin':'Bounding box of the official segmented object, matched to exact XYZ pixels; simulates user object selection.',
            'depth_usage':'Official per-pixel XYZ in world frame is indexed by SAM2 mask; no guessed camera intrinsics or new depth network.',
            'segmentation_model':'SAM2 Hiera Large original checkpoint, BF16 inference; no optional CUDA postprocessing',
            'grasp_model':'Official Robotiq-2F-140 PointNet++ checkpoint, FP32, 10 diffusion steps',
            'execution':'Both models resident simultaneously; single cold image inference; checkpoint loading excluded from pipeline time.',
            'reference_note':'Reference is correspondence to provided segmented points, not independently annotated segmentation ground truth.',
            'mask_reference_valid_iou':float((ref_valid&pred_valid).sum()/max((ref_valid|pred_valid).sum(),1)),
            'sam_predicted_iou_score':confidence,'mask_pixels':int(mask.sum()),'valid_object_points':len(target),
            'raw_grasps':raw_count,'above_08':len(poses),'collision_free':int(free.sum()),
            'collision_parameters':{'threshold_m':.02,'scene_points':len(background),'gripper_surface_samples':2000,'batch_size':4},
            'io_seconds':io_seconds,'sam_loading':sam_load,'grasp_loading':grasp_load,'stages':stages,'total':total,
            'limitations':['One scene and one view; no multiview fusion, approach-path collision, arm reachability or physical grasp execution.','Sparse observed-scene collision filtering is not a guarantee for unobserved geometry.']}
    (ROOT/'runs/pipeline_summary.json').write_text(json.dumps(report,indent=2))
    print(json.dumps(report,indent=2),flush=True)

if __name__=='__main__':main()
