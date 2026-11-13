"""Replay measured box/cup proxies in PyBullet and compare both camera views."""
import argparse
import json
import math
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw
import pybullet as p


def polygon_mask(size,coordinates):
    image=Image.new('L',size,0)
    ImageDraw.Draw(image).polygon(list(zip(coordinates[::2],coordinates[1::2])),fill=1)
    return np.asarray(image,dtype=bool)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('geometry',type=Path)
    args=parser.parse_args()
    geometry_path=args.geometry.resolve()
    geom=json.loads(geometry_path.read_text(encoding='utf-8'))
    reg=json.loads(Path(geom['registration']).read_text(encoding='utf-8'))
    config=json.loads(Path(geom['view_config']).read_text(encoding='utf-8'))
    objects=geom['objects']
    out=geometry_path.parent/'scene_replay'
    out.mkdir(exist_ok=True)
    client=p.connect(p.DIRECT)
    if client<0:
        raise RuntimeError('PyBullet DIRECT connection failed')
    try:
        plane=p.createCollisionShape(p.GEOM_PLANE)
        visual=p.createVisualShape(p.GEOM_BOX,halfExtents=[.65,.65,.005],
                                   rgbaColor=[.88,.88,.85,1])
        p.createMultiBody(0,plane,visual,[0,.4,-.005])
        box=objects['box']
        half=(np.asarray(box['dimensions_m'])/2).tolist()
        shape=p.createCollisionShape(p.GEOM_BOX,halfExtents=half)
        color=p.createVisualShape(p.GEOM_BOX,halfExtents=half,
                                  rgbaColor=[.66,.47,.27,1])
        box_id=p.createMultiBody(0,shape,color,box['center_table_m'],
                                 p.getQuaternionFromEuler([0,0,box['yaw_table_rad']]))
        cup=objects['cup']
        shape=p.createCollisionShape(p.GEOM_CYLINDER,
                                     radius=cup['radius_m'],height=cup['height_m'])
        color=p.createVisualShape(p.GEOM_CYLINDER,
                                  radius=cup['radius_m'],length=cup['height_m'],
                                  rgbaColor=[.08,.10,.13,1])
        cup_id=p.createMultiBody(0,shape,color,cup['center_table_m'])
        contacts=p.getClosestPoints(box_id,cup_id,distance=.5)
        clearance=min((float(point[8]) for point in contacts),default=None)
        if clearance is None or clearance <= 0:
            raise ValueError('Reconstructed box and cup collision shapes overlap')
        metrics=[]
        for i,source in enumerate(reg['views']):
            source=Path(source)
            real=np.load(source/'depth_m.npy',allow_pickle=False)
            capture=json.loads((source/'report.json').read_text(encoding='utf-8'))
            height,width=real.shape
            K=capture['aligned_depth_intrinsic']
            T_cam_view_table=np.asarray(reg['T_view_camera_table'][i])
            T_ref_view_table=np.asarray(reg['T_reference_table_view_table'][i])
            T_cam_ref_table=T_cam_view_table@np.linalg.inv(T_ref_view_table)
            T_ref_table_cam=np.linalg.inv(T_cam_ref_table)
            eye=T_ref_table_cam[:3,3]
            R=T_ref_table_cam[:3,:3]
            forward=R@np.array([0.,0.,1.])
            up=R@np.array([0.,-1.,0.])
            view=p.computeViewMatrix(eye.tolist(),(eye+forward).tolist(),up.tolist())
            near,far=.05,3.
            vfov=math.degrees(2*math.atan(height/(2*K['fy'])))
            projection=p.computeProjectionMatrixFOV(vfov,width/height,near,far)
            _,_,rgba,buffer,labels=p.getCameraImage(width,height,view,projection,
                                                     renderer=p.ER_TINY_RENDERER)
            image=np.asarray(rgba,np.uint8).reshape(height,width,4)[...,:3]
            Image.fromarray(image).save(out/f'view_{i+1}_render.png')
            ids=np.asarray(labels).reshape(height,width)&((1<<24)-1)
            sim_depth=far*near/(far-(far-near)*np.asarray(buffer).reshape(height,width))
            one={'view_index':i+1,'objects':{}}
            for name,body in [('box',box_id),('cup',cup_id)]:
                seen=polygon_mask((width,height),config['views'][i][f'{name}_polygon'])
                rendered=ids==body
                intersection=seen&rendered
                union=seen|rendered
                compare=intersection&(real>.2)&(real<2.)
                difference=np.abs(real[compare]-sim_depth[compare])
                one['objects'][name]={
                    'rendered_pixels':int(rendered.sum()),
                    'observed_polygon_pixels':int(seen.sum()),
                    'mask_iou':float(intersection.sum()/union.sum()) if union.any() else 0.,
                    'depth_compared_pixels':int(compare.sum()),
                    'depth_abs_p50_m':float(np.percentile(difference,50)) if len(difference) else None,
                    'depth_abs_p90_m':float(np.percentile(difference,90)) if len(difference) else None}
            metrics.append(one)
        result={'geometry':str(geometry_path),'views':metrics,
                'box_cup_collision_clearance_m':clearance,
                'scope':'Static table, measured-size box and solid-cylinder cup proxy; no robot-base calibration.',
                'limits':'Manual observed polygons; approximate principal point; cup interior and unseen backs omitted.'}
        (out/'result.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
        print(json.dumps(result,indent=2))
        print(out)
    finally:
        p.disconnect()


if __name__=='__main__':
    main()
