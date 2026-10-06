"""Target segmentation; explicit user box or an explicitly supplied binary mask."""
import sys
import numpy as np
from .config import project_path
from .scene_io import load_array
from .types import Segmentation


class SAM2Segmenter:
    def __init__(self, cfg):
        import torch
        repo = project_path(cfg['sam2_repo'])
        self.checkpoint = project_path(cfg['checkpoint'])
        if not repo.is_dir() or not self.checkpoint.is_file():
            raise FileNotFoundError('SAM2 repository/checkpoint missing; see README setup paths')
        if cfg['precision'] not in ['bfloat16', 'float32']:
            raise ValueError('SAM2 precision must be bfloat16 or float32')
        if cfg['precision'] == 'bfloat16' and not torch.cuda.is_bf16_supported():
            raise RuntimeError('selected GPU does not support BF16')
        sys.path.insert(0, str(repo))
        from sam2.build_sam import build_sam2
        from sam2.sam2_image_predictor import SAM2ImagePredictor
        model = build_sam2(cfg['model_config'], str(self.checkpoint), device='cuda', apply_postprocessing=False)
        self.predictor = SAM2ImagePredictor(model)
        self.precision = cfg['precision']

    def segment(self, scene):
        import torch
        from contextlib import nullcontext
        box = np.asarray(scene.target['box_xyxy'], dtype=np.float32)
        precision = torch.autocast('cuda', dtype=torch.bfloat16) if self.precision == 'bfloat16' else nullcontext()
        with torch.inference_mode(), precision:
            self.predictor.set_image(scene.rgb)
            masks, scores, _ = self.predictor.predict(box=box, multimask_output=True)
        if not np.isfinite(scores).all():
            raise RuntimeError('SAM2 returned nonfinite quality scores')
        best = int(np.argmax(scores))
        return validate_segmentation(Segmentation(masks[best].astype(bool), 'sam2', float(scores[best]),
                                    {'box_xyxy': box.tolist(), 'selection': 'highest SAM2 predicted quality',
                                     'prompt_source': scene.target.get('prompt_source', 'input_manifest'),
                                     'prompt_origin': scene.target.get('prompt_origin'),
                                     'quality_note': 'SAM2 predicted mask quality is not grasp success probability'}), scene)


def provided_segmentation(scene):
    mask = load_array(scene.target['mask_path'])
    if not np.isfinite(mask).all() or not np.isin(mask, [0, 1, 255]).all():
        raise ValueError('provided target mask must be binary (0/1 or 0/255)')
    return validate_segmentation(Segmentation(mask.astype(bool), 'provided_mask', None,
                                            {'source': scene.target['mask_path']}), scene)


def validate_segmentation(result, scene):
    if result.mask.shape != scene.rgb.shape[:2]:
        raise ValueError('target mask must match RGB resolution')
    if not result.mask.any():
        raise ValueError('segmentation produced an empty target mask')
    return result
