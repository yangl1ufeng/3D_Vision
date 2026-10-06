"""One official diffusion call, retaining every pose and score for audit."""
from pathlib import Path
import numpy as np
from .config import project_path
from .runtime import set_seed
from .types import Grasps


class GraspGenerator:
    def __init__(self, cfg):
        from grasp_gen.grasp_server import GraspGenSampler, load_grasp_cfg
        if cfg['precision'] != 'float32':
            raise ValueError('this GraspGen adapter is verified for float32 only')
        self.model_config = project_path(cfg['gripper_config'])
        model = load_grasp_cfg(str(self.model_config))
        if model.data.gripper_name != cfg['expected_gripper']:
            raise ValueError('gripper configuration does not match expected_gripper')
        self.sampler = GraspGenSampler(model)
        self.gripper_name = model.data.gripper_name
        self.weight_paths = {'generator': Path(model.eval.checkpoint), 'discriminator': Path(model.discriminator.checkpoint)}
        self.steps = int(model.diffusion.num_diffusion_iters_eval)

    def generate(self, target_world_m, count, seed):
        import torch
        set_seed(seed)
        pc = torch.as_tensor(target_world_m, dtype=torch.float32, device='cuda')
        with torch.inference_mode():
            output = self.sampler.sample(pc, threshold=0., num_grasps=count, remove_outliers=False)
            poses, scores = output[:2]
            if len(poses) == 0:
                p = np.empty((0, 4, 4), dtype=np.float32)
                s = np.empty(0, dtype=np.float32)
            else:
                poses[:, 3, 3] = 1.  # Official run_inference wrapper also sets this row element.
                p, s = poses.cpu().numpy(), scores.cpu().numpy()
        # sample(threshold=0) is the upstream interface, without retries or top-k.
        if len(p) != count:
            raise RuntimeError(f'upstream returned {len(p)} raw grasps, expected {count}; candidate accounting failed')
        return Grasps(p, s, {'gripper': self.gripper_name, 'diffusion_steps': self.steps,
                            'model_config': str(self.model_config), 'seed': seed,
                            'candidate_budget': count, 'precision': 'float32',
                            'pose_frame': 'graspgen_gripper', 'output_unit': 'm',
                            'note': 'Model scores are ranking scores, not measured physical grasp success.'})
