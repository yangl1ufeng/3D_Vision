"""Small, recorded compatibility edits; no changes to the selected model math."""
from pathlib import Path
ROOT = Path(__file__).resolve().parent
REPO = ROOT/'third_party/GraspGen'
for name in ['generator', 'discriminator']:
    p = REPO/f'grasp_gen/models/{name}.py'
    s = p.read_text()
    s = s.replace('\nfrom grasp_gen.models.ptv3.ptv3 import PointTransformerV3\n','\n')
    for condition in ['elif obs_backbone == "ptv3":', 'elif self.obs_backbone == "ptv3":']:
        if condition+'\n            from grasp_gen.models.ptv3.ptv3 import PointTransformerV3' not in s:
            s = s.replace(condition,condition+'\n            from grasp_gen.models.ptv3.ptv3 import PointTransformerV3')
    p.write_text(s)
p = REPO/'pointnet2_ops/setup.py'
s = p.read_text().replace('"cxx": ["-O3"]','"cxx": ["/O2"]')
if 'library_dirs=' not in s:
    s = s.replace('include_dirs=[osp.join(this_dir, _ext_src_root, "include")],',
                  'include_dirs=[osp.join(this_dir, _ext_src_root, "include")],\n            library_dirs=[osp.join(__import__("os").environ["CUDA_HOME"], "lib")],')
if 'extra_link_args=' not in s:
    s = s.replace('sources=_ext_sources,', 'sources=_ext_sources,\n            extra_link_args=["/MANIFEST:NO"],')
p.write_text(s)
for p in (REPO/'pointnet2_ops/pointnet2_ops/_ext-src').rglob('*'):
    if p.suffix in ['.cpp','.cu','.h']:
        s = p.read_text()
        s = s.replace('#include <ATen/cuda/CUDAContext.h>', '#include <c10/cuda/CUDAStream.h>')
        s = s.replace('#include <ATen/cuda/Exceptions.h>', '#include <c10/cuda/CUDAException.h>')
        s = s.replace('at::cuda::getCurrentCUDAStream()', 'c10::cuda::getCurrentCUDAStream()')
        s = s.replace('at::cuda::OptionalCUDAGuard', 'c10::cuda::OptionalCUDAGuard')
        p.write_text(s)
