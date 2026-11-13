"""Back-project a saved Gemini 2 depth image to a camera-frame PLY point cloud."""
import argparse
from datetime import datetime
import hashlib
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent


def sha256(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('capture', nargs='?', type=Path,
                        help='Capture directory; defaults to the newest Gemini 2 single or fused capture')
    parser.add_argument('--stride', type=int, default=3, help='Keep one in N pixels per axis')
    parser.add_argument('--near', type=float, default=.2, help='Nearest exported depth in metres')
    parser.add_argument('--far', type=float, default=2., help='Farthest exported depth in metres')
    args = parser.parse_args()
    if args.stride < 1 or not 0 < args.near < args.far:
        parser.error('require stride >= 1 and 0 < near < far')
    if args.capture is None:
        captures = sorted([*(ROOT/'runs/gemini2_probe').glob('*/report.json'),
                           *(ROOT/'runs/gemini_static').glob('*/report.json')],
                          key=lambda path: path.stat().st_mtime)
        if not captures:
            parser.error('No Gemini 2 capture found; run start_gemini2_capture.cmd first')
        source = captures[-1].parent.resolve()
    else:
        source = args.capture.resolve()
    report_path, depth_path = source/'report.json', source/'depth_m.npy'
    camera = json.loads(report_path.read_text(encoding='utf-8'))
    depth = np.load(depth_path, allow_pickle=False)
    h, w = depth.shape
    if [h, w] != camera['depth_shape']:
        raise ValueError('Depth dimensions differ from capture report')
    if camera['aligned_depth_distortion']['model'].split('.')[-1] != 'BROWN_CONRADY_K6':
        raise ValueError('Unexpected depth distortion model; inspect calibration first')
    if any(abs(camera['aligned_depth_distortion'][key]) > 1e-7
           for key in ('k1','k2','k3','k4','k5','k6','p1','p2')):
        raise ValueError('Aligned depth has nonzero distortion; undistort before projection')
    K = camera['aligned_depth_intrinsic']
    if (K['width'], K['height']) != (w, h) or K['fx'] <= 0 or K['fy'] <= 0:
        raise ValueError('Invalid aligned depth intrinsics')
    sampled = depth[::args.stride, ::args.stride]
    v, u = np.mgrid[0:h:args.stride, 0:w:args.stride]
    mask = np.isfinite(sampled) & (sampled >= args.near) & (sampled <= args.far)
    z = sampled[mask]
    if len(z) < 100:
        raise ValueError('Too few valid depth samples in selected range')
    xyz = np.empty((len(z), 3), dtype='<f4')
    xyz[:, 0] = (u[mask] - K['cx']) * z / K['fx']
    xyz[:, 1] = (v[mask] - K['cy']) * z / K['fy']
    xyz[:, 2] = z
    out = ROOT/'runs/gemini_pointcloud'/datetime.now().strftime('%Y%m%d_%H%M%S_%f')
    out.mkdir(parents=True)
    ply = out/'visible_surface_camera_m.ply'
    header = (f'ply\nformat binary_little_endian 1.0\n'
              f'comment Gemini 2 visible depth only; OpenCV camera frame, metres\n'
              f'element vertex {len(xyz)}\n'
              'property float x\nproperty float y\nproperty float z\nend_header\n')
    with ply.open('wb') as stream:
        stream.write(header.encode('ascii'))
        stream.write(xyz.tobytes())
    result = {'passed': True, 'input': str(source),
              'input_report_sha256': sha256(report_path), 'input_depth_sha256': sha256(depth_path),
              'output': str(ply), 'output_sha256': sha256(ply),
              'point_count': len(xyz), 'stride': args.stride,
              'depth_range_m': [args.near, args.far],
              'xyz_min_m': xyz.min(axis=0).tolist(), 'xyz_max_m': xyz.max(axis=0).tolist(),
              'frame': 'OpenCV camera: x right, y down, z forward',
              'limitations': 'Visible surfaces only. No table/world calibration, target separation, hidden-surface completion or collision mesh.'}
    (out/'result.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
