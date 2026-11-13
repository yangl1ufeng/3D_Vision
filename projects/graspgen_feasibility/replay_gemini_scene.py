"""Build a static PyBullet table-and-box scene from a fitted Gemini 2 snapshot."""
import argparse
from datetime import datetime
import json
import math
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw
import pybullet as p

ROOT = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('fit', type=Path, help='result.json from fit_static_scene.py')
    args = parser.parse_args()
    fit = json.loads(args.fit.resolve().read_text(encoding='utf-8'))
    source = Path(fit['source'])
    capture = json.loads((source/'report.json').read_text(encoding='utf-8'))
    width, height = capture['depth_shape'][1], capture['depth_shape'][0]
    K = capture['aligned_depth_intrinsic']
    box = fit['box_pose_table']
    center = np.asarray(box['center_m'], float)
    half = np.asarray(box['half_extents_m'], float)
    if np.any(half <= 0) or abs(center[2]-half[2]) > 1e-5:
        raise ValueError('Box must rest on the fitted z=0 table plane')
    T_camera_table = np.asarray(fit['T_camera_table'], float)
    T_table_camera = np.linalg.inv(T_camera_table)
    eye = T_table_camera[:3, 3]
    R = T_table_camera[:3, :3]
    forward = R @ np.array([0., 0., 1.])
    up = R @ np.array([0., -1., 0.])
    out = ROOT/'runs/gemini_sim_scene'/datetime.now().strftime('%Y%m%d_%H%M%S_%f')
    out.mkdir(parents=True)
    p.connect(p.DIRECT)
    try:
        plane_shape = p.createCollisionShape(p.GEOM_PLANE)
        plane_visual = p.createVisualShape(p.GEOM_BOX, halfExtents=[.7, .7, .005],
                                           rgbaColor=[.87,.87,.84,1])
        table_id = p.createMultiBody(0, plane_shape, plane_visual, [0,0,-.005])
        collision = p.createCollisionShape(p.GEOM_BOX, halfExtents=half.tolist())
        visual = p.createVisualShape(p.GEOM_BOX, halfExtents=half.tolist(),
                                     rgbaColor=[.67,.48,.28,1])
        yaw = float(box['yaw_rad'])
        box_id = p.createMultiBody(0, collision, visual, center.tolist(),
                                   p.getQuaternionFromEuler([0,0,yaw]))
        near, far = .05, 3.
        view = p.computeViewMatrix(eye.tolist(), (eye+forward).tolist(), up.tolist())
        vfov = math.degrees(2*math.atan(height/(2*K['fy'])))
        projection = p.computeProjectionMatrixFOV(vfov, width/height, near, far)
        _, _, rgba, _, labels = p.getCameraImage(width, height, view, projection,
                                                renderer=p.ER_TINY_RENDERER)
        rgb = np.asarray(rgba, np.uint8).reshape(height, width, 4)[..., :3]
        Image.fromarray(rgb).save(out/'render.png')
        ids = np.asarray(labels).reshape(height,width) & ((1<<24)-1)
        rendered_mask = ids == box_id
        observed_image = Image.new('L', (width,height), 0)
        polygon = fit.get('box_polygon_xy')
        if polygon:
            ImageDraw.Draw(observed_image).polygon(list(zip(polygon[::2], polygon[1::2])), fill=1)
        observed_mask = np.asarray(observed_image, np.uint8)
        intersection = int((rendered_mask & observed_mask.astype(bool)).sum())
        union = int((rendered_mask | observed_mask.astype(bool)).sum())
        ys, xs = np.where(rendered_mask)
        render_bbox = [int(xs.min()),int(ys.min()),int(xs.max()+1),int(ys.max()+1)] if len(xs) else None
        result = {'source_fit': str(args.fit.resolve()), 'table_frame': 'fitted plane z=0, not robot base',
                  'box_dimensions_m': (half*2).tolist(), 'box_position_table_m': center.tolist(),
                  'box_yaw_table_rad': yaw, 'dimensions_source': box.get('dimensions_source','depth estimate'),
                  'camera_render_bbox_xyxy': render_bbox,
                  'rendered_box_pixels': int(rendered_mask.sum()),
                  'observed_polygon_pixels': int(observed_mask.sum()),
                  'box_mask_iou': intersection/union if union else 0.,
                  'projection_note': 'FOV uses fy and centred principal point; slight mismatch to real K is expected.',
                  'scope': 'Static fitted table and box only; room, unseen surfaces and robot-base transform not modeled.'}
        (out/'result.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
        print(json.dumps(result, indent=2))
        print(out)
    finally:
        p.disconnect()


if __name__ == '__main__':
    main()
