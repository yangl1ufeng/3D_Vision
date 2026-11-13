"""Fit a table plane and a single box from one fixed Gemini 2 depth snapshot.

ROIs are explicit pixel coordinates. This is a measured first scene, not an
automatic object detector or a robot/world calibration.
"""
import argparse
from datetime import datetime
import json
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent


def points_in_roi(depth, K, roi, stride=2):
    x0, y0, x1, y1 = roi
    if not (0 <= x0 < x1 <= depth.shape[1] and 0 <= y0 < y1 <= depth.shape[0]):
        raise ValueError(f'Invalid ROI {roi}')
    v, u = np.mgrid[y0:y1:stride, x0:x1:stride]
    z = depth[y0:y1:stride, x0:x1:stride]
    valid = np.isfinite(z) & (z > .2) & (z < 2.)
    xyz = np.column_stack(((u[valid]-K['cx'])*z[valid]/K['fx'],
                           (v[valid]-K['cy'])*z[valid]/K['fy'], z[valid]))
    return xyz.astype(np.float64), float(valid.mean())


def points_in_polygon(depth, K, coordinates):
    if len(coordinates) < 6 or len(coordinates) % 2:
        raise ValueError('Box polygon needs at least three x y corners')
    polygon = np.asarray(coordinates, np.int32).reshape(-1, 2)
    if (polygon < 0).any() or (polygon[:, 0] >= depth.shape[1]).any() or (polygon[:, 1] >= depth.shape[0]).any():
        raise ValueError('Box polygon lies outside the image')
    mask = np.zeros(depth.shape, np.uint8)
    cv2.fillPoly(mask, [polygon], 1)
    v, u = np.nonzero(mask)
    z = depth[v, u]
    valid = np.isfinite(z) & (z > .2) & (z < 2.)
    xyz = np.column_stack(((u[valid]-K['cx'])*z[valid]/K['fx'],
                           (v[valid]-K['cy'])*z[valid]/K['fy'], z[valid]))
    return xyz.astype(np.float64), float(valid.mean())


def fit_plane(points, threshold=.008, iterations=180):
    if len(points) < 500:
        raise ValueError('Insufficient table depth points')
    rng = np.random.default_rng(1000)
    best = np.zeros(len(points), bool)
    for _ in range(iterations):
        a, b, c = points[rng.choice(len(points), 3, replace=False)]
        normal = np.cross(b-a, c-a)
        length = np.linalg.norm(normal)
        if length < 1e-6:
            continue
        normal /= length
        good = np.abs((points-a) @ normal) < threshold
        if good.sum() > best.sum():
            best = good
    if best.sum() < .5*len(points):
        raise ValueError('No dominant table plane; refine table ROIs')
    center = points[best].mean(axis=0)
    _, _, vt = np.linalg.svd(points[best]-center, full_matrices=False)
    normal = vt[-1]
    offset = -normal @ center
    if offset < 0:  # positive height points toward the camera
        normal, offset = -normal, -offset
    error = np.abs(points[best] @ normal + offset)
    return normal, offset, best, float(np.percentile(error, 95))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('snapshot', type=Path)
    parser.add_argument('--table-roi', nargs=4, type=int, action='append', required=True,
                        metavar=('X0','Y0','X1','Y1'))
    selection = parser.add_mutually_exclusive_group(required=True)
    selection.add_argument('--box-roi', nargs=4, type=int,
                           metavar=('X0','Y0','X1','Y1'))
    selection.add_argument('--box-polygon', nargs='+', type=int,
                           help='Consecutive x y pixel coordinates around the visible box')
    parser.add_argument('--measured-box-cm', nargs=3, type=float,
                        metavar=('LENGTH','WIDTH','HEIGHT'),
                        help='Independent ruler measurements, in centimetres')
    args = parser.parse_args()
    source = args.snapshot.resolve()
    report = json.loads((source/'report.json').read_text(encoding='utf-8'))
    depth = np.load(source/'depth_m.npy', allow_pickle=False)
    K = report['aligned_depth_intrinsic']
    table_parts = [points_in_roi(depth, K, roi) for roi in args.table_roi]
    table = np.concatenate([part[0] for part in table_parts])
    normal, offset, inliers, residual = fit_plane(table)
    if args.box_polygon:
        obj, obj_coverage = points_in_polygon(depth, K, args.box_polygon)
    else:
        obj, obj_coverage = points_in_roi(depth, K, args.box_roi, stride=1)
    height = obj @ normal + offset
    raised = obj[(height > .02) & (height < .5)]
    if len(raised) < 500:
        raise ValueError('Too few points above table in box ROI')
    raised_height = raised @ normal + offset
    top = np.percentile(raised_height, 95)
    top_surface = raised[raised_height >= max(.02, top-.015)]
    if len(top_surface) < 500:
        raise ValueError('Too few points on the visible upper box surface')
    # Construct a table-local frame for a rough footprint estimate.
    x_axis = np.array([1., 0., 0.])
    x_axis -= normal * (x_axis @ normal)
    x_axis /= np.linalg.norm(x_axis)
    y_axis = np.cross(normal, x_axis)
    uv = np.column_stack((top_surface @ x_axis, top_surface @ y_axis)).astype(np.float32)
    rectangle = cv2.minAreaRect(uv)
    width, length = rectangle[1]
    corners = cv2.boxPoints(rectangle)
    edges = np.roll(corners, -1, axis=0) - corners
    long_axis = edges[np.argmax(np.linalg.norm(edges, axis=1))]
    long_axis /= np.linalg.norm(long_axis)
    yaw = float(np.arctan2(long_axis[1], long_axis[0]))
    T_camera_table = np.eye(4)
    T_camera_table[:3, :3] = np.column_stack((x_axis, y_axis, normal))
    T_camera_table[:3, 3] = -offset * normal
    result = {'source': str(source), 'table_rois_xyxy': args.table_roi,
              'box_roi_xyxy': args.box_roi, 'box_polygon_xy': args.box_polygon,
              'table_points': len(table), 'table_inliers': int(inliers.sum()),
              'table_roi_valid_fractions': [part[1] for part in table_parts],
              'table_plane_camera': {'normal': normal.tolist(), 'offset_m': float(offset)},
              'table_plane_inlier_p95_error_m': residual,
              'box_roi_valid_fraction': obj_coverage, 'box_points_above_table': len(raised),
              'box_top_surface_points': len(top_surface),
              'box_height_p95_m': float(top),
              'visible_top_footprint_min_area_rect_m': [float(width), float(length)],
              'T_camera_table': T_camera_table.tolist(),
              'box_pose_table': {'center_m': [float(rectangle[0][0]), float(rectangle[0][1]), float(top/2)],
                                 'yaw_rad': yaw,
                                 'half_extents_m': [float(max(width, length)/2),
                                                    float(min(width, length)/2), float(top/2)]},
              'result_scope': 'Visible geometry and manual ROIs; no physical size truth, hidden surface or robot registration.'}
    if args.measured_box_cm:
        measured = np.asarray(args.measured_box_cm, float) / 100
        if not np.isfinite(measured).all() or (measured <= 0).any():
            raise ValueError('Measured box dimensions must be positive and finite')
        estimate = np.array([max(width, length), min(width, length), top])
        result['measured_dimensions_m'] = measured.tolist()
        result['estimated_dimensions_m'] = estimate.tolist()
        result['absolute_dimension_error_m'] = abs(estimate-measured).tolist()
        result['box_pose_table']['half_extents_m'] = (measured/2).tolist()
        result['box_pose_table']['center_m'][2] = float(measured[2]/2)
        result['box_pose_table']['dimensions_source'] = 'independent ruler measurements'
        result['result_scope'] = 'Manual image ROIs and ruler dimensions; camera/table frame only, not robot registration.'
    out = ROOT/'runs/gemini_scene_fit'/datetime.now().strftime('%Y%m%d_%H%M%S_%f')
    out.mkdir(parents=True)
    (out/'result.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
    print(json.dumps(result, indent=2))
    print(out)


if __name__ == '__main__':
    main()
