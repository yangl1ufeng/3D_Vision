"""Render a measured Gemini 2 PLY as a 3D point preview without filling gaps."""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import trimesh


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('pointcloud', type=Path)
    parser.add_argument('fit', type=Path, help='Fitted table transform for view orientation')
    args = parser.parse_args()
    cloud_path = args.pointcloud.resolve()
    cloud = trimesh.load(cloud_path, process=False)
    xyz = np.asarray(cloud.vertices, np.float64)
    fit = json.loads(args.fit.read_text(encoding='utf-8'))
    T = np.asarray(fit['T_camera_table'], np.float64)
    desk = (xyz - T[:3, 3]) @ T[:3, :3]
    # Viewing crop only. The original PLY remains unchanged and includes all
    # retained depth points in the camera frame.
    keep = ((desk[:, 0] > -.4) & (desk[:, 0] < .55) &
            (desk[:, 1] > .05) & (desk[:, 1] < .7) &
            (desk[:, 2] > -.02) & (desk[:, 2] < .18))
    desk = desk[keep]
    if len(desk) < 1000:
        raise ValueError('Viewing crop contains too few points')
    rng = np.random.default_rng(1000)
    if len(desk) > 20000:
        desk = desk[rng.choice(len(desk), 20000, replace=False)]
    fig = plt.figure(figsize=(13, 5.8), dpi=160)
    ax = fig.add_subplot(121, projection='3d')
    colors = np.clip(desk[:, 2], 0, .09)
    ax.scatter(desk[:, 0], desk[:, 1], desk[:, 2], c=colors, s=.5,
               alpha=.65, cmap='turbo', vmin=0, vmax=.09, rasterized=True)
    ax.set(xlabel='table X (m)', ylabel='table Y (m)', zlabel='height (m)',
           xlim=(-.35,.55), ylim=(.05,.7), zlim=(-.02,.16))
    ax.view_init(elev=28, azim=-60)
    ax.set_box_aspect((.9,.65,.18), zoom=.88)
    ax.set_title('Measured depth points: oblique view')
    top = fig.add_subplot(122)
    sc = top.scatter(desk[:, 0], desk[:, 1], c=colors, s=.6, alpha=.7,
                     cmap='turbo', vmin=0, vmax=.09, rasterized=True)
    top.set(xlabel='table X (m)', ylabel='table Y (m)',
            xlim=(-.35,.55), ylim=(.05,.7), title='Measured depth points: top view')
    top.set_aspect('equal')
    fig.colorbar(sc, ax=top, label='height above fitted table (m)', shrink=.75)
    fig.suptitle('Gemini 2 fixed-camera depth fusion | visible surfaces only')
    fig.tight_layout()
    output = cloud_path.with_name('pointcloud_preview.png')
    fig.savefig(output, bbox_inches='tight')
    plt.close(fig)
    print(output)


if __name__ == '__main__':
    main()
