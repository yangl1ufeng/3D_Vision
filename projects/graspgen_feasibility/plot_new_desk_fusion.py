"""Compare measured single-view and two-view object point clouds."""
import argparse
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from fuse_new_desk_views import load_labeled_ply


def draw(ax, xyz, rgb, title, top=False):
    if top:
        ax.scatter(xyz[:, 0], xyz[:, 1], c=rgb/255, s=1.4, alpha=.85, rasterized=True)
        ax.set(xlim=(-.13, .18), ylim=(-.10, .19), xlabel='table X (m)',
               ylabel='table Y (m)', title=title)
        ax.set_aspect('equal')
    else:
        ax.scatter(xyz[:, 0], xyz[:, 1], xyz[:, 2], c=rgb/255,
                   s=2, alpha=.85, depthshade=False, rasterized=True)
        ax.set(xlim=(-.13, .18), ylim=(-.10, .19), zlim=(0, .12),
               xlabel='X (m)', ylabel='Y (m)', zlabel='Z (m)', title=title)
        ax.set_box_aspect((.31, .29, .12), zoom=.9)
        ax.view_init(elev=22, azim=-65)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('first_ply', type=Path)
    parser.add_argument('fused_ply', type=Path)
    args = parser.parse_args()
    first_xyz, first_rgb, _ = load_labeled_ply(args.first_ply)
    fused_xyz, fused_rgb, _ = load_labeled_ply(args.fused_ply)
    fig = plt.figure(figsize=(14, 10), dpi=150)
    draw(fig.add_subplot(221, projection='3d'), first_xyz, first_rgb,
         f'First view | {len(first_xyz):,} measured points')
    draw(fig.add_subplot(222, projection='3d'), fused_xyz, fused_rgb,
         f'Two views | {len(fused_xyz):,} measured points')
    draw(fig.add_subplot(223), first_xyz, first_rgb, 'First view | overhead', top=True)
    draw(fig.add_subplot(224), fused_xyz, fused_rgb, 'Two views | overhead', top=True)
    fig.suptitle('Gemini 2 tabletop objects: mug, earbuds case, figurine\n'
                 'Visible RGB-D samples only; no generated hidden surfaces', fontsize=16)
    fig.tight_layout()
    output = args.fused_ply.with_name('two_view_pointcloud_preview.png')
    fig.savefig(output, bbox_inches='tight')
    plt.close(fig)
    print(output)


if __name__ == '__main__':
    main()
