"""Render saved visibility and fusion comparisons without rerunning experiments."""
import argparse
import csv
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch, Rectangle


def main(root):
    plt.rcParams.update({'font.size': 10, 'axes.spines.top': False, 'axes.spines.right': False})
    colors = ['#dce0e6', '#91cbb6', '#dc8e64', '#b37bc3']
    with np.load(root/'visibility/visibility_arrays.npz', allow_pickle=False) as data:
        fig, axes = plt.subplots(1, 2, figsize=(11, 5.6), layout='constrained')
        vertices = data['gripper_vertices'] + data['gripper_pose'][:3, 3]
        for ax, key, title in zip(axes, ['single_states', 'multi_states'], ['One view: behind the blocker is unknown', 'Add side view: free corridor is observed']):
            ax.pcolormesh(data['query_x'], data['query_z'], data[key], cmap=ListedColormap(colors), vmin=0, vmax=3, shading='nearest')
            ax.scatter(vertices[:, 0], vertices[:, 2], s=.5, color='#263746', alpha=.5)
            ax.scatter(vertices[:, 0], vertices[:, 2]-.1, s=.3, color='#263746', alpha=.2)
            ax.add_patch(Rectangle((-.15, .53), .30, .04, fill=False, edgecolor='#614939', linestyle='--', linewidth=1.2))
            ax.text(0, .495, 'Occluder (reference outline)', ha='center', fontsize=8)
            ax.annotate('', xy=(.22, .72), xytext=(.22, .62), arrowprops={'arrowstyle': '->', 'lw': 2, 'color': '#263746'})
            ax.text(.22, .75, '100 mm\napproach', ha='center', fontsize=9)
            ax.set(xlabel='World x (m)', ylabel='World z (m)', title=title, xlim=(-.3, .3), ylim=(.2, 1.15))
            ax.set_aspect('equal')
        fig.legend(handles=[Patch(color=c, label=t) for c, t in zip(colors, ['Unknown', 'Observed free', 'Observed surface', 'Conflicting evidence'])], loc='outside lower center', ncol=4)
        fig.suptitle('Synthetic depth visibility | y = 0 slice\nFront camera at z = 0; dark outlines show final and pregrasp gripper geometry', fontsize=12)
        fig.savefig(root/'visibility_comparison.png', dpi=180)
        plt.close(fig)
    with (root/'fusion/summary.csv').open(encoding='utf-8-sig') as stream:
        rows = list(csv.DictReader(stream))
    fig = plt.figure(figsize=(14, 9), layout='constrained')
    colors = ['#347ac0', '#d28132', '#48946c']
    for col, n in enumerate([1, 2, 3]):
        ax = fig.add_subplot(2, 3, col+1, projection='3d')
        with np.load(root/f'fusion/A/known_{n}/fusion_diagnostic.npz') as data:
            mask = data['observed_target_mask']
            pts, source = data['observed_points_world_m'][mask], data['source_view_index'][mask]
            for i in range(n):
                subset = pts[source==i][::3]
                ax.scatter(*subset.T, s=1, alpha=.5, color=colors[i], label=f'View {i+1}')
        row = next(r for r in rows if r['layout']=='A' and r['case']==f'known_{n}')
        ax.set(title=f'{n} known view(s): {100*float(row["target_coverage_fraction"]):.1f}% coverage',
               xlim=(-.08, .08), ylim=(-.07, .07), zlim=(0, .20), xlabel='x (m)', ylabel='y (m)', zlabel='z (m)')
        ax.view_init(elev=25, azim=-55)
        ax.legend(fontsize=8)
    for col, case, title in [(1, 'overlap_3_raw', 'Wrong poses: double contours, rejected'), (2, 'overlap_3_refined', 'Shared faces: refined and accepted (B)')]:
        ax = fig.add_subplot(2, 3, 3+col)
        with np.load(root/f'fusion/B/{case}/fusion_diagnostic.npz') as data:
            select = data['observed_target_mask']
            pts, src = data['observed_points_world_m'][select], data['source_view_index'][select]
            for i in range(3):
                p = pts[src==i][::2]
                ax.scatter(p[:, 0]*1000, p[:, 1]*1000, s=1.5, alpha=.4, color=colors[i])
        ax.set(title=title, xlabel='World x (mm)', ylabel='World y (mm)', xlim=(-60, 100), ylim=(-70, 90))
        ax.set_aspect('equal')
        ax.grid(alpha=.15)
    ax = fig.add_subplot(2, 3, 6)
    for layout, color in [('A', colors[0]), ('B', colors[1])]:
        cover = [100*float(next(r for r in rows if r['layout']==layout and r['case']==f'known_{n}')['target_coverage_fraction']) for n in [1,2,3]]
        ax.plot([1,2,3], cover, 'o-', color=color, label=f'Layout {layout}')
    ax.set(title='Target surface coverage', xlabel='Known camera views', ylabel='Coverage at 6 mm tolerance (%)', xticks=[1,2,3], ylim=(0,100))
    ax.grid(alpha=.2)
    ax.legend()
    fig.suptitle('Multi-view pre-experiment | exact synthetic depths, static scene\nTrue box surfaces used for evaluation only; failed registration attempts remain refused', fontsize=13)
    fig.savefig(root/'fusion_comparison.png', dpi=180, bbox_inches='tight', pad_inches=.25)
    plt.close(fig)
    print(root/'visibility_comparison.png')
    print(root/'fusion_comparison.png')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('root', type=Path)
    main(parser.parse_args().root)
