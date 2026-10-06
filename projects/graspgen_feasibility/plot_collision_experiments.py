"""Render audit charts from saved CSV summaries, without rerunning inference."""
import argparse
import csv
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def main(folder):
    with (folder/'path_summary.csv').open(encoding='utf-8-sig') as stream:
        path = list(csv.DictReader(stream))
    with (folder/'baseline_summary.csv').open(encoding='utf-8-sig') as stream:
        baseline = list(csv.DictReader(stream))
    plt.rcParams.update({'font.size': 10, 'axes.spines.top': False, 'axes.spines.right': False})
    fig, axes = plt.subplots(1, 3, figsize=(15, 5), layout='constrained')
    names = ['Final only', 'Approach only', 'Combined']
    for ax, split, total in zip(axes[:2], ['all', 'held_out'], [216, 108]):
        rows = [r for r in path if r['split'] == split]
        x = np.arange(len(rows))
        safe = [int(r['retained_reference_clear']) for r in rows]
        missed = [int(r['false_accept']) for r in rows]
        rejected = [int(r['false_reject']) for r in rows]
        ax.bar(x-.22, safe, .22, label='Retained, reference clear', color='#2a8b77')
        ax.bar(x, missed, .22, label='Retained, reference collision', color='#d34f4f')
        ax.bar(x+.22, rejected, .22, label='Rejected, reference clear', color='#dcaa41')
        for offset, counts in [(-.22, safe), (0, missed), (.22, rejected)]:
            for j, count in enumerate(counts):
                ax.text(j+offset, count+1, str(count), ha='center', fontsize=9)
        ax.set_xticks(x, names, rotation=12)
        ax.set_ylim(0, 73)
        ax.set_ylabel('Candidate evaluations')
        ax.set_title(f'{"All scenes" if split == "all" else "Held-out templates"}: {total} evaluations')
        ax.grid(axis='y', alpha=.15)
        ax.set_axisbelow(True)
    axes[0].legend(loc='upper right', fontsize=8)
    rows = [r for r in baseline if r['split']=='all' and r['background_points']=='8192' and r['surface_points']=='2000']
    ax = axes[2]
    x = [float(r['margin_m'])*1000 for r in rows]
    ax.plot(x, [int(r['false_accept']) for r in rows], 'o-', color='#d34f4f', label='Missed final collisions')
    ax.plot(x, [int(r['false_reject']) for r in rows], 'o-', color='#dcaa41', label='Rejected clear finals')
    ax.set_xticks(x)
    ax.set_xlabel('Baseline distance threshold (mm)')
    ax.set_ylabel('Candidate evaluations')
    ax.set_title('Final-only threshold tradeoff\n8,192 background / 2,000 gripper points')
    ax.grid(alpha=.15)
    ax.legend(fontsize=8)
    fig.suptitle('Synthetic collision audit | 72 fixed candidates x 3 sampling seeds\nOriginal mesh + solid-box reference; counts do not measure real grasp success', fontsize=13)
    fig.savefig(folder/'collision_results.png', dpi=180)
    plt.close(fig)
    print(folder/'collision_results.png')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('folder', type=Path)
    main(parser.parse_args().folder)
