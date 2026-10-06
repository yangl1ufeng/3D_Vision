"""Plots and a failure gallery from saved preparation 5/6 evidence only."""
import argparse
import csv
import json
from pathlib import Path
import numpy as np
from PIL import Image
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle


def table(root):
    with (root/'summary.csv').open(encoding='utf-8-sig') as stream: return list(csv.DictReader(stream))


def robust_plots(root):
    rows=table(root)
    fig,axes=plt.subplots(2,3,figsize=(14,8),layout='constrained')
    families=[('noise',1000,'Depth noise sigma (mm)'),('dropout',100,'Random missing depth (%)'),
              ('hole',100,'Central missing rectangle (% box area)'),('erode',1,'Mask erosion (pixels)'),
              ('dilate',1,'Mask dilation (pixels)'),('box_shift',100,'Box shift (% original box width)')]
    for ax,(family,scale,xlabel) in zip(axes.flat,families):
        layouts=['real_prompt'] if family=='box_shift' else ['A','B']
        for layout in layouts:
            selected=[r for r in rows if r['layout']==layout and r['family'] in [family,'box_clean' if family=='box_shift' else 'clean']]
            levels=sorted({float(r['level']) for r in selected})
            mean,lo,hi=[],[],[]
            for level in levels:
                values=[int(r['eligible'] or 0) for r in selected if float(r['level'])==level]
                mean.append(np.mean(values));lo.append(min(values));hi.append(max(values))
            x=np.asarray(levels)*scale
            ax.plot(x,mean,'o-',label=layout)
            ax.fill_between(x,lo,hi,alpha=.15)
        ax.set(xlabel=xlabel,ylabel='Score-qualified candidates / 100',ylim=(0,100))
        ax.grid(alpha=.2);ax.legend()
    fig.suptitle('Input degradation | 3 seeds, same model and thresholds\nBand: min/max; failed preprocessing has zero output, not a measured zero model score',fontsize=13)
    fig.savefig(root/'robustness_curves.png',dpi=160)
    plt.close(fig)
    fig,axes=plt.subplots(1,5,figsize=(16,4.2),layout='constrained')
    for ax,(kind,level,title) in zip(axes,[('clean',0,'Clean'),('noise',.02,'Noise sigma 20 mm'),('dropout',.75,'75% random dropout'),('erode',6,'Erosion 6 pixels'),('dilate',6,'Dilation 6 pixels')]):
        folder=root/'synthetic/A'/f'{kind}_{level:g}'/'1000'
        depth=np.load(folder/'input/depth.npy')
        mask=np.load(folder/'input/mask.npy')
        result=json.loads((folder/'evaluation.json').read_text())
        view=np.ma.masked_where(depth<=0,depth)
        ax.imshow(view,cmap='viridis',vmin=.35,vmax=.95)
        if mask.any():ax.contour(mask,levels=[.5],colors=['#ff614d'],linewidths=.7)
        state='preprocessing failed' if result['status']!='ok' else 'ran; degraded input' if result['material_degradation'] else 'clean reference'
        ax.set_title(title+'\n'+state);ax.axis('off')
    fig.suptitle('Saved failure examples | depth with target-mask outline (layout A, seed 1000)')
    fig.savefig(root/'degradation_examples.png',dpi=160)
    plt.close(fig)
    source=json.loads((root/'prompt_source.json').read_text())
    rgb=np.array(Image.open(source['asset_paths']['rgb']).convert('RGB'))
    base=np.asarray(source['input_manifest']['target']['box_xyxy'])
    width,height=base[2:]-base[:2]
    x0,y0=np.maximum(base[:2]-[width,height],0).astype(int)
    x1,y1=np.minimum(base[2:]+[2*width,height],[rgb.shape[1],rgb.shape[0]]).astype(int)
    fig,axes=plt.subplots(1,3,figsize=(13,4),layout='constrained')
    for ax,(case,label) in zip(axes,[('box_clean_0','Original box'),('box_shift_0.5','Shift 50% of width'),('box_shift_1','Shift 100% of width')]):
        folder=root/'prompts'/case
        mask=np.load(folder/'sam2_mask.npy')
        out=rgb.copy();out[mask]=(out[mask]*.5+np.array([50,220,110])*.5).astype(np.uint8)
        box=json.loads((folder/'segmentation.json').read_text())['box_xyxy']
        iou=json.loads((folder/'1000/evaluation.json').read_text())['mask_iou']
        ax.imshow(out[y0:y1,x0:x1])
        ax.add_patch(Rectangle((box[0]-x0,box[1]-y0),box[2]-box[0],box[3]-box[1],fill=False,edgecolor='#ff614d',linewidth=1.5))
        ax.set_title(f'{label}\nIoU vs clean SAM2 mask: {iou:.3f}');ax.axis('off')
    fig.suptitle('Actual SAM2 prompt response | same real RGB image; clean mask is not human ground truth')
    fig.savefig(root/'prompt_examples.png',dpi=160)
    plt.close(fig)


def stress_plots(root):
    rows=table(root)
    fig,axes=plt.subplots(2,2,figsize=(13,8),layout='constrained')
    methods=list(dict.fromkeys(r['method'] for r in rows))
    for method in methods:
        selected=[r for r in rows if r['method']==method]
        counts=sorted({int(r['input_points']) for r in selected})
        x=np.asarray(counts)/1e6
        def med(field):
            return [np.median([float(r[field]) for r in selected if int(r['input_points'])==n and r[field]]) if any(r[field] for r in selected if int(r['input_points'])==n) else np.nan for n in counts]
        axes[0,0].plot(x,med('preprocess_seconds'),'o-',label=method)
        axes[0,1].plot(x,med('fixed_100_collision_seconds'),'o-',label=method)
        axes[1,0].plot(x,np.asarray(med('peak_cuda_allocated_mib'))/1024,'o-',label=method)
        loss=[sum(r['sampling_lost_observed_obstacle']=='True' for r in selected if int(r['input_points'])==n)/3 for n in counts]
        axes[1,1].plot(x,loss,'o-',label=method)
    for ax,ylabel,title in zip(axes.flat,['Seconds','Seconds','GiB','Fraction of 3 seeds'],
                               ['Preprocessing (median; includes failed filter attempts)','Sampled collision, fixed 100 poses (median)',
                                'Peak allocated CUDA memory (median)','Thin obstacle lost by reduction (full geometry detects it)']):
        ax.set(xlabel='Raw fused observations (million points)',ylabel=ylabel,title=title)
        ax.grid(alpha=.2);ax.legend(fontsize=8)
    axes[1,1].set_ylim(-.05,1.05)
    fig.suptitle('Point-cloud pressure | 3 views, actual denser rays, 100 candidates\n12 mm target voxels may fail the unchanged outlier filter; retained obstacle does not guarantee model usability',fontsize=13)
    fig.savefig(root/'pressure_comparison.png',dpi=160)
    plt.close(fig)
    lost=next(r for r in rows if r['sampling_lost_observed_obstacle']=='True' and r['status']=='ok')
    control=next(r for r in rows if r['scale']==lost['scale'] and r['seed']==lost['seed'] and r['method']=='voxel4mm32768')
    protocol=json.loads((root/'protocol.json').read_text())
    lower,upper=np.asarray(protocol['thin_post']['lower']),np.asarray(protocol['thin_post']['upper'])
    with np.load(root/f"scale_{lost['scale']}/input.npz") as data:
        full=data['points_world_m'][data['labels']==4]
    fig,axes=plt.subplots(1,2,figsize=(9,5),layout='constrained')
    for ax,row in zip(axes,[lost,control]):
        with np.load(root/row['folder']/'arrays.npz') as data:
            points=data['background_collision_world_m']
            kept=points[np.all((points>=lower-2e-6)&(points<=upper+2e-6),axis=1)]
        ax.scatter(full[:,0]*1000,full[:,2]*1000,s=45,facecolors='none',edgecolors='#607d8b',label='Original depth returns')
        if len(kept):ax.scatter(kept[:,0]*1000,kept[:,2]*1000,s=18,color='#df6f28',label='Retained collision points')
        ax.add_patch(Rectangle((lower[0]*1000,lower[2]*1000),(upper[0]-lower[0])*1000,(upper[2]-lower[2])*1000,fill=False,color='#263238',linestyle='--'))
        ax.set(xlabel='World x (mm)',ylabel='World z (mm)',xlim=(297,303),ylim=(254,286),
               title=f"{row['method']} | retained {len(kept)}/{len(full)}\nProbe detects obstacle: {row['sampled_geometry_probe_detected']}")
        ax.grid(alpha=.2);ax.legend(fontsize=8)
    fig.suptitle('A visible 1 mm-wide obstacle can disappear during point reduction\nThe same probe is a confirmed solid collision; full geometry still detects it')
    fig.savefig(root/'thin_obstacle_loss.png',dpi=170)
    plt.close(fig)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--robustness',type=Path)
    parser.add_argument('--stress',type=Path)
    args=parser.parse_args()
    if args.robustness:robust_plots(args.robustness)
    if args.stress:stress_plots(args.stress)
