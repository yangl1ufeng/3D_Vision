"""Render measured multi-view Gemini 2 PLY as a static 3D preview."""
import argparse
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import trimesh


def sampled(points, colors, limit, seed):
    if len(points)<=limit:
        return points,colors
    ids=np.random.default_rng(seed).choice(len(points),limit,replace=False)
    return points[ids],colors[ids]


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('ply',type=Path)
    args=parser.parse_args()
    ply=args.ply.resolve()
    cloud=trimesh.load(ply,process=False)
    xyz=np.asarray(cloud.vertices)
    rgb=np.asarray(cloud.colors)[:,:3]/255.
    fig=plt.figure(figsize=(15,5.7),dpi=150)
    p,c=sampled(xyz,rgb,24000,1000)
    ax=fig.add_subplot(131,projection='3d')
    ax.scatter(p[:,0],p[:,1],p[:,2],c=c,s=.7,alpha=.75,rasterized=True)
    ax.set(xlim=(-.5,.65),ylim=(.03,.85),zlim=(-.025,.28),
           xlabel='table X (m)',ylabel='table Y (m)',zlabel='height (m)',
           title='Measured desk | oblique')
    ax.set_box_aspect((1.15,.82,.3),zoom=.86)
    ax.view_init(elev=24,azim=-64)
    focus=((xyz[:,0]>-.34)&(xyz[:,0]<.36)&
           (xyz[:,1]>.24)&(xyz[:,1]<.68)&
           (xyz[:,2]>-.01)&(xyz[:,2]<.25))
    p,c=sampled(xyz[focus],rgb[focus],22000,1001)
    ax=fig.add_subplot(132,projection='3d')
    ax.scatter(p[:,0],p[:,1],p[:,2],c=c,s=1.1,alpha=.8,rasterized=True)
    ax.set(xlim=(-.34,.36),ylim=(.24,.68),zlim=(-.01,.25),
           xlabel='table X (m)',ylabel='table Y (m)',zlabel='height (m)',
           title='Measured box + cup | close view')
    ax.set_box_aspect((.7,.44,.27),zoom=.94)
    ax.view_init(elev=22,azim=-64)
    ax=fig.add_subplot(133)
    sc=ax.scatter(p[:,0],p[:,1],c=np.clip(p[:,2],0,.22),s=1.1,
                  cmap='turbo',vmin=0,vmax=.22,rasterized=True)
    ax.set(xlim=(-.34,.36),ylim=(.24,.68),xlabel='table X (m)',
           ylabel='table Y (m)',title='Top view | height color')
    ax.set_aspect('equal')
    fig.colorbar(sc,ax=ax,shrink=.7,label='measured height above table (m)')
    fig.suptitle('Gemini 2 | two aligned RGB-D views | measured points only')
    fig.tight_layout()
    output=ply.with_name('multi_view_pointcloud_preview.png')
    fig.savefig(output,bbox_inches='tight')
    plt.close(fig)
    print(output)
    figure=plt.figure(figsize=(10,7.5),dpi=170)
    main=figure.add_subplot(111,projection='3d')
    objects=((xyz[:,0]>-.34)&(xyz[:,0]<.13)&
             (xyz[:,1]>.25)&(xyz[:,1]<.59)&
             (xyz[:,2]>-.01)&(xyz[:,2]<.25))
    q,r=sampled(xyz[objects],rgb[objects],45000,1002)
    main.scatter(q[:,0],q[:,1],q[:,2],c=r,s=1.1,alpha=.82,rasterized=True)
    main.set(xlim=(-.34,.13),ylim=(.25,.59),zlim=(-.01,.25),
             xlabel='table X (m)',ylabel='table Y (m)',zlabel='height (m)')
    main.set_box_aspect((.47,.34,.27),zoom=.98)
    main.view_init(elev=25,azim=-67)
    figure.suptitle('Measured box and cup | two Gemini 2 viewpoints | no filled-in surfaces')
    large=ply.with_name('box_cup_3d_preview.png')
    figure.savefig(large,bbox_inches='tight')
    plt.close(figure)
    print(large)


if __name__=='__main__':
    main()
