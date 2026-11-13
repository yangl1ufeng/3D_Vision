"""Fuse registered measured RGB-D points into one colored desk PLY."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image


def digest(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda:f.read(1024*1024),b''):
            h.update(chunk)
    return h.hexdigest()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('registration',type=Path)
    parser.add_argument('--stride',type=int,default=2)
    parser.add_argument('--voxel-mm',type=float,default=3)
    args=parser.parse_args()
    if args.stride<1 or args.voxel_mm<=0:
        parser.error('stride and voxel size must be positive')
    reg_path=args.registration.resolve()
    reg=json.loads(reg_path.read_text(encoding='utf-8'))
    if reg.get('passed') is not True:
        raise ValueError('Registration failed its quality gate; refuse to merge views')
    all_xyz=[]; all_rgb=[]; all_view=[]; counts=[]
    for i,source in enumerate(reg['views']):
        source=Path(source)
        report=json.loads((source/'report.json').read_text(encoding='utf-8'))
        depth=np.load(source/'depth_m.npy',allow_pickle=False)
        rgb=np.asarray(Image.open(source/'rgb.png').convert('RGB'))
        if depth.shape!=rgb.shape[:2]:
            raise ValueError('RGB and depth dimensions differ')
        K=report['aligned_depth_intrinsic']
        v,u=np.mgrid[0:depth.shape[0]:args.stride,0:depth.shape[1]:args.stride]
        z=depth[::args.stride,::args.stride]
        valid=np.isfinite(z)&(z>.2)&(z<2.)
        u,v,z=u[valid],v[valid],z[valid]
        xyz=np.column_stack(((u-K['cx'])*z/K['fx'],
                             (v-K['cy'])*z/K['fy'],z)).astype(np.float64)
        T_camera_table=np.asarray(reg['T_view_camera_table'][i])
        local=(xyz-T_camera_table[:3,3])@T_camera_table[:3,:3]
        T_ref_table=np.asarray(reg['T_reference_table_view_table'][i])
        ref=local@T_ref_table[:3,:3].T+T_ref_table[:3,3]
        # Explicit scene crop around the common tabletop, leaving the original
        # depth data untouched. Unknown and out-of-range regions stay absent.
        keep=((ref[:,0]>-.55)&(ref[:,0]<.65)&
              (ref[:,1]>.03)&(ref[:,1]<.85)&
              (ref[:,2]>-.025)&(ref[:,2]<.28))
        all_xyz.append(ref[keep].astype('<f4'))
        all_rgb.append(rgb[v[keep],u[keep]])
        all_view.append(np.full(keep.sum(),i,np.uint8))
        counts.append(int(keep.sum()))
    xyz=np.concatenate(all_xyz)
    colors=np.concatenate(all_rgb)
    view_id=np.concatenate(all_view)
    if len(xyz)<1000:
        raise ValueError('Too few registered desk points')
    voxel=np.floor(xyz/(args.voxel_mm/1000)).astype(np.int32)
    _,indices=np.unique(voxel,axis=0,return_index=True)
    xyz,colors,view_id=xyz[indices],colors[indices],view_id[indices]
    data=np.empty(len(xyz),dtype=[('x','<f4'),('y','<f4'),('z','<f4'),
                                  ('red','u1'),('green','u1'),('blue','u1'),('view','u1')])
    for j,name in enumerate(('x','y','z')):
        data[name]=xyz[:,j]
    for j,name in enumerate(('red','green','blue')):
        data[name]=colors[:,j]
    data['view']=view_id
    out=reg_path.parent
    ply=out/'desk_box_cup_measured.ply'
    header=(f'ply\nformat binary_little_endian 1.0\n'
            f'comment Measured Gemini 2 depth points only; first-view table frame, metres\n'
            f'element vertex {len(data)}\n'
            'property float x\nproperty float y\nproperty float z\n'
            'property uchar red\nproperty uchar green\nproperty uchar blue\n'
            'property uchar view\nend_header\n')
    with ply.open('wb') as f:
        f.write(header.encode('ascii'))
        f.write(data.tobytes())
    result={'registration':str(reg_path),'registration_sha256':digest(reg_path),
            'points_before_voxel_by_view':counts,'points_after_voxel':len(data),
            'voxel_mm':args.voxel_mm,'stride':args.stride,
            'ply':str(ply),'ply_sha256':digest(ply),
            'frame':'first-view fitted tabletop, z up, metres',
            'provenance':'Only measured depth points sampled from accepted camera views; no size priors or inferred surfaces.',
            'limitations':'Scene crop and voxel selection; no unseen/back surface completion; manual object polygons used for registration.'}
    (out/'fusion.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps(result,indent=2))


if __name__=='__main__':
    main()
