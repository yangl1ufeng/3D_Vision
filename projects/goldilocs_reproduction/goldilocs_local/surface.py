"""Optional continuous surface rasterization of a structured pointmap.

An explicit independent renderer alternative. It does not alter inferred geometry,
fill unseen regions, or interpolate across filtered-out vertices/depth edges.
"""
import numpy as np
from numba import njit
from .geometry import project, reverse_depth_keep


@njit(cache=True)
def _draw(uv, z, good, source_depth, keep, colors, height, width, edge_ratio):
    sh,sw = keep.shape
    rgb = np.zeros((height,width,3),np.uint8)
    depth = np.full((height,width),np.inf,np.float64)
    for sy in range(sh-1):
        for sx in range(sw-1):
            for triangle in range(2):
                ia = sy*sw+sx
                ib = ia+1 if triangle==0 else ia+sw+1
                ic = ia+sw+1 if triangle==0 else ia+sw
                if not (keep.flat[ia] and keep.flat[ib] and keep.flat[ic] and good[ia] and good[ib] and good[ic]):
                    continue
                da,db,dc = source_depth.flat[ia],source_depth.flat[ib],source_depth.flat[ic]
                if min(da,db,dc)<=0 or max(da,db,dc)/min(da,db,dc)>edge_ratio:
                    continue
                ax,ay = uv[ia,0],uv[ia,1]
                bx,by = uv[ib,0],uv[ib,1]
                cx,cy = uv[ic,0],uv[ic,1]
                det = (by-cy)*(ax-cx)+(cx-bx)*(ay-cy)
                if abs(det)<1e-9:
                    continue
                x0,x1 = max(0,int(np.ceil(min(ax,bx,cx)))),min(width-1,int(np.floor(max(ax,bx,cx))))
                y0,y1 = max(0,int(np.ceil(min(ay,by,cy)))),min(height-1,int(np.floor(max(ay,by,cy))))
                for y in range(y0,y1+1):
                    for x in range(x0,x1+1):
                        a = ((by-cy)*(x-cx)+(cx-bx)*(y-cy))/det
                        b = ((cy-ay)*(x-cx)+(ax-cx)*(y-cy))/det
                        c = 1-a-b
                        if min(a,b,c)<-1e-7:
                            continue
                        inverse_depth = a/z[ia]+b/z[ib]+c/z[ic]
                        if inverse_depth<=0:
                            continue
                        zz = 1/inverse_depth
                        if zz<depth[y,x]:
                            depth[y,x] = zz
                            wa,wb,wc = a/z[ia]*zz,b/z[ib]*zz,c/z[ic]*zz
                            for channel in range(3):
                                value = wa*colors[ia,channel]+wb*colors[ib,channel]+wc*colors[ic,channel]
                                rgb[y,x,channel] = np.uint8(min(255.,max(0.,value+.5)))
    return rgb,depth


def rasterize_surface(points, colors, source_depth, intrinsic, pose, shape, keep=None, edge_ratio=1.05):
    if edge_ratio<1:
        raise ValueError('edge_ratio must be >= 1')
    h,w = points.shape[:2]
    if keep is None:
        keep = np.ones((h,w),dtype=bool)
    if keep.shape!=(h,w) or source_depth.shape!=(h,w):
        raise ValueError('Source grid dimensions differ')
    uv,z,good = project(points,intrinsic,pose)
    rgb,depth = _draw(uv,z,good,np.asarray(source_depth,dtype=np.float64),keep,
                     colors.reshape(-1,3),shape[0],shape[1],edge_ratio)
    return rgb,depth.astype(np.float32),np.isfinite(depth)


def render_pair_surface(scene, target, intrinsic, shape, edge_ratio=1.05, tolerance=0.):
    keep = [reverse_depth_keep(scene[f'points_{i}'],scene[f'depth_{1-i}'],scene[f'K_{1-i}'],scene[f'pose_{1-i}'],tolerance)
            .reshape(scene[f'depth_{i}'].shape) for i in (0,1)]
    def render(i, mask=None):
        return rasterize_surface(scene[f'points_{i}'],scene[f'rgb_{i}'],scene[f'depth_{i}'],
            intrinsic,scene[f'pose_{target}'],shape,keep=mask,edge_ratio=edge_ratio)
    cross,_,valid = render(1-target)
    a,da,va = render(0,keep[0])
    b,db,vb = render(1,keep[1])
    nearer = db<da
    clean = np.where(nearer[:,:,None],b,a)
    return cross,clean,valid,va|vb,[float(k.mean()) for k in keep]
