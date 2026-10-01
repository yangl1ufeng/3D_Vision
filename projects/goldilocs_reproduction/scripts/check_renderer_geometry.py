"""Analytic geometric controls; no RC-3D labels or threshold fitting."""
import json
import sys
from pathlib import Path
import numpy as np
from PIL import Image
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from goldilocs_local.geometry import project, rasterize, reverse_depth_keep
from goldilocs_local.surface import rasterize_surface


def main():
    out = ROOT/'audit/revision/geometry_controls'
    out.mkdir(parents=True,exist_ok=True)
    h,w = 96,128
    yy,xx = np.mgrid[:h,:w]
    k = np.array([[100.,0.,63.5],[0.,100.,47.5],[0.,0.,1.]])
    depth = np.full((h,w),2.)
    pts = np.stack([(xx-k[0,2])*depth/100,(yy-k[1,2])*depth/100,depth],axis=-1)
    colors = np.stack([xx*2,yy*2,np.full_like(xx,100)],axis=-1).astype('uint8')
    uv,z,good = project(pts,k,np.eye(4))
    error = float(np.max(np.abs(uv-np.stack([xx,yy],axis=-1).reshape(-1,2))))
    reports = []
    for angle in (0.,20.,40.):
        a = np.radians(angle)
        pose = np.eye(4)
        pose[:3,:3] = [[np.cos(a),0,np.sin(a)],[0,1,0],[-np.sin(a),0,np.cos(a)]]
        # Orbit around plane centre while looking at it, then move closer.
        pose[:3,3] = np.array([0,0,2.])-pose[:3,:3]@np.array([0,0,1.])
        target_k = k.copy()
        target_k[:2] *= 4
        target_k[:2,2] += 1.5  # Pixel-centred 4x resize.
        shape = (h*4,w*4)
        im,dep,valid = rasterize(pts,colors,target_k,pose,shape,radius=4)
        surface,surface_depth,surface_valid = rasterize_surface(pts,colors,depth,target_k,pose,shape)
        rays = np.stack(np.meshgrid(np.arange(shape[1]),np.arange(shape[0])),axis=-1)
        dirs_cam = np.concatenate([rays,np.ones((*shape,1))],axis=-1)@np.linalg.inv(target_k).T
        dirs_world = dirs_cam@pose[:3,:3].T
        t = (2.-pose[2,3])/dirs_world[:,:,2]
        hits = pose[:3,3]+dirs_world*t[:,:,None]
        # Only the plane interior; do not confuse legitimate image borders with holes.
        supported = (hits[:,:,0]>pts[2,2,0])&(hits[:,:,0]<pts[-3,-3,0])&(hits[:,:,1]>pts[2,2,1])&(hits[:,:,1]<pts[-3,-3,1])&(t>0)
        expected_z = t  # Camera rays have z=1.
        both = supported&valid
        reports.append({'yaw_degrees':angle,'expected_interior_pixels':int(supported.sum()),
                        'interior_hole_fraction':float((supported&~valid).sum()/supported.sum()),
                        'depth_absolute_error_mean':float(np.abs(dep[both]-expected_z[both]).mean()),
                        'surface_interior_hole_fraction':float((supported&~surface_valid).sum()/supported.sum()),
                        'surface_depth_error_mean':float(np.abs(surface_depth[supported&surface_valid]-expected_z[supported&surface_valid]).mean())})
        Image.fromarray(im).save(out/f'plane_{int(angle)}.png')
        Image.fromarray(surface).save(out/f'plane_surface_{int(angle)}.png')
        assert (supported&~surface_valid).sum()==0
        assert reports[-1]['surface_depth_error_mean']<1e-6
    # Reverse-depth rule with perfect known geometry.
    foreground = pts.copy()
    foreground[30:60,40:70] *= .7
    keep = reverse_depth_keep(foreground,depth,k,np.eye(4)).reshape(h,w)
    interior = np.zeros((h,w),dtype=bool)
    interior[30:60,40:70] = True
    removed_inside = int((~keep&interior).sum())
    removed_outside = int((~keep&~interior).sum())
    assert error < 1e-8 and removed_inside==900 and removed_outside==0
    report = {'kind':'analytic_geometry_not_benchmark','self_projection_max_error_px':error,
              'known_foreground_points_removed':removed_inside,'static_points_wrongly_removed':removed_outside,
              'point_renderer_controls':reports,
              'interpretation':'Fixed-radius point splats have finite surface support; this test does not establish real MASt3R accuracy.'}
    (out/'summary.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps(report,indent=2))


if __name__=='__main__':
    main()
