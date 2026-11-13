"""Estimate box and cup collision proxies from registered measured depth only."""
import argparse
import json
from pathlib import Path

import cv2
import numpy as np

from register_gemini_landmarks import examine


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('registration',type=Path)
    parser.add_argument('view_config',type=Path)
    args=parser.parse_args()
    registration_path=args.registration.resolve()
    reg=json.loads(registration_path.read_text(encoding='utf-8'))
    if reg.get('passed') is not True:
        raise ValueError('Refusing to fit geometry from a failed registration')
    config=json.loads(args.view_config.read_text(encoding='utf-8'))
    specs=config['views'][:len(reg['views'])]
    if len(specs)!=len(reg['views']):
        raise ValueError('Number of view polygons differs from registration')
    views=[examine(spec) for spec in specs]
    for view,expected in zip(views,reg['views']):
        if Path(view['folder']).resolve()!=Path(expected).resolve():
            raise ValueError('View config and registration point to different captures')
    objects={}
    for name in ('box','cup'):
        samples=[]
        for i,v in enumerate(views):
            T=np.asarray(reg['T_reference_table_view_table'][i])
            p=v['objects'][name]
            samples.append(p@T[:3,:3].T+T[:3,3])
        points=np.concatenate(samples)
        raised=points[(points[:,2]>.025)&(points[:,2]<.3)]
        if len(raised)<500:
            raise ValueError(f'Too few measured {name} surface points')
        top=float(np.percentile(raised[:,2],95))
        top_points=raised[raised[:,2]>=top-.016]
        if len(top_points)<300:
            raise ValueError(f'Too few measured {name} upper points')
        rect=cv2.minAreaRect(top_points[:,:2].astype(np.float32))
        center=np.asarray(rect[0],float)
        if name=='box':
            corners=cv2.boxPoints(rect)
            edges=np.roll(corners,-1,axis=0)-corners
            major=edges[np.argmax(np.linalg.norm(edges,axis=1))]
            yaw=float(np.arctan2(major[1],major[0]))
            dims=[float(max(rect[1])),float(min(rect[1])),top]
            objects[name]={'shape':'box','center_table_m':[float(center[0]),float(center[1]),top/2],
                           'dimensions_m':dims,'yaw_table_rad':yaw}
        else:
            radius=float(max(rect[1])/2)
            objects[name]={'shape':'solid_cylinder_proxy',
                           'center_table_m':[float(center[0]),float(center[1]),top/2],
                           'radius_m':radius,'height_m':top,
                           'note':'Cup hollow interior and handle are not measured as a collision shape.'}
        objects[name].update(measured_surface_points=len(points),
                             upper_surface_points=len(top_points),
                             upper_footprint_extent_m=[float(x) for x in rect[1]],
                             dimensions_source='Gemini 2 measured depth, no ruler or prior size')
    result={'registration':str(registration_path),'view_config':str(args.view_config.resolve()),
            'frame':'first-view fitted tabletop, metres, z up',
            'table':{'shape':'plane','z_m':0.},'objects':objects,
            'limitations':'Manual image polygons and measured visible surfaces; missing backs and hollow cup are approximated.'}
    out=registration_path.parent/'geometry.json'
    out.write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps(result,indent=2))
    print(out)


if __name__=='__main__':
    main()
