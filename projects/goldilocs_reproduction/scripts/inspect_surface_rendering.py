"""Compare renderer support on fixed regression examples without annotations."""
import json
import sys
from pathlib import Path
import numpy as np
from PIL import Image,ImageDraw
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from goldilocs_local.geometry import original_resolution_intrinsic
from goldilocs_local.surface import render_pair_surface

out = ROOT/'audit/revision/surface_examples'
out.mkdir(parents=True,exist_ok=True)
rows = []
for pid in ('part1_banana','part1_bottle','part1_bowl','part1_yellow_box'):
    folder = ROOT/'runs/rc3d_native'/pid
    with np.load(folder/'geometry/scene.npz') as f:
        scene = dict(f)
    meta = json.loads((folder/'geometry/geometry.json').read_text())
    k = original_resolution_intrinsic(scene['K_1'],meta['image_transforms'][1])
    target = np.array(Image.open(folder/'view_1/target.png'))
    cross,clean,valid,clean_valid,kept = render_pair_surface(scene,1,k,target.shape[:2])
    dest = out/pid
    dest.mkdir(exist_ok=True)
    for name,im in [('cross_surface',cross),('clean_surface',clean),('valid',valid.astype('uint8')*255),('clean_valid',clean_valid.astype('uint8')*255)]:
        Image.fromarray(im).save(dest/f'{name}.png')
    old_valid = np.array(Image.open(folder/'view_1/valid.png'))>0
    old_clean_valid = np.array(Image.open(folder/'view_1/clean_valid.png'))>0
    canvas = Image.new('RGB',(1280,1020),'#202020')
    for n,(title,im) in enumerate([
        ('Original point renderer',Image.open(folder/'view_1/cross_render.png')),
        ('Surface renderer (edge guarded)',Image.fromarray(cross)),
        ('Original clean scene',Image.open(folder/'view_1/clean_render.png')),
        ('Surface clean scene',Image.fromarray(clean))]):
        x,y = n%2*640,n//2*510
        canvas.paste(im.resize((640,480)),(x,y+30))
        ImageDraw.Draw(canvas).text((x+10,y+8),title,fill='white')
    canvas.save(dest/'comparison.jpg',quality=93)
    rows.append({'pair':pid,'point_valid_fraction':float(old_valid.mean()),'surface_valid_fraction':float(valid.mean()),
                 'point_clean_valid_fraction':float(old_clean_valid.mean()),'surface_clean_valid_fraction':float(clean_valid.mean()),
                 'surface_depth_edge_ratio':1.05,'ground_truth_used':False})
    print(json.dumps(rows[-1]),flush=True)
(out/'summary.json').write_text(json.dumps(rows,indent=2),encoding='utf-8')
