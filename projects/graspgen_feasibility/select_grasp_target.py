"""Click a saved RGB-D image, preview SAM2 selection, and run measured grasping."""
import argparse
from datetime import datetime
import gc
import json
import os
from pathlib import Path
import queue
import subprocess
import threading

import numpy as np
from PIL import Image, ImageTk

from click_selection import PointSegmenter, observation, export_selection, preview
from simulation.scene_spec import DEFAULT_SCENE, PROJECT, load_scene


def new_output():
    return PROJECT/'runs/click_selection'/datetime.now().strftime('%Y%m%d_%H%M%S_%f')


def ui(spec, obs):
    import tkinter as tk
    from tkinter import ttk, messagebox
    root = tk.Tk()
    root.title('已保存画面 · 点选抓取目标')
    root.resizable(False, False)
    state = {'positive': [], 'negative': [], 'mask': None, 'engine': None,
             'busy': False, 'output': None, 'process': None}
    messages = queue.Queue()
    width, height = 960, round(obs['rgb'].shape[0]*960/obs['rgb'].shape[1])
    scale = width/obs['rgb'].shape[1]
    ttk.Label(root, text='左键点物体内部，右键点需要排除的背景；可补点后重新生成区域。',
              font=('Microsoft YaHei UI', 11)).pack(padx=14, pady=(12, 6), anchor='w')
    canvas = tk.Canvas(root, width=width, height=height, highlightthickness=0)
    canvas.pack(padx=14)
    status = tk.StringVar(value='这是保存的桌面照片。请先点选一个已建模物体。')
    ttk.Label(root, textvariable=status, wraplength=940, font=('Microsoft YaHei UI', 10)).pack(padx=14, pady=8, anchor='w')
    buttons = ttk.Frame(root)
    buttons.pack(pady=(0, 12), fill='x', padx=14)

    def draw():
        picture = preview(obs['rgb'], state['mask']) if state['mask'] is not None else Image.fromarray(obs['rgb'])
        state['photo'] = ImageTk.PhotoImage(picture.resize((width, height), Image.Resampling.LANCZOS))
        canvas.delete('all')
        canvas.create_image(0, 0, anchor='nw', image=state['photo'])
        for key, color in [('positive', '#16ff70'), ('negative', '#ff4242')]:
            for x, y in state[key]:
                x, y = x*scale, y*scale
                canvas.create_oval(x-4, y-4, x+4, y+4, fill=color, outline='white', width=1)

    def click(event, key):
        if state['busy']:
            return
        state[key].append((min(obs['rgb'].shape[1]-1, event.x/scale),
                           min(obs['rgb'].shape[0]-1, event.y/scale)))
        state['mask'], state['output'] = None, None
        run_button.config(state='disabled')
        status.set(f'已选 {len(state["positive"])} 个物体点、{len(state["negative"])} 个排除点。点击“生成区域”。')
        draw()

    def reset():
        if state['busy']:
            return
        state.update(positive=[], negative=[], mask=None, output=None)
        run_button.config(state='disabled')
        status.set('请重新点选物体。')
        draw()

    def segment():
        if state['busy'] or not state['positive']:
            return
        state['busy'] = True
        state['output'] = None
        segment_button.config(state='disabled')
        run_button.config(state='disabled')
        status.set('正在生成目标区域，首次运行需要加载模型……')
        def work():
            try:
                if state['engine'] is None:
                    state['engine'] = PointSegmenter()
                mask, report = state['engine'].segment(obs['rgb'], state['positive'], state['negative'])
                messages.put(('mask', mask))
                output = new_output()
                saved = export_selection(spec, obs, mask, report, output, 'interactive_user_click')
                messages.put(('selected', mask, saved, output))
            except Exception as exc:
                messages.put(('error', str(exc)))
        threading.Thread(target=work, daemon=True).start()

    def execute():
        if state['busy'] or state['output'] is None:
            return
        state['busy'] = True
        run_button.config(state='disabled')
        segment_button.config(state='disabled')
        status.set('正在生成抓取候选并运行仿真，完成后会显示结果。')
        output = state['output']
        run_dir = output/('grasp_'+datetime.now().strftime('%H%M%S_%f'))
        def work():
            try:
                state['engine'] = None
                gc.collect()
                import torch
                torch.cuda.empty_cache()
                with (output/'grasp.log').open('w', encoding='utf-8') as log:
                    process = subprocess.Popen([str(PROJECT/'.venv-sim/Scripts/python.exe'),
                        str(PROJECT/'run_measured_grasp.py'), '--scene', str(output/'scene.json'),
                        '--output', str(run_dir)], cwd=PROJECT, stdout=log, stderr=subprocess.STDOUT,
                        creationflags=subprocess.CREATE_NO_WINDOW if os.name=='nt' else 0)
                    state['process'] = process
                    process.wait()
                    state['process'] = None
                task_path = run_dir/'execution/task.json'
                if not task_path.exists():
                    raise RuntimeError(f'抓取流程没有完成，请查看 {output / "grasp.log"}')
                messages.put(('executed', json.loads(task_path.read_text()), run_dir))
            except Exception as exc:
                messages.put(('error', str(exc)))
        threading.Thread(target=work, daemon=True).start()

    def poll():
        try:
            while True:
                event = messages.get_nowait()
                if event[0] == 'mask':
                    state['mask'] = event[1]
                    draw()
                    continue
                state['busy'] = False
                segment_button.config(state='normal')
                if event[0] == 'selected':
                    _, mask, report, output = event
                    state.update(mask=mask, output=output)
                    run_button.config(state='normal')
                    status.set(f'已选择 {report["target"]}，提取 {report["exported_points"]} 个深度点。请检查绿色区域，再运行抓取。')
                    draw()
                elif event[0] == 'executed':
                    _, task, run_dir = event
                    status.set(('抓放成功。' if task['task_success'] else '本次未完成抓放，失败原因已保存。') + f'结果：{run_dir}')
                    run_button.config(state='normal')
                    if task.get('animation') and os.name=='nt':
                        os.startfile(task['animation'])
                else:
                    detail = event[1]
                    if 'only part of the object' in detail:
                        detail = '绿色区域只覆盖了物体的一部分，请在未选中的物体表面补一个左键点。'
                    elif 'too much background' in detail:
                        detail = '选区包含过多背景，请补点，或用右键排除背景。'
                    elif 'multiple modeled objects' in detail:
                        detail = '选区覆盖了多个物体，请用右键排除不需要的物体。'
                    status.set('当前选择未完成：'+detail)
        except queue.Empty:
            pass
        root.after(100, poll)

    def close():
        process = state['process']
        if process and process.poll() is None:
            if not messagebox.askyesno('关闭窗口', '抓取仿真仍在运行。关闭窗口并停止任务？'):
                return
            if os.name=='nt':
                subprocess.run(['taskkill', '/PID', str(process.pid), '/T', '/F'],
                               creationflags=subprocess.CREATE_NO_WINDOW, capture_output=True)
            else:
                process.terminate()
        root.destroy()

    ttk.Button(buttons, text='重新选择', command=reset).pack(side='left', padx=(0, 8))
    segment_button = ttk.Button(buttons, text='生成区域', command=segment)
    segment_button.pack(side='left', padx=8)
    run_button = ttk.Button(buttons, text='运行抓取', command=execute, state='disabled')
    run_button.pack(side='left', padx=8)
    ttk.Button(buttons, text='打开选择结果', command=lambda: os.startfile(state['output'])
               if state['output'] and os.name=='nt' else None).pack(side='left', padx=8)
    canvas.bind('<Button-1>', lambda e: click(e, 'positive'))
    canvas.bind('<Button-3>', lambda e: click(e, 'negative'))
    root.protocol('WM_DELETE_WINDOW', close)
    draw()
    root.after(100, poll)
    root.mainloop()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--scene', type=Path, default=DEFAULT_SCENE)
    parser.add_argument('--view-index', type=int, default=0)
    parser.add_argument('--point', nargs=2, type=float, action='append')
    parser.add_argument('--negative', nargs=2, type=float, action='append', default=[])
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    spec = load_scene(args.scene)
    obs = observation(spec, args.view_index)
    if args.point:
        engine = PointSegmenter()
        mask, report = engine.segment(obs['rgb'], args.point, args.negative)
        output = args.output or new_output()
        result = export_selection(spec, obs, mask, report, output, 'automated_test_points')
        print(json.dumps({'output': str(output.resolve()), **result}, indent=2), flush=True)
    else:
        ui(spec, obs)


if __name__ == '__main__':
    main()
