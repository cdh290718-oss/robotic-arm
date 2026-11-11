"""Render a simple timeline chart of the 8x run gripper layers using PIL (no matplotlib)."""
import json
import numpy as np
from pathlib import Path
from PIL import Image, ImageDraw

base = Path(r'D:\桌面\机械臂\policy_demo8x_trial\results\20261111_213457_906_Grasp_bf16')
rows = [json.loads(l) for l in (base/'control.jsonl').read_text(encoding='utf-8').splitlines()]
el = np.array([r['elapsed_s'] for r in rows])
des = np.array([r['desired'][5] if r['desired'] else np.nan for r in rows])
cmd = np.array([r['raw_command'][5] for r in rows], float)
st = np.array([r['state'][5] for r in rows])
GRMIN, GRMAX = 2028, 3474
cmd = (cmd-GRMIN)/(GRMAX-GRMIN)*100

W, H = 1280, 400
img = Image.new('RGB', (W, H), 'white')
d = ImageDraw.Draw(img)
x0, y0, w, h = 60, 30, W-80, H-70
def X(t): return x0 + t/el[-1]*w
def Y(v): return y0 + h - v/40*h   # 0..40%
for v in range(0, 41, 10):
    d.line([x0, Y(v), x0+w, Y(v)], fill=(220,220,220))
    d.text((8, Y(v)-6), f'{v}%', fill='black')
for t in range(0, 121, 20):
    d.line([X(t), y0, X(t), y0+h], fill=(235,235,235))
    d.text((X(t)-10, y0+h+6), f'{t}s', fill='black')
# colors: desired red, command orange, measured blue
for arr, color in ((des, (220,30,30)), ((cmd), (255,150,0)), (st, (30,90,220))):
    pts = [(X(t), Y(min(40, max(0, v)))) for t, v in zip(el[::2], arr[::2]) if np.isfinite(v)]
    if len(pts) > 1: d.line(pts, fill=color, width=2)
d.text((x0, 6), 'gripper (0=closed, 100=open): RED=model target  ORANGE=limited command  BLUE=measured', fill='black')
img.save(base.parent.parent/'8x夹爪时间线.png')
print('saved')
