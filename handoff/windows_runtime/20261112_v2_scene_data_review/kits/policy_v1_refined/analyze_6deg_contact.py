"""Timeline of the 6-degree run (224253): gripper layers + body pose near contacts."""
import json
import numpy as np
from pathlib import Path

base = Path(r'D:\桌面\萌芽杯\policy_v1_refined\results\20261111_224253_835_Grasp_bf16')
rows = [json.loads(l) for l in (base/'control.jsonl').read_text(encoding='utf-8').splitlines()]
CAL = json.loads((base.parent.parent/'reference_calibration.json').read_text())
DEG = 360/4095
GN = ('shoulder_pan','shoulder_lift','elbow_flex','wrist_flex')
def jd(t, n):
    c = CAL[n]
    return (np.asarray(t)-(c['range_min']+c['range_max'])/2)*DEG

el = np.array([r['elapsed_s'] for r in rows])
des = np.array([r['desired'] if r['desired'] is not None else [np.nan]*6 for r in rows], float)
st = np.array([r['state'] for r in rows], float)
GRMIN, GRMAX = 2028, 3474
cmd_g = (np.array([r['raw_command'] for r in rows])[:,5]-GRMIN)/(GRMAX-GRMIN)*100
idx = np.array([r['chunk_index'] for r in rows])

def show(t0, t1):
    print(f'--- t={t0}..{t1}s (elapsed) ---')
    print('   t    des_g cmd_g meas_g | command: lift   elbow  wf   | meas: lift elbow')
    for t in np.arange(t0, t1+1e-9, 0.5):
        i = int(np.argmin(np.abs(el-t)))
        rc = rows[i]['raw_command']
        print(f'{el[i]:6.2f} {des[i,5]:5.1f} {cmd_g[i]:5.1f} {st[i,5]:5.1f} | '
              f'{jd(rc[1], "shoulder_lift"):6.1f} {jd(rc[2], "elbow_flex"):6.1f} {jd(rc[3], "wrist_flex"):5.1f} | '
              f'{st[i,1]:6.1f} {st[i,2]:6.1f}')

print('frames>=25: %.1f%%' % (np.mean(idx>=25)*100))
# find all ticks where measured gripper < 12 (approaching closed)
m = st[:,5] < 12
segs, s = [], None
for i,v in enumerate(m):
    if v and s is None: s=i
    if not v and s is not None: segs.append((s,i-1)); s=None
if s is not None: segs.append((s,len(m)-1))
print('measured<12%% segments:', len(segs))
for a,b in segs:
    if el[b]-el[a] < 0.1: continue
    print(f'  [{el[a]:6.2f}..{el[b]:6.2f}] {el[b]-el[a]:4.2f}s meas_min={st[a:b+1,5].min():5.1f}% des_min={des[a:b+1,5].min():5.1f}%')

show(108, 120)
