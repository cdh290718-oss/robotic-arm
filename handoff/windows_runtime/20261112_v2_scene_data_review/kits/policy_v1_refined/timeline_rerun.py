"""Timeline around the close/open episodes of the V1 rerun (221952)."""
import json
import numpy as np
from pathlib import Path

base = Path(r'D:\桌面\萌芽杯\policy_v1_refined\results\20261111_221952_936_Grasp_bf16')
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

def show(t0, t1, step=0.5):
    print(f'--- t={t0}..{t1}s ---')
    print('  t     grip_desc grip_cmd grip_meas | lift   elbow  wf    pan')
    for t in np.arange(t0, t1+1e-9, step):
        i = int(np.argmin(np.abs(el-t)))
        print(f'{el[i]:6.2f}  {des[i,5]:7.1f} {cmd_g[i]:7.1f} {st[i,5]:8.1f} | '
              f'{jd(rows[i]["raw_command"][1], "shoulder_lift"):6.1f} {jd(rows[i]["raw_command"][2], "elbow_flex"):6.1f} '
              f'{jd(rows[i]["raw_command"][3], "wrist_flex"):5.1f} {jd(rows[i]["raw_command"][0], "shoulder_pan"):5.1f}')

# last closure + preceding reopen/approach
show(105, 112.4)
show(112.4, 120.0)
