"""Quick behavior summary of the V1 rerun (20261111_221952)."""
import json
import numpy as np
from pathlib import Path

base = Path(r'D:\桌面\萌芽杯\policy_v1_refined\results\20261111_221952_936_Grasp_bf16')
rows = [json.loads(l) for l in (base/'control.jsonl').read_text(encoding='utf-8').splitlines()]
chunks = [json.loads(l) for l in (base/'chunks.jsonl').read_text(encoding='utf-8').splitlines()]
el = np.array([r['elapsed_s'] for r in rows])
des = np.array([r['desired'] if r['desired'] is not None else [np.nan]*6 for r in rows], float)
cmd = np.array([r['raw_command'] for r in rows], float)
st = np.array([r['state'] for r in rows], float)
idx = np.array([r['chunk_index'] for r in rows])
cid = np.array([int(r['chunk_id']) for r in rows])
obs_age = np.array([r['observation_age_ms'] for r in rows])
GRMIN, GRMAX = 2028, 3474
cmd_g = (cmd[:,5]-GRMIN)/(GRMAX-GRMIN)*100

print('ticks', len(rows), 'duration', round(el[-1],1))
print('frames>=25 executed: %.1f%% (chunks used %d)' % (np.mean(idx>=25)*100, len(set(cid))))
print('obs age p50=%.0fms p95=%.0fms' % (np.median(obs_age), np.percentile(obs_age,95)))
print('gripper desired: min=%.2f p50=%.1f max=%.1f' % (np.nanmin(des[:,5]), np.nanmedian(des[:,5]), np.nanmax(des[:,5])))
print('gripper command: min=%.2f p50=%.1f max=%.1f' % (cmd_g.min(), np.median(cmd_g), cmd_g.max()))
print('gripper meas:    min=%.2f p50=%.1f max=%.1f' % (st[:,5].min(), np.median(st[:,5]), st[:,5].max()))
# closure episodes (<10%)
m = st[:,5] < 10
eps, s = [], None
for i,v in enumerate(m):
    if v and s is None: s=i
    if not v and s is not None: eps.append((s,i-1)); s=None
if s is not None: eps.append((s,len(m)-1))
print('measured gripper <10%% episodes:', len(eps))
for a,b in eps:
    print(f'  [{el[a]:6.2f}..{el[b]:6.2f}] {el[b]-el[a]:5.2f}s meas_min={st[a:b+1,5].min():.2f}%')
# body tracking errors vs command
CAL = json.loads((base.parent.parent/'reference_calibration.json').read_text())
DEG = 360/4095
def jd(t,n):
    c=CAL[n]; return (np.asarray(t)-(c['range_min']+c['range_max'])/2)*DEG
for i,n in enumerate(('shoulder_pan','shoulder_lift','elbow_flex','wrist_flex')):
    err = np.abs(st[:,i]-jd(cmd[:,i],n))
    print(f'{n}: trackerr med={np.median(err):.2f} p95={np.percentile(err,95):.2f} max={err.max():.2f} deg')
