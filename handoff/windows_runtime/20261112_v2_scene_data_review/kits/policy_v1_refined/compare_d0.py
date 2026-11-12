"""Compare D=0 run vs D=32 run (same V2 params)."""
import json
import numpy as np
from pathlib import Path

base = Path(r'D:\桌面\萌芽杯\policy_v1_refined\results')
runs = {'D=32 211450': base/'20261112_211450_921_Grasp_bf16',
        'D=0 214119': base/'20261112_214119_107_Grasp_bf16'}
CAL = json.loads((base.parent/'reference_calibration.json').read_text())
DEG = 360/4095
GN = ('shoulder_pan','shoulder_lift','elbow_flex','wrist_flex')
def jd(t, n):
    c = CAL[n]
    return (np.asarray(t)-(c['range_min']+c['range_max'])/2)*DEG

for tag, run in runs.items():
    rows = [json.loads(l) for l in (run/'control.jsonl').read_text(encoding='utf-8').splitlines()]
    st = np.array([r['state'] for r in rows])
    el = np.array([r['elapsed_s'] for r in rows])
    idx = np.array([r['chunk_index'] for r in rows])
    obs = np.array([r['observation_age_ms'] for r in rows])
    cmd = np.array([r['raw_command'] for r in rows])
    g = st[:,5]
    m = g < 10
    eps, s = [], None
    for i,v in enumerate(m):
        if v and s is None: s=i
        if not v and s is not None: eps.append((s,i-1)); s=None
    if s is not None: eps.append((s,len(m)-1))
    print(f'== {tag} ==')
    print(f'ticks {len(rows)}, frames>=25 {np.mean(idx>=25)*100:.0f}%, obs p50 {np.median(obs):.0f}ms')
    print(f'gripper meas min={g.min():.2f} p50={np.median(g):.1f} max={g.max():.1f}, <10% episodes={len(eps)}')
    for a,b in eps:
        if el[b]-el[a] >= 0.5: print(f'   [{el[a]:6.2f}..{el[b]:6.2f}] {el[b]-el[a]:5.2f}s min={g[a:b+1].min():5.2f}%')
    for i, n in enumerate(GN):
        err = np.abs(st[:,i]-jd(cmd[:,i], n))
        print(f'  {n}: trackerr med={np.median(err):.2f}° p95={np.percentile(err,95):.2f}° max={err.max():.2f}°')
    dv = np.abs(np.diff(st[:, :4], axis=0)/np.diff(el)[:,None])
    print('  measured body vel max: ' + ', '.join(f'{n} {v.max():.1f}' for n, v in zip(GN, dv.T)))
    s2 = json.loads((run/'summary.json').read_text(encoding='utf-8'))
    print('  final goal:', s2['motion_cleanup'].get('raw_goal'))
    print()
