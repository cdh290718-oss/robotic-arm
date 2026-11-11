"""Round 2: effective speed, chunk-boundary jumps, contact stalls, 4x comparison."""
import json
import numpy as np
from pathlib import Path

K8 = Path(r'D:\桌面\机械臂\policy_demo8x_trial')
K4 = Path(r'D:\桌面\萌芽杯\policy_speed4_trial')
RUNS = {
    '8x_213457': K8/'results'/'20261111_213457_906_Grasp_bf16',
    '4x_202520': K4/'results'/'20261111_202520_620_Grasp_bf16',
}
CAL = json.loads((K8/'reference_calibration.json').read_text())
MOTORS = ('shoulder_pan','shoulder_lift','elbow_flex','wrist_flex','wrist_roll','gripper')
DEG = 360/4095
GR_MIN, GR_MAX = CAL['gripper']['range_min'], CAL['gripper']['range_max']

def load(run):
    rows = [json.loads(l) for l in (run/'control.jsonl').read_text(encoding='utf-8').splitlines()]
    el = np.array([r['elapsed_s'] for r in rows])
    des = np.array([r['desired'] if r['desired'] else [np.nan]*6 for r in rows], float)
    cmd = np.array([r['raw_command'] for r in rows], float)
    st = np.array([r['state'] for r in rows], float)
    clip = np.array([r['clipped'] for r in rows], bool)
    cid = np.array([int(r['chunk_id']) for r in rows])
    return rows, el, des, cmd, st, clip, cid

def deg_j(tick, name):
    c = CAL[name]
    return (np.asarray(tick)-(c['range_min']+c['range_max'])/2)*DEG

for tag, run in RUNS.items():
    rows, el, des, cmd, st, clip, cid = load(run)
    print(f'\n########## {tag} ##########')
    print('duration', round(el[-1],1), 'ticks', len(el))
    # measured effective velocities
    dv = np.diff(st, axis=0)/np.diff(el)[:,None]
    dv = dv[:5]  # body
    print('measured body velocity deg/s (p50, p95, max):')
    for i, name in enumerate(MOTORS[:5]):
        v = np.abs(dv[:,i])
        print(f'  {name}: {np.median(v):5.2f} {np.percentile(v,95):5.2f} {v.max():5.2f}')
    # command velocity ticks/tick
    dc = np.abs(np.diff(cmd, axis=0))
    print('command step ticks/update (p50, p95, max):')
    for i, name in enumerate(MOTORS):
        print(f'  {name}: {np.median(dc[:,i]):4.1f} {np.percentile(dc[:,i],95):4.1f} {dc[:,i].max():4.1f}')
    # gripper target reversals
    d = np.diff(des[:,5]); s = np.sign(d[np.abs(d)>1.0])
    print('gripper desired reversals:', int(np.sum(s[1:]!=s[:-1])))
    # chunk-boundary jumps vs within
    sw = np.where(np.diff(cid)!=0)[0]
    within = np.ones(len(des)-1, bool); within[sw] = False
    for i, name in enumerate(['pan','lift','elbow','wf','gripper']):
        dd = np.abs(np.diff(des[:,i]))
        print(f'  |d_des| {name}: within med={np.median(dd[within]):.2f} p95={np.percentile(dd[within],95):.2f} | chunk-switch med={np.median(dd[sw]):.2f} p95={np.percentile(dd[sw],95):.2f}')
    # full-closure episodes with measured stall check
    g = st[:,5]; dg = des[:,5]
    closed = g < 2.5
    eps, start = [], None
    for i, m in enumerate(closed):
        if m and start is None: start = i
        if not m and start is not None: eps.append((start, i-1)); start = None
    if start is not None: eps.append((start, len(g)-1))
    print('full-closure episodes (measured<2.5%):')
    for s_, e_ in eps:
        if el[e_]-el[s_] < 0.2: continue
        print(f'  [{el[s_]:6.2f}..{el[e_]:6.2f}] {el[e_]-el[s_]:5.2f}s meas_min={g[s_:e_+1].min():.2f}% des_min={dg[s_:e_+1].min():.2f}%')
    # contact stalls: measured flat while command closing
    dm = np.diff(g); dc5 = np.diff(cmd[:,5])
    stall = (np.abs(dm) < 0.06) & (dc5 < -0.3)
    idx = np.where(stall)[0]
    print(f'contact-stall ticks: {len(idx)}')
    if len(idx):
        runs2, s_, p_ = [], idx[0], idx[0]
        for i in idx[1:]:
            if i-p_ > 3: runs2.append((s_, p_)); s_ = i
            p_ = i
        runs2.append((s_, p_))
        for s_, e_ in runs2:
            if el[e_]-el[s_] < 0.1: continue
            print(f'  stall [{el[s_]:6.2f}..{el[e_]:6.2f}] {el[e_]-el[s_]:4.2f}s meas={np.median(g[s_:e_+1]):5.1f}% des={np.median(dg[s_:e_+1]):5.1f}% cmd_min={g[s_:e_+1].min():5.1f}%')
