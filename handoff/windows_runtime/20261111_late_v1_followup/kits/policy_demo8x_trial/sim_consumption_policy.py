"""Offline consumption-policy experiment for the 8x run's recorded chunks (no robot).

Baseline : replace active chunk with newest every tick, index=int(age*30)  (current behavior)
V1       : keep active chunk for at least 20 ticks, index=int(age*30)
V2       : keep active chunk for at least 60 ticks, index=int(age*15)   (15 Hz playback)

Outputs per policy: gripper closure stats (<10%), fraction of ticks executing
frames 25-49 (the model's closure region), per-tick body jump count (raw desired),
and hold ticks (plan expired).
"""
import json
import numpy as np
from pathlib import Path

base = Path(r'D:\桌面\机械臂\policy_demo8x_trial\results\20261111_213457_906_Grasp_bf16')
chunks = [json.loads(l) for l in (base/'chunks.jsonl').read_text(encoding='utf-8').splitlines()]
chunks.sort(key=lambda c: c['observation_time'])
A = np.array([np.asarray(c['actions']) for c in chunks], dtype=np.float64)
obs_t = np.array([c['observation_time'] for c in chunks])
t0 = obs_t[0]
DUR = 120.0
HZ = 30.0
N = int(DUR*HZ)

def simulate(hold_ticks, action_hz):
    t = t0 + np.arange(N)/HZ
    # arrivals: pending chunk = newest chunk with obs_time <= t and age <= 1.4
    pending = -1
    j = 0
    active = -1
    active_start_t = 0.0
    used = 0
    rows = []
    hold = 0
    for i, now in enumerate(t):
        while j < len(obs_t) and obs_t[j] <= now:
            pending = j
            j += 1
        # promote pending to active per policy
        if pending >= 0 and pending != active:
            promote = False
            if active < 0:
                promote = True
            elif used >= hold_ticks:
                promote = True
            if promote:
                active = pending
                active_start_t = now
                used = 0
        if active < 0:
            hold += 1
            rows.append((np.nan, np.nan, 0))
            continue
        age = now - obs_t[active]
        idx = int(age*action_hz)
        if idx >= 50:
            # plan expired: hold at last command (runner behavior)
            hold += 1
            rows.append((np.nan, np.nan, 0))
            continue
        rows.append((active, float(age*action_hz), idx))
        used += 1
    # build executed desired arrays (NaN where hold)
    executed = np.full((N, 6), np.nan)
    idxs = np.array([r[2] for r in rows])
    acts = np.array([r[0] for r in rows])
    for i in range(N):
        if not np.isnan(acts[i]):
            executed[i] = A[int(acts[i]), int(idxs[i])]
    return executed, hold, idxs

def closure(g):
    m = g[~np.isnan(g)] < 10.0
    if not m.any():
        return {'closed_s': 0.0, 'longest_s': 0.0, 'eps': 0}
    best = cur = 0
    eps = 0
    prev = False
    run = 0
    for v in m:
        cur = cur+1 if v else 0
        best = max(best, cur)
        if v and not prev: eps += 1
        prev = v
    return {'closed_s': float(m.sum()/HZ), 'longest_s': float(best/HZ), 'eps': eps}

def body_jumps(exe):
    d = np.abs(np.diff(exe[:, :5], axis=0))
    d = d[~np.isnan(d)]
    return {'gt2': int((d > 2).sum()), 'gt5': int((d > 5).sum()), 'path_deg': float(d.sum())}

policies = {
    'baseline 30Hz replace-each-tick': (0, 30.0),
    'V1 hold>=20ticks 30Hz': (20, 30.0),
    'V1b hold>=25ticks 30Hz': (25, 30.0),
    'V2 hold>=60ticks 15Hz': (60, 15.0),
}
out = {}
for name, (hold_ticks, hz) in policies.items():
    exe, hold_n, idxs = simulate(hold_ticks, hz)
    g = exe[:, 5][~np.isnan(exe[:, 5])]
    c = closure(g)
    late = float(np.mean(idxs >= 25)) if len(idxs) else 0.0
    bj = body_jumps(exe)
    out[name] = {'hold_ticks': hold_n, 'closed_s': c['closed_s'], 'longest_s': c['longest_s'],
                 'eps': c['eps'], 'frac_frames_25_49': late, 'body_jumps': bj}
    print(f'{name}: hold={hold_n} closed={c["closed_s"]:.1f}s longest={c["longest_s"]:.2f}s eps={c["eps"]} '
          f'frames25-49={late*100:.0f}% body_jumps>2deg={bj["gt2"]} >5deg={bj["gt5"]}')

(base.parent.parent/'consumption_policy_summary.json').write_text(json.dumps(out, indent=2, ensure_ascii=False))
print('\nsaved consumption_policy_summary.json')
