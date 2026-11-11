"""Sweep consumption policy parameters on the recorded chunks (no robot)."""
import json
import numpy as np
from pathlib import Path

base = Path(r'D:\桌面\机械臂\policy_demo8x_trial\results\20261111_213457_906_Grasp_bf16')
chunks = [json.loads(l) for l in (base/'chunks.jsonl').read_text(encoding='utf-8').splitlines()]
chunks.sort(key=lambda c: c['observation_time'])
A = np.array([np.asarray(c['actions']) for c in chunks], dtype=np.float64)
obs_t = np.array([c['observation_time'] for c in chunks])
fin_t = np.array([c['finished_time'] for c in chunks])
avail_order = np.argsort(fin_t)
fin_sorted = fin_t[avail_order]
t0 = obs_t[0]
HZ = 30.0
N = int(120.0*HZ)

def simulate(hold_ticks, action_hz):
    t = t0 + np.arange(N)/HZ
    available = []
    j = 0
    active = -1
    used = 0
    idxs = np.zeros(N, int)
    acts = np.full(N, -1)
    for i, now in enumerate(t):
        while j < len(fin_sorted) and fin_sorted[j] <= now:
            available.append(int(avail_order[j]))
            j += 1
        pending = available[-1] if available else -1
        if active < 0 and pending >= 0:
            active = pending
            used = 0
        elif active >= 0 and pending >= 0 and pending != active:
            expired = (now - obs_t[active])*action_hz >= 50
            if used >= hold_ticks or expired:
                active = pending
                used = 0
        if active < 0:
            continue
        idx = int((now - obs_t[active])*action_hz)
        if idx >= 50:
            continue
        idxs[i] = idx
        acts[i] = active
        used += 1
    executed = np.full((N, 6), np.nan)
    for i in range(N):
        if acts[i] >= 0:
            executed[i] = A[int(acts[i]), idxs[i]]
    return executed, idxs

def summarize(executed, idxs):
    g = executed[:, 5]
    valid = ~np.isnan(g)
    m = g[valid] < 10.0
    best = cur = 0
    eps = 0
    prev = False
    for v in m:
        cur = cur+1 if v else 0
        best = max(best, cur)
        if v and not prev: eps += 1
        prev = v
    d = np.abs(np.diff(executed[:, :5], axis=0))
    d = d[~np.isnan(d)]
    return {'closed_s': float(m.sum()/HZ), 'longest_s': float(best/HZ), 'eps': eps,
            'late': float(np.mean(idxs[valid] >= 25)) if valid.any() else 0.0,
            'valid_ticks': int(valid.sum()),
            'gt2': int((d > 2).sum()), 'gt5': int((d > 5).sum())}

print('hold_ticks action_hz  valid closed_s longest_s eps frames25-49 body>2 body>5')
best = None
for hz in (10.0, 15.0, 20.0, 30.0):
    for hold in (20, 30, 40, 60, 80, 100, 120, 150):
        exe, idxs = simulate(hold, hz)
        s = summarize(exe, idxs)
        line = (f'{hold:5d}  {hz:5.1f}  {s["valid_ticks"]:5d} {s["closed_s"]:6.1f} {s["longest_s"]:7.2f} '
                f'{s["eps"]:4d} {s["late"]*100:6.0f}% {s["gt2"]:6d} {s["gt5"]:5d}')
        print(line)
        if best is None or s['longest_s'] > best[0]['longest_s']:
            best = (s, hz, hold)
print('\nbest longest closure:', best)
