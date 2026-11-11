"""Focused comparison: baseline vs V1/V2 candidates (corrected availability sim)."""
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
            active = pending; used = 0
        elif active >= 0 and pending >= 0 and pending != active:
            expired = (now - obs_t[active])*action_hz >= 50
            if used >= hold_ticks or expired:
                active = pending; used = 0
        if active < 0:
            continue
        idx = int((now - obs_t[active])*action_hz)
        if idx >= 50:
            continue
        idxs[i] = idx; acts[i] = active; used += 1
    executed = np.full((N, 6), np.nan)
    for i in range(N):
        if acts[i] >= 0:
            executed[i] = A[int(acts[i]), idxs[i]]
    return executed, idxs

def summarize(executed, idxs):
    g = executed[:, 5]
    valid = ~np.isnan(g)
    m = g[valid] < 10.0
    best = cur = eps = 0
    prev = False
    for v in m:
        cur = cur+1 if v else 0
        best = max(best, cur)
        if v and not prev: eps += 1
        prev = v
    d = np.abs(np.diff(executed[:, :5], axis=0)); d = d[~np.isnan(d)]
    return dict(valid=int(valid.sum()), closed=float(m.sum()/HZ), longest=float(best/HZ),
                eps=eps, late=float(np.mean(idxs[valid] >= 25)), gt2=int((d > 2).sum()))

print(f'{"policy":34s} valid closed_s longest_s eps frames25-49 body>2')
for name, hold, hz in (
        ('baseline hold=0 30Hz', 0, 30.0),
        ('V1 hold=25 30Hz', 25, 30.0),
        ('V1b hold=40 30Hz', 40, 30.0),
        ('V2 hold=80 15Hz', 80, 15.0),
        ('V2b hold=120 10Hz', 120, 10.0)):
    exe, idxs = simulate(hold, hz)
    s = summarize(exe, idxs)
    print(f'{name:34s} {s["valid"]:5d} {s["closed"]:6.1f} {s["longest"]:7.2f} {s["eps"]:4d} '
          f'{s["late"]*100:6.0f}% {s["gt2"]:6d}')
