"""Offline replay of the recorded 8x run chunks under three consumption policies.

No robot, no camera: only chunks.jsonl (390 x 50-frame plans) and control.jsonl times.

(a) age-index stitching, faithful to bounded_run.py (chunk replaced as soon as a
    newer one exists; index = int((t-observation_time)*30); frame index >= 50 -> hold)
(b) full sequential playback: chunk i frames 0..49 in order at 30Hz, then chunk i+1
(c) 25-frame sequential playback: chunk i frames 0..24, then chunk i+1

Metrics: gripper closure (0=closed) total/longest, closure episodes, body per-tick
jump counts, cumulative body path, chunk-boundary reopen jumps.
"""
import json
import numpy as np
from pathlib import Path

base = Path(r'D:\桌面\机械臂\policy_demo8x_trial\results\20261111_213457_906_Grasp_bf16')
rows = [json.loads(l) for l in (base/'control.jsonl').read_text(encoding='utf-8').splitlines()]
chunks = [json.loads(l) for l in (base/'chunks.jsonl').read_text(encoding='utf-8').splitlines()]
chunks.sort(key=lambda c: c['observation_time'])
A = np.array([np.asarray(c['actions']) for c in chunks], dtype=np.float32)  # (390,50,6)
obs_t = np.array([c['observation_time'] for c in chunks])
row_t = np.array([r['time'] for r in rows])
actual_des = np.array([r['desired'] if r['desired'] else [np.nan]*6 for r in rows], float)
HZ = 30.0
MAX_RESULT_AGE = 1.4

def closure_stats(g):
    m = g < 10.0
    best = cur = 0
    eps = 0
    for v in m:
        cur = cur+1 if v else 0
        best = max(best, cur)
    # episodes
    i = 0
    while i < len(m):
        if m[i]:
            j = i
            while j < len(m) and m[j]: j += 1
            if j-i >= 3: eps += 1
            i = j
        else:
            i += 1
    return {'closed_frac': float(m.mean()), 'closed_s': float(m.sum()/HZ),
            'longest_s': float(best/HZ), 'episodes': eps}

def body_stats(traj):  # traj (n,6) body actions in deg/%
    d = np.abs(np.diff(traj[:, :5], axis=0))
    return {'jump_gt2': int((d > 2).sum()), 'jump_gt5': int((d > 5).sum()),
            'path_deg': float(d.sum())}

# ---------- (a) faithful age-index stitching ----------
sim_a = np.full((len(rows), 6), np.nan)
hold_ticks = 0
for i, t in enumerate(row_t):
    # active chunk = newest with obs_time <= t and age <= MAX_RESULT_AGE
    j = np.searchsorted(obs_t, t, side='right') - 1
    if j < 0:
        hold_ticks += 1
        continue
    age = t - obs_t[j]
    if age > MAX_RESULT_AGE:
        hold_ticks += 1
        continue
    idx = int(age * HZ)
    if idx >= 50:
        hold_ticks += 1
        continue
    sim_a[i] = A[j, idx]

valid = ~np.isnan(sim_a[:, 5])
print(f'(a) sim: {valid.sum()}/{len(rows)} ticks executed, {hold_ticks} hold ticks')
for i, name in enumerate(['pan','lift','elbow','wf','roll','gripper']):
    dd = np.abs(sim_a[valid, i] - actual_des[valid, i])
    print(f'  sim vs actual desired {name}: MAD={np.nanmean(dd):.3f} max={np.nanmax(dd):.3f}')
print('  (a) gripper:', closure_stats(sim_a[valid, 5]))

# ---------- (b) full sequential playback ----------
seq_b = A.reshape(-1, 6)  # 390*50 frames
print(f'\n(b) full sequential: {len(seq_b)} frames = {len(seq_b)/HZ:.0f}s of playback')
print('  (b) first 120s gripper:', closure_stats(seq_b[:int(120*HZ), 5]))
print('  (b) all frames gripper:', closure_stats(seq_b[:, 5]))
jumps = np.abs(np.diff(seq_b[:, 5]))
print(f'  (b) gripper |d| at chunk boundaries: med={np.median(jumps[49::50]):.2f} p95={np.percentile(jumps[49::50],95):.2f}')
print('  (b) body:', body_stats(seq_b[:int(120*HZ)]))

# ---------- (c) 25-frame sequential playback ----------
seq_c = np.concatenate([A[i, :25] for i in range(len(A))])
print(f'\n(c) 25-frame sequential: {len(seq_c)} frames = {len(seq_c)/HZ:.0f}s')
print('  (c) first 120s gripper:', closure_stats(seq_c[:int(120*HZ), 5]))
print('  (c) all frames gripper:', closure_stats(seq_c[:, 5]))
jumps_c = np.abs(np.diff(seq_c[:, 5]))
print(f'  (c) gripper |d| at 25-frame boundaries: med={np.median(jumps_c[24::25]):.2f} p95={np.percentile(jumps_c[24::25],95):.2f}')

# ---------- body comparison for (a) ----------
print('\n(a) body (executed sectors):', body_stats(sim_a[valid]))

# summarize as json
out = {
    'a_age_index': {'ticks': int(valid.sum()), 'hold_ticks': hold_ticks,
                    'gripper': closure_stats(sim_a[valid, 5])},
    'b_full_sequential': {'frames': len(seq_b), 'first120_gripper': closure_stats(seq_b[:int(120*HZ), 5]),
                          'all_gripper': closure_stats(seq_b[:, 5])},
    'c_25_sequential': {'frames': len(seq_c), 'first120_gripper': closure_stats(seq_c[:int(120*HZ), 5])},
}
(base.parent.parent/'offline_replay_summary.json').write_text(json.dumps(out, indent=2, ensure_ascii=False))
print('\nsaved offline_replay_summary.json')
