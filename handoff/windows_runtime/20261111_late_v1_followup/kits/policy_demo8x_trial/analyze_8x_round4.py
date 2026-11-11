"""Round 4: which frames of each chunk get executed + demo gripper depth reference."""
import json
import numpy as np
from pathlib import Path

base = Path(r'D:\桌面\机械臂\policy_demo8x_trial\results\20261111_213457_906_Grasp_bf16')
rows = [json.loads(l) for l in (base/'control.jsonl').read_text(encoding='utf-8').splitlines()]
cid = np.array([int(r['chunk_id']) for r in rows])
idx = np.array([int(r['chunk_index']) for r in rows])
obs_age = np.array([r['observation_age_ms'] for r in rows])

print('chunk_index used: p1/p50/p99 of first-use index per chunk:')
first = {}
for i in range(len(cid)):
    first.setdefault(cid[i], idx[i])
vals = np.array(list(first.values()))
print(f'  first index: p1={np.percentile(vals,1):.0f} p50={np.percentile(vals,50):.0f} p95={np.percentile(vals,95):.0f} max={vals.max()}')
used = np.bincount(cid)
print(f'ticks per chunk: p5={np.percentile(used,5):.0f} p50={np.percentile(used,50):.0f} p95={np.percentile(used,95):.0f}')
frames = np.bincount(idx)
print(f'frame index histogram (executed): frames 0-14: {frames[:15].sum()} ticks, 15-24: {frames[15:25].sum()}, 25-49: {frames[25:].sum()}')
print(f'total ticks: {len(idx)}; fraction executed from frames 0-24: {(frames[:25].sum()/frames.sum()):.1%}')

# demo samples
samples = Path(r'G:\LeRobot\offline_kit\samples')
fs = sorted(samples.glob('*.npz'))
print(f'\ndemo windows: {len(fs)}')
gs_action, gs_state = [], []
for f in fs:
    z = np.load(f)
    gs_action.append(z['action'][:,5])
    gs_state.append(z['observation.state'][5])
gs_action = np.array(gs_action)  # (33,50)
gs_state = np.array(gs_state)    # (33,)
print('demo gripper ACTIONS per window (min/median/max over 33 windows):')
print(f'  window min:  p5={np.percentile(gs_action.min(1),5):.1f} p50={np.percentile(gs_action.min(1),50):.1f} max={gs_action.min(1).max():.1f}')
print(f'  window med:  p5={np.percentile(np.median(gs_action,1),5):.1f} p50={np.percentile(np.median(gs_action,1),50):.1f} max={np.median(gs_action,1).max():.1f}')
print(f'  window max:  p5={np.percentile(gs_action.max(1),5):.1f} p50={np.percentile(gs_action.max(1),50):.1f} max={gs_action.max(1).max():.1f}')
print('demo gripper STATE (observation): p5=%.1f p50=%.1f p95=%.1f' % tuple(np.percentile(gs_state,[5,50,95])))
# how long does demo action stay below 10% within a window?
for thr in (10, 5):
    longest = []
    for w in gs_action:
        m = w < thr
        best = cur = 0
        for v in m:
            cur = cur+1 if v else 0
            best = max(best, cur)
        longest.append(best)
    print(f'  demo windows with gripper<{thr}% held: longest run frames p50={np.median(longest):.0f} p95={np.percentile(longest,95):.0f} (of 50)')
