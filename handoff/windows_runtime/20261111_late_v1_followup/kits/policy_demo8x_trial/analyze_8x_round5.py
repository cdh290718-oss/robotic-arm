"""Round 5: where in the chunk plans does the gripper close? Late frames never executed?"""
import json
import numpy as np
from pathlib import Path

base = Path(r'D:\桌面\机械臂\policy_demo8x_trial\results\20261111_213457_906_Grasp_bf16')
pred = np.load(base/'predictions.npz')
A = pred['actions']  # (390,50,6)
g = A[:,:,5]

early = g[:, :10].mean(axis=1)   # frames that get executed
late = g[:, 25:].mean(axis=1)    # frames that almost never execute

print('chunk early(0-9) vs late(25-49) gripper mean:')
print(f'  early<10%: {(early<10).sum()}/{len(g)}  early>15%: {(early>15).sum()}')
print(f'  late<10%: {(late<10).sum()}/{len(g)}  late>15%: {(late>15).sum()}')
m = (early > 15) & (g.min(axis=1) < 10)
print(f'chunks that start open (>15%) but contain a close (<10%) somewhere: {m.sum()}')
if m.sum():
    sub = g[m]
    close_frame = np.argmin(sub, axis=1)
    print(f'  position of min-gripper frame within these chunks: med={np.median(close_frame):.0f} p90={np.percentile(close_frame,90):.0f}')
    hold = (sub < 10).sum(axis=1)
    print(f'  frames <10% per such chunk: med={np.median(hold):.0f} p90={np.percentile(hold,90):.0f} (of 50)')

# simulation: full sequential chunk playback vs actual (stitched) playback
def run(steps):
    gs = np.concatenate([g[i, :steps[i]] for i in range(len(g))])
    dur = (gs < 10).mean() * (50/30)  # seconds of closed fraction in "playback"
    return gs, dur

for label, steps in (('actual-ish (first 10 frames/chunk)', np.full(len(g), 10)),
                     ('first 25 frames/chunk', np.full(len(g), 25)),
                     ('full 50 frames/chunk', np.full(len(g), 50))):
    gs, dur = run(steps)
    # longest continuous run <10%
    m2 = gs < 10
    best = cur = 0
    for v in m2:
        cur = cur+1 if v else 0
        best = max(best, cur)
    print(f'{label}: <10% longest run = {best/30:.2f}s, total <10% = {m2.sum()/30:.1f}s')

# per-joint chunk plan: late frames vs early frames direction (which joints move later?)
print('\nbody chunk early(0-9) vs late(25-49) delta (deg):')
for i, name in enumerate(['pan','lift','elbow','wrist_flex']):
    de = A[:, 25:, i].mean(axis=1) - A[:, :10, i].mean(axis=1)
    print(f'  {name}: med={np.median(de):6.2f} p5={np.percentile(de,5):6.2f} p95={np.percentile(de,95):6.2f}')
