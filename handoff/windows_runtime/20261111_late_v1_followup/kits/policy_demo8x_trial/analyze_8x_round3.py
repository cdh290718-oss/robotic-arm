"""Round 3: per-chunk gripper profile, measured reversals, arm pose near closure, demo comparison."""
import json
import numpy as np
from pathlib import Path

base = Path(r'D:\桌面\机械臂\policy_demo8x_trial\results\20261111_213457_906_Grasp_bf16')
rows = [json.loads(l) for l in (base/'control.jsonl').read_text(encoding='utf-8').splitlines()]
chunks = [json.loads(l) for l in (base/'chunks.jsonl').read_text(encoding='utf-8').splitlines()]
pred = np.load(base/'predictions.npz')
el = np.array([r['elapsed_s'] for r in rows])
des = np.array([r['desired'] if r['desired'] else [np.nan]*6 for r in rows], float)
cmd = np.array([r['raw_command'] for r in rows], float)
st = np.array([r['state'] for r in rows], float)
cid = np.array([int(r['chunk_id']) for r in rows])
CAL = json.loads((base.parent.parent/'reference_calibration.json').read_text())
DEG = 360/4095
def jd(t, n):
    c = CAL[n]
    return (np.asarray(t)-(c['range_min']+c['range_max'])/2)*DEG
GN = ('shoulder_pan','shoulder_lift','elbow_flex','wrist_flex','wrist_roll')

# ---- per-chunk gripper profile ----
A = pred['actions']  # (390,50,6)
g = A[:,:,5]
rng = g.max(axis=1)-g.min(axis=1)
start = g[:,0]; mid = g[:,25]; end = g[:,-1]
print('per-chunk gripper (50-frame plan):')
print(f'  internal range: med={np.median(rng):.1f} p90={np.percentile(rng,90):.1f} max={rng.max():.1f}')
print(f'  start->end drop: med={np.median(start-end):.1f}')
print(f'  chunk[0] first 12:', np.round(g[0,:12],1))
print(f'  chunk[0] at 25..37:', np.round(g[0,25:38],1))
stab = (rng < 5).sum()
print(f'  chunks with internal gripper range <5%: {stab}/{len(g)}')

# ---- measured gripper reversals ----
gm = st[:,5]
dg = np.diff(gm); s = np.sign(dg[np.abs(dg) > 0.5])
print(f'\nmeasured gripper reversals (|d|>0.5%): {int(np.sum(s[1:]!=s[:-1]))}')
print(f'measured gripper min={gm.min():.1f} max={gm.max():.1f}')

# ---- body pose around the 80.6-82.8 closure ----
print('\narm pose during closure window (t=80.6..82.8):')
m = (el >= 80.6) & (el <= 82.8)
for i, n in enumerate(GN[:4]):
    print(f'  {n}: {np.median(jd(cmd[m,i], n)):.1f}° .. {jd(cmd[m,i], n).max():.1f} (median..max)')
print(f'  gripper: desired med={np.median(des[m,5]):.1f}% cmd med={np.median(gm[m]):.1f}% meas med={np.median(gm[m]):.1f}%')

# ---- demo gripper pattern: find grasp episodes in offline samples ----
samples = Path(r'G:\LeRobot\offline_kit\samples')
print('\nsamples dir exists:', samples.exists())
if samples.exists():
    fs = sorted(samples.glob('*.npz'))
    print('npz files:', len(fs), [f.name for f in fs[:5]])
    if fs:
        z = np.load(fs[0], allow_pickle=True)
        print('keys:', z.files)
        for k in z.files[:6]:
            v = z[k]
            print(f'  {k}: {v.shape} {v.dtype}')
