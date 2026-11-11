"""Analyze the 8x grasp run: gripper timing, limiter lag, tracking, chunk dynamics.

Layers kept separate:
  desired  = model raw action target (body: deg; gripper: 0-100 %)
  command  = rate-limited goal actually sent (ticks)
  measured = actual present position (body: deg; gripper: 0-100 %)
"""
import json
import numpy as np
from pathlib import Path

base = Path(r'D:\桌面\机械臂\policy_demo8x_trial\results\20261111_213457_906_Grasp_bf16')
rows = [json.loads(l) for l in (base/'control.jsonl').read_text(encoding='utf-8').splitlines()]
chunks = [json.loads(l) for l in (base/'chunks.jsonl').read_text(encoding='utf-8').splitlines()]
pred = np.load(base/'predictions.npz')

CAL = json.loads((base.parent.parent/'reference_calibration.json').read_text())
MOTORS = ('shoulder_pan','shoulder_lift','elbow_flex','wrist_flex','wrist_roll','gripper')
DEG_PER_TICK = 360/4095
GR_MIN, GR_MAX = CAL['gripper']['range_min'], CAL['gripper']['range_max']
GR_RANGE = GR_MAX-GR_MIN

def g_pct(tick):  # raw ticks -> normalized %
    return (np.asarray(tick)-GR_MIN)/GR_RANGE*100.0

def j_deg(tick, name):
    c = CAL[name]
    return (np.asarray(tick)-(c['range_min']+c['range_max'])/2)*DEG_PER_TICK

elapsed = np.array([r['elapsed_s'] for r in rows])
des = np.array([r['desired'] if r['desired'] is not None else [np.nan]*6 for r in rows], dtype=float)
cmd = np.array([r['raw_command'] for r in rows], dtype=float)
st = np.array([r['state'] for r in rows], dtype=float)  # 5 deg + gripper %
clip = np.array([r['clipped'] for r in rows], dtype=bool)
chunk_ids = np.array([int(r['chunk_id']) for r in rows])
obs_age = np.array([r['observation_age_ms'] for r in rows])
plan_hold = np.array([r['plan_expired_hold'] for r in rows])

print('TICKS', len(rows), 'DURATION', elapsed[-1])
print('mean interval ms', np.diff(elapsed).mean()*1000, 'p95', np.percentile(np.diff(elapsed), 95)*1000)

# ---------- gripper ----------
des_g = des[:,5]
cmd_g = g_pct(cmd[:,5])
meas_g = st[:,5]
print('\n=== GRIPPER (%, 0=closed 100=open) ===')
for name, arr in (('desired', des_g), ('command', cmd_g), ('measured', meas_g)):
    q = np.percentile(arr, [1,5,25,50,75,95,99])
    print(f'{name}: min={arr.min():.2f} max={arr.max():.2f} p1={q[0]:.2f} p5={q[1]:.2f} p25={q[2]:.2f} p50={q[3]:.2f} p75={q[4]:.2f} p95={q[5]:.2f} p99={q[6]:.2f}')

# closure episodes: desired below 50/30/10
def episodes(mask, gap_s=0.15):
    eps, start = [], None
    for i, m in enumerate(mask):
        if m and start is None: start = i
        if not m and start is not None:
            if elapsed[i-1]-elapsed[start] >= gap_s: eps.append((start, i-1))
            start = None
    if start is not None and elapsed[-1]-elapsed[start] >= gap_s: eps.append((start, len(mask)-1))
    return eps

print('\nclosure episodes of desired gripper:')
for thr in (50, 30, 10, 3):
    eps = episodes(des_g < thr)
    total = sum(elapsed[e]-elapsed[s] for s,e in eps)
    print(f'  desired < {thr}%: {len(eps)} episodes, total {total:.2f}s')
    for s,e in eps[:12]:
        print(f'    [{elapsed[s]:6.2f}..{elapsed[e]:6.2f}] {elapsed[e]-elapsed[s]:5.2f}s  min={des_g[s:e+1].min():.2f}%  chunk_ids {chunk_ids[s]}..{chunk_ids[e]}')

# per-second timeline buckets
print('\nGRIPPER TIMELINE (2s buckets): desired min/med, command min/med, measured min/med, |cmd-des| med, clip%')
nb = int(elapsed[-1]//2)+1
for b in range(nb):
    m = (elapsed >= b*2) & (elapsed < (b+1)*2)
    if not m.any(): continue
    print(f'  t={b*2:3d}s des=[{des_g[m].min():5.1f},{np.median(des_g[m]):5.1f}] cmd=[{cmd_g[m].min():5.1f},{np.median(cmd_g[m]):5.1f}] '
          f'meas=[{meas_g[m].min():5.1f},{np.median(meas_g[m]):5.1f}] |c-d|med={np.median(np.abs(cmd_g[m]-des_g[m])):4.1f} clip={clip[m,5].mean()*100:3.0f}%')

# ---------- body tracking & limiter ----------
print('=== BODY TRACKING (measured vs sent command) ===')
for i, name in enumerate(MOTORS[:5]):
    if i == 4:
        print(f'{name}: fixed; |meas-0.22deg| max drift {np.abs(st[:,4]-st[0,4]).max():.3f} deg')
        continue
    err = np.abs(st[:,i] - j_deg(cmd[:,i], name))
    print(f'{name}: err med={np.median(err):.2f}° p95={np.percentile(err,95):.2f}° max={err.max():.2f}°  >2deg:{int((err>2).sum())} >4deg:{int((err>4).sum())}')

# command rate vs desired (limiter lag in degrees)
print('\n=== LIMITER LAG (desired vs command, degrees) ===')
for i, name in enumerate(MOTORS[:4]):
    lag = np.abs(des[:,i] - j_deg(cmd[:,i], name))
    print(f'{name}: |des-cmd| med={np.median(lag):.2f}° p95={np.percentile(lag,95):.2f}° max={lag.max():.2f}° clipped={clip[:,i].mean()*100:.0f}%')

# gripper limiter lag
lag_g = des_g - cmd_g  # + means command is more open than desired
print(f'gripper: |des-cmd| med={np.median(np.abs(lag_g)):.1f}% p95={np.percentile(np.abs(lag_g),95):.1f}% max={np.abs(lag_g).max():.1f}% clipped={clip[:,5].mean()*100:.0f}%')

# gripper measured following command
gap_mc = meas_g - cmd_g
print(f'gripper: |meas-cmd| med={np.median(np.abs(gap_mc)):.1f}% p95={np.percentile(np.abs(gap_mc),95):.1f}% max={np.abs(gap_mc).max():.1f}%')

# stalls: measured flat while command keeps closing
dmeas = np.diff(meas_g)
dcmd = np.diff(cmd_g)
stall = (np.abs(dmeas) < 0.05) & (dcmd < -0.2)
print(f'gripper contact-stall ticks (meas flat, cmd closing): {stall.sum()}')

# ---------- chunk dynamics ----------
print('\n=== CHUNK DYNAMICS ===')
obs_age = np.array([r['observation_age_ms'] for r in rows])
print(f'chunks={len(chunks)} chunks/s={len(chunks)/elapsed[-1]:.2f} obs_age p50={np.median(obs_age):.0f}ms p95={np.percentile(obs_age,95):.0f}ms max={obs_age.max():.0f}ms')
switch = np.sum(np.diff(chunk_ids) != 0)
print(f'chunk switches={switch} ({switch/elapsed[-1]:.1f}/s)')
a = pred['actions']  # (390,50,6)
min_per_chunk = a[:,:,5].min(axis=1)
max_per_chunk = a[:,:,5].max(axis=1)
print(f'chunks whose gripper action reaches <50%: {int((min_per_chunk<50).sum())}/{len(a)}; <30%: {int((min_per_chunk<30).sum())}; <10%: {int((min_per_chunk<10).sum())}')
print(f'gripper action range per chunk: min={min_per_chunk.min():.1f} max={max_per_chunk.max():.1f}')

# reversals of desired gripper (sign changes, min 0.5s apart)
d = np.diff(des_g)
sign = np.sign(d[(np.abs(d) > 1.0)])
rev = np.sum(sign[1:] != sign[:-1])
print(f'desired gripper reversals (|delta|>1%): {rev} in {elapsed[-1]:.0f}s')

# plan hold
print(f'plan_expired_hold ticks: {int(plan_hold.sum())}')
