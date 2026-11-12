"""Scene alignment check: demo reference (ep056_frame00048) vs live frames."""
import numpy as np
import cv2
from pathlib import Path

def yellow(img_bgr):
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    m = cv2.inRange(hsv, (15, 80, 80), (35, 255, 255))
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    n, labels, stats, cents = cv2.connectedComponentsWithStats(m, 8)
    if n <= 1:
        return None
    idx = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    x, y, w, h, area = stats[idx]
    return dict(area=int(area), cx=float(cents[idx][0]), cy=float(cents[idx][1]), w=int(w), h=int(h))

def bright(img_bgr):
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    m = cv2.inRange(hsv, (0, 0, 200), (180, 60, 255))
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((7, 7), np.uint8))
    n, labels, stats, cents = cv2.connectedComponentsWithStats(m, 8)
    blobs = []
    for i in range(1, n):
        x, y, w, h, area = stats[i]
        if area > 3000:
            blobs.append((int(area), float(cents[i][0]), float(cents[i][1]), int(w), int(h)))
    return sorted(blobs, reverse=True)[:4]

def sim(a, b):
    ga = cv2.cvtColor(cv2.resize(a, (64, 48)), cv2.COLOR_BGR2GRAY).astype(np.float32)
    gb = cv2.cvtColor(cv2.resize(b, (64, 48)), cv2.COLOR_BGR2GRAY).astype(np.float32)
    a2, b2 = ga - ga.mean(), gb - gb.mean()
    return float((a2 * b2).sum() / (np.sqrt((a2**2).sum() * (b2**2).sum()) + 1e-9))

z = np.load(r'G:\LeRobot\offline_kit\samples\ep056_frame00048.npz')
demo_ext = cv2.cvtColor(np.transpose(z['observation.images.3'], (1, 2, 0)).astype(np.uint8), cv2.COLOR_RGB2BGR)
demo_wr = cv2.cvtColor(np.transpose(z['observation.images.1'], (1, 2, 0)).astype(np.uint8), cv2.COLOR_RGB2BGR)

live_dir = Path(r'D:\桌面\萌芽杯\policy_v1_refined\camera_checks\20261112_210517')
live_wr = cv2.imdecode(np.frombuffer((live_dir/'camera_0.jpg').read_bytes(), np.uint8), cv2.IMREAD_COLOR)
live_ext = cv2.imdecode(np.frombuffer((live_dir/'camera_2.jpg').read_bytes(), np.uint8), cv2.IMREAD_COLOR)

print('== EXTERNAL view (demo reference vs live) ==')
print('demo:', yellow(demo_ext))
print('live:', yellow(live_ext))
print('demo bright blobs:', bright(demo_ext))
print('live bright blobs:', bright(live_ext))
print('global corr (downsampled):', round(sim(demo_ext, live_ext), 3))

print('\n== WRIST view (demo reference vs live) ==')
print('demo wrist yellow:', yellow(demo_wr))
print('live wrist yellow:', yellow(live_wr))
print('demo wrist bright:', bright(demo_wr))
print('live wrist bright:', bright(live_wr))
print('global corr (downsampled):', round(sim(demo_wr, live_wr), 3))
