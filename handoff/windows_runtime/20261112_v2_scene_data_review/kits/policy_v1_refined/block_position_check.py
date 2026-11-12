"""Compare yellow-block position in demo reference vs live external camera views."""
import numpy as np
import cv2
from pathlib import Path

def yellow_centroid(img_bgr):
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    m = cv2.inRange(hsv, (15, 80, 80), (35, 255, 255))
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    m = cv2.dilate(m, np.ones((3, 3), np.uint8))
    n, labels, stats, cents = cv2.connectedComponentsWithStats(m, 8)
    if n <= 1:
        return None
    idx = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    x, y, w, h, area = stats[idx]
    cx, cy = cents[idx]
    return dict(area=int(area), cx=float(cx), cy=float(cy), w=int(w), h=int(h))

# demo reference frame (external camera = images.3)
z = np.load(r'G:\LeRobot\offline_kit\samples\ep056_frame00048.npz')
img3 = z['observation.images.3']  # (3,480,640) RGB
demo = np.transpose(img3, (1, 2, 0))  # HWC RGB
demo_bgr = cv2.cvtColor(demo.astype(np.uint8), cv2.COLOR_RGB2BGR)
print('demo ref:', yellow_centroid(demo_bgr), 'size', demo_bgr.shape)

# live: left half = external camera
for tag, p in (('rerun 221952', r'D:\桌面\萌芽杯\policy_v1_refined\results\20261111_221952_936_Grasp_bf16\camera_before.jpg'),
               ('V1 first 221438', r'D:\桌面\萌芽杯\policy_v1_refined\results\20261111_221438_804_Grasp_bf16\camera_before.jpg'),
               ('8x baseline 213457', r'D:\桌面\机械臂\policy_demo8x_trial\results\20261111_213457_906_Grasp_bf16\camera_before.jpg'),
               ('demo6 204401', r'D:\桌面\萌芽杯\policy_demo6_trial\results\20261111_204401_775_Grasp_bf16\camera_before.jpg')):
    img = cv2.imdecode(np.frombuffer(Path(p).read_bytes(), np.uint8), cv2.IMREAD_COLOR)
    h, w = img.shape[:2]
    left = img[:, :w//2]
    print(f'{tag}:', yellow_centroid(left))
