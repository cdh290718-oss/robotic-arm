"""Live two-camera preview and saved demonstration comparison; no robot connection."""
import argparse
import threading
import time
from pathlib import Path
import tkinter as tk

import cv2
import numpy as np
from PIL import Image, ImageTk

REFERENCE = Path(r'G:\LeRobot\offline_kit\samples\ep056_frame00048.npz')
WINDOW = 'SO101 cameras - Q or Esc: quit | R: reference on/off'
W, H = 640, 480


def label(frame, text):
    out = cv2.resize(frame, (W, H))
    cv2.rectangle(out, (0, 0), (W, 31), (0, 0, 0), -1)
    cv2.putText(out, text, (8, 22), cv2.FONT_HERSHEY_SIMPLEX, .56, (255, 255, 255), 1, cv2.LINE_AA)
    return out


def empty(text):
    frame = np.zeros((H, W, 3), dtype=np.uint8)
    cv2.putText(frame, text, (25, H//2), cv2.FONT_HERSHEY_SIMPLEX, .7, (0, 180, 255), 2, cv2.LINE_AA)
    return frame


def get_reference(path):
    if not path.is_file():
        return None
    with np.load(path, allow_pickle=False) as sample:
        images = [sample['observation.images.3'], sample['observation.images.1']]
        frames = []
        for a in images:
            if a.shape != (3, H, W) or a.dtype != np.uint8:
                raise ValueError('Unexpected reference image format')
            frames.append(cv2.cvtColor(np.transpose(a, (1, 2, 0)), cv2.COLOR_RGB2BGR))
    return frames


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--external', type=int, default=2, help='DSHOW index for external view')
    p.add_argument('--wrist', type=int, default=0, help='DSHOW index for wrist view')
    p.add_argument('--reference', type=Path, default=REFERENCE)
    args = p.parse_args()
    if args.external == args.wrist:
        p.error('external and wrist cameras need different indices')
    try:
        ref = get_reference(args.reference)
    except (ValueError, KeyError, OSError) as e:
        print('Reference unavailable:', e, flush=True)
        ref = None
    show_ref = [ref is not None]
    stop = threading.Event()
    lock = threading.Lock()
    frames = [None, None]
    timestamps = [0., 0.]

    def capture(slot, index):
        cap = cv2.VideoCapture(index, cv2.CAP_DSHOW)
        try:
            if not cap.isOpened():
                print(f'Camera {index} unavailable; close other apps or adjust index.', flush=True)
                return
            cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, W)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, H)
            cap.set(cv2.CAP_PROP_FPS, 30)
            while not stop.is_set():
                ok, frame = cap.read()
                if not ok or frame is None:
                    time.sleep(.05)
                    continue
                with lock:
                    frames[slot] = frame
                    timestamps[slot] = time.monotonic()
        finally:
            cap.release()

    print('Opening cameras. R: reference; Q/Esc: exit. No motor connection or recording.', flush=True)
    root = tk.Tk()
    root.title(WINDOW)
    image_label = tk.Label(root)
    image_label.pack()
    text = tk.Label(root, text='R: show/hide reference     Q / Esc: exit     Top: reference, bottom: live')
    text.pack()

    def close(_event=None):
        stop.set()
        root.destroy()

    def toggle(_event=None):
        if ref is not None:
            show_ref[0] = not show_ref[0]

    root.protocol('WM_DELETE_WINDOW', close)
    for key in ('<Escape>', '<q>', '<Q>'):
        root.bind(key, close)
    for key in ('<r>', '<R>'):
        root.bind(key, toggle)
    for slot, index in enumerate((args.external, args.wrist)):
        threading.Thread(target=capture, args=(slot,index), daemon=True).start()

    labels = [f'LIVE external (camera index {args.external})', f'LIVE wrist (camera index {args.wrist})']
    def update():
        if stop.is_set():
            return
        with lock:
            copies = [f.copy() if f is not None and time.monotonic()-t < 2 else None
                      for f,t in zip(frames,timestamps)]
        live = [label(f if f is not None else empty('NO CAMERA FRAME'), text)
                for f,text in zip(copies,labels)]
        panel = np.hstack(live)
        if show_ref[0]:
            reference = np.hstack([label(ref[0], 'REFERENCE external: ep056 ~10%'),
                                   label(ref[1], 'REFERENCE wrist: ep056 ~10%')])
            panel = np.vstack((reference, panel))
        h,w=panel.shape[:2]
        scale=min(1., (root.winfo_screenwidth()-60)/w, (root.winfo_screenheight()-120)/h)
        rgb=cv2.cvtColor(panel,cv2.COLOR_BGR2RGB)
        image=Image.fromarray(rgb).resize((max(1,int(w*scale)),max(1,int(h*scale))))
        photo=ImageTk.PhotoImage(image)
        image_label.configure(image=photo)
        image_label.image=photo
        root.after(33,update)
    root.after(0,update)
    root.mainloop()
    stop.set()
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
