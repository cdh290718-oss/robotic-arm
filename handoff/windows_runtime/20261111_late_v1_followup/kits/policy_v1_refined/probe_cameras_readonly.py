"""Enumerate currently available DSHOW cameras and save local-only diagnostic frames."""
import json
import time
from datetime import datetime
from pathlib import Path
import cv2
import numpy as np
from pygrabber.dshow_graph import FilterGraph

out=Path(__file__).parent/'camera_checks'/datetime.now().strftime('%Y%m%d_%H%M%S')
out.mkdir(parents=True)
rows=[]
for index,name in enumerate(FilterGraph().get_input_devices()):
    row={'index':index,'name':name,'backend':'DSHOW'};rows.append(row)
    cap=cv2.VideoCapture(index,cv2.CAP_DSHOW)
    try:
        row['opened']=cap.isOpened()
        if not row['opened']:continue
        cap.set(cv2.CAP_PROP_FOURCC,cv2.VideoWriter_fourcc(*'MJPG'))
        cap.set(cv2.CAP_PROP_FRAME_WIDTH,640);cap.set(cv2.CAP_PROP_FRAME_HEIGHT,480)
        cap.set(cv2.CAP_PROP_FPS,30)
        raw=int(cap.get(cv2.CAP_PROP_FOURCC));row['fourcc']=''.join(chr((raw >> 8*i)&255) for i in range(4))
        row['width']=cap.get(cv2.CAP_PROP_FRAME_WIDTH);row['height']=cap.get(cv2.CAP_PROP_FRAME_HEIGHT);row['fps_reported']=cap.get(cv2.CAP_PROP_FPS)
        start=time.perf_counter();frame=None;count=0
        while count<20 and time.perf_counter()-start<3:
            ok,candidate=cap.read()
            if ok:frame=candidate;count+=1
        row['frames']=count
        if frame is not None:
            gray=frame.astype(np.float32).mean(axis=2)
            row['mean']=float(gray.mean());row['p99']=float(np.percentile(gray,99));row['near_black']=row['p99']<18
            ok,encoded=cv2.imencode('.jpg',frame)
            if ok:(out/f'camera_{index}.jpg').write_bytes(encoded.tobytes())
    finally:cap.release()
    print(json.dumps(row,ensure_ascii=False),flush=True)
(out/'summary.json').write_text(json.dumps(rows,ensure_ascii=False,indent=2),encoding='utf-8')
print('LOCAL RESULTS:',out)
