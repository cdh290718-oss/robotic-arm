"""Read-only live inputs + SmolVLA predictions. Never executes predicted actions."""
from __future__ import annotations
import argparse
from collections import deque
from datetime import datetime, timezone
import hashlib
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from functools import partial
import importlib.metadata
import json
import os
from pathlib import Path
import subprocess
import sys
import threading
import time
import traceback
import webbrowser
import zipfile
from readonly_bus import MOTORS, ReadOnlyArm, validate_calibration

HERE=Path(__file__).resolve().parent
BASE='HuggingFaceTB/SmolVLM2-500M-Video-Instruct'
REV='7b375e1b73b11138ff12fe22c8f2822d8fe03467'
TASK='Pick up the yellow cube and place it into the mesh container.'
RENAME={'observation.images.3':'observation.images.camera1','observation.images.1':'observation.images.camera2'}


def write_json(path,data):
    Path(path).write_text(json.dumps(data,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')


def digest(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
    return h.hexdigest()


def run_command(cmd):
    try:
        p=subprocess.run(cmd,capture_output=True,text=True,errors='replace',timeout=20)
        return {'returncode':p.returncode,'stdout':p.stdout,'stderr':p.stderr}
    except Exception as e:return {'error':str(e)}


def power_status():
    if os.name!='nt':return {'available':False}
    import ctypes
    class Status(ctypes.Structure):
        _fields_=[('ACLineStatus',ctypes.c_ubyte),('BatteryFlag',ctypes.c_ubyte),
                  ('BatteryLifePercent',ctypes.c_ubyte),('SystemStatusFlag',ctypes.c_ubyte),
                  ('BatteryLifeTime',ctypes.c_ulong),('BatteryFullLifeTime',ctypes.c_ulong)]
    s=Status()
    if not ctypes.windll.kernel32.GetSystemPowerStatus(ctypes.byref(s)):return {'available':False}
    return {name:getattr(s,name) for name,_ in s._fields_}


class Telemetry:
    def __init__(self,output):self.output=output;self.proc=None;self.handle=None;self.info={}
    def start(self):
        fields='timestamp,pstate,utilization.gpu,memory.used,memory.free,power.draw,power.limit,temperature.gpu,clocks.current.graphics,clocks.current.memory'
        test=run_command(['nvidia-smi',f'--query-gpu={fields}','--format=csv'])
        self.info['initial_query']=test
        if test.get('returncode')!=0:
            fields='timestamp,pstate,utilization.gpu,memory.used,memory.free,temperature.gpu'
        self.info['fields']=fields
        try:
            self.handle=(self.output/'gpu_telemetry.csv').open('w',encoding='utf-8')
            self.proc=subprocess.Popen(['nvidia-smi',f'--query-gpu={fields}','--format=csv','-lms','1000'],
                                       stdout=self.handle,stderr=subprocess.STDOUT,
                                       creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
        except Exception:self.info['error']=traceback.format_exc()
    def close(self):
        if self.proc is not None:
            if self.proc.poll() is None:
                self.proc.terminate()
                try:self.proc.wait(timeout=3)
                except subprocess.TimeoutExpired:self.proc.kill();self.proc.wait(timeout=3)
            self.info['returncode']=self.proc.returncode
        if self.handle:self.handle.close()


class Preview:
    def __init__(self,output,enabled):
        self.output=output;self.server=None;self.url=None
        html='''<!doctype html><meta charset="utf-8"><title>SO101 read-only preview</title>
<style>body{font:18px sans-serif;margin:24px;background:#161b22;color:#eee}img{max-width:100%}pre{white-space:pre-wrap}</style>
<h2>Read-only input preview — no motor commands</h2><p>Left: EXTERNAL camera 3 → camera1. Right: WRIST camera 1 → camera2.</p>
<img id="view"><pre id="state">Waiting for inputs...</pre><p>Stop in PowerShell with Ctrl+C. Closing this page does not stop the probe.</p>
<script>setInterval(async()=>{document.getElementById('view').src='latest_preview.jpg?t='+Date.now();try{let r=await fetch('preview_status.json?t='+Date.now());document.getElementById('state').textContent=JSON.stringify(await r.json(),null,2)}catch(e){}},1000)</script>'''
        (output/'preview.html').write_text(html,encoding='utf-8')
        if enabled:
            class QuietHandler(SimpleHTTPRequestHandler):
                def log_message(self,*args):pass
            self.server=ThreadingHTTPServer(('127.0.0.1',0),partial(QuietHandler,directory=str(output.resolve())))
            threading.Thread(target=self.server.serve_forever,daemon=True).start()
            self.url=f'http://127.0.0.1:{self.server.server_port}/preview.html'
            print('PREVIEW: '+self.url,flush=True)
            webbrowser.open(self.url)
    def update(self,snapshot,record,save_name=None):
        import cv2
        import numpy as np
        images=[]
        for key,label in [('3','EXTERNAL / camera 3 -> model camera1'),('1','WRIST / camera 1 -> model camera2')]:
            image=cv2.cvtColor(snapshot['images'][key],cv2.COLOR_RGB2BGR)
            cv2.rectangle(image,(0,0),(640,34),(0,0,0),-1)
            cv2.putText(image,label,(10,23),cv2.FONT_HERSHEY_SIMPLEX,0.55,(0,255,255),1)
            images.append(image)
        panel=np.hstack(images)
        ok,buffer=cv2.imencode('.jpg',panel,[cv2.IMWRITE_JPEG_QUALITY,85])
        if not ok:raise RuntimeError('Preview JPEG encoding failed')
        temp=self.output/'preview.tmp'
        temp.write_bytes(buffer.tobytes())
        for attempt in range(5):
            try:
                os.replace(temp,self.output/'latest_preview.jpg')
                break
            except PermissionError:
                if attempt==4:
                    # Windows may briefly hold a JPEG open while serving the preview.
                    # Skip this preview update; inference/state logging remain valid.
                    break
                time.sleep(.01)
        if save_name:(self.output/save_name).write_bytes(buffer.tobytes())
        write_json(self.output/'preview_status.json',record)
    def close(self):
        if self.server:self.server.shutdown();self.server.server_close()


class Hardware:
    def __init__(self,args,report):
        self.args=args;self.report=report;self.cameras={};self.arm=None
        self.stop=threading.Event();self.thread=None;self.lock=threading.Lock()
        self.latest=None;self.error=None;self.state_count=0;self.read_ms=[]
    def start(self):
        from lerobot.cameras.opencv.configuration_opencv import OpenCVCameraConfig
        from lerobot.cameras.opencv.camera_opencv import OpenCVCamera
        from lerobot.robots.so_follower.config_so_follower import SOFollowerRobotConfig
        # Reject a differing local unit default instead of silently changing training semantics.
        if not SOFollowerRobotConfig(port='unused').use_degrees:
            raise ValueError('Local SOFollower default use_degrees differs from the training source')
        config=json.loads(self.args.robot_config.read_text(encoding='utf-8-sig'))
        selected={str(c['name']):c for c in config['cameras'] if str(c['name']) in ('1','3')}
        if set(selected)!= {'1','3'} or len([c for c in config['cameras'] if str(c['name']) in ('1','3')])!=2:
            raise ValueError('Robot config must contain exactly one camera named 1 and one named 3')
        if selected['1']['camera_index']==selected['3']['camera_index']:
            raise ValueError('Two camera roles point to the same device index')
        calibration=validate_calibration(self.args.calibration,HERE/'reference_calibration.json')
        self.report['configuration']={'robot_config':str(self.args.robot_config),'robot_config_sha256':digest(self.args.robot_config),
            'calibration':str(self.args.calibration),'calibration_sha256':digest(self.args.calibration),
            'cameras':selected,'port':self.args.port or config['follower_port'],'rename_map':RENAME,
            'software_timestamp_note':'Camera timestamp after decode, not hardware exposure time; no hardware synchronization claim.'}
        class TrackedCamera(OpenCVCamera):
            def __init__(self,cfg):
                super().__init__(cfg)
                self.arrivals=deque(maxlen=30000);self.arrival_lock=threading.Lock();self.delivered=0
            def _postprocess_image(self,frame,*args,**kwargs):
                frame=super()._postprocess_image(frame,*args,**kwargs)
                with self.arrival_lock:self.arrivals.append(time.perf_counter());self.delivered+=1
                return frame
        # No automatic search/reordering, recalibration, exposure or focus changes.
        for name in ('3','1'):
            c=selected[name]
            if (c.get('width'),c.get('height'),c.get('fps'),c.get('fourcc'))!=(640,480,30,'MJPG'):
                raise ValueError(f'Camera {name} differs from the trained 640x480/30fps/MJPG setup')
            camera=TrackedCamera(OpenCVCameraConfig(index_or_path=c['camera_index'],width=640,height=480,
                                fps=30,fourcc='MJPG',color_mode='rgb',backend=c.get('backend',0)))
            self.cameras[name]=camera
            camera.connect()
            cap=camera.videocapture
            self.report.setdefault('opened_cameras',{})[name]={'backend':cap.getBackendName(),
                 'width':cap.get(3),'height':cap.get(4),'fps_reported':cap.get(5)}
        self.arm=ReadOnlyArm(self.args.port or config['follower_port'],calibration)
        self.arm.connect()
        self.thread=threading.Thread(target=self._read_states,daemon=True)
        self.thread.start()
        deadline=time.perf_counter()+3
        while self.latest is None and self.error is None and time.perf_counter()<deadline:time.sleep(.01)
        if self.error:raise RuntimeError(self.error)
        if self.latest is None:raise TimeoutError('No initial robot state')
    def _read_states(self):
        import numpy as np
        try:
            with (self.args.output/'states.jsonl').open('w',encoding='utf-8') as f:
                while not self.stop.is_set():
                    begin=time.perf_counter();values=self.arm.read();end=time.perf_counter()
                    if not np.isfinite(values).all():raise ValueError('Nonfinite measured joint state')
                    row={'wall_time_ns':time.time_ns(),'read_start':begin,'read_end':end,
                         'timestamp':(begin+end)/2,'state':values,'read_ms':(end-begin)*1000}
                    with self.lock:self.latest=row
                    self.state_count+=1;self.read_ms.append(row['read_ms'])
                    f.write(json.dumps(row)+'\n');f.flush()
                    self.stop.wait(max(0,1/30-(time.perf_counter()-begin)))
        except Exception:self.error=traceback.format_exc();self.stop.set()
    def snapshot(self):
        import numpy as np
        if self.error:raise RuntimeError(self.error)
        with self.lock:state=dict(self.latest)
        images={};timestamps={}
        for name,camera in self.cameras.items():
            if camera.thread is None or not camera.thread.is_alive():raise RuntimeError(f'Camera {name} thread stopped')
            with camera.frame_lock:
                if camera.latest_frame is None:raise RuntimeError(f'Camera {name} has no frame')
                images[name]=camera.latest_frame.copy();timestamps[name]=camera.latest_timestamp
            if images[name].shape!=(480,640,3) or images[name].dtype!=np.uint8:
                raise ValueError(f'Unexpected RGB shape/dtype for camera {name}')
        now=time.perf_counter()
        ages={k:(now-v)*1000 for k,v in timestamps.items()}
        state_age=(now-state['timestamp'])*1000
        if max(*ages.values(),state_age)>250:raise TimeoutError(f'Stale input: cameras={ages}, state={state_age:.1f} ms')
        return {'images':images,'state':np.array(state['state'],dtype=np.float32),
                'image_timestamps':timestamps,'state_timestamp':state['timestamp'],
                'camera_age_ms':ages,'state_age_ms':state_age,
                'pair_timestamp_skew_ms':abs(timestamps['1']-timestamps['3'])*1000}
    def close(self):
        self.stop.set()
        if self.thread:self.thread.join(timeout=3)
        cleanup=[]
        if self.arm:
            try:
                if self.thread and self.thread.is_alive():raise RuntimeError('State thread did not stop; no further bus reads issued')
                self.arm.close()
            except Exception:cleanup.append(traceback.format_exc())
            self.report['read_only_bus']=self.arm.report()
        for name,camera in self.cameras.items():
            with camera.arrival_lock:arrivals=list(camera.arrivals)
            elapsed=arrivals[-1]-arrivals[0] if len(arrivals)>1 else 0
            self.report.setdefault('camera_capture',{})[name]={'frames':camera.delivered,
                   'software_delivery_fps':(len(arrivals)-1)/elapsed if elapsed else None}
            try:
                if camera.is_connected:camera.disconnect()
            except Exception:cleanup.append(traceback.format_exc())
        self.report['state_reads']={'count':self.state_count,'read_latency_ms':percentiles(self.read_ms)}
        if cleanup:self.report['cleanup_errors']=cleanup


class Replay:
    def __init__(self,args):
        import numpy as np
        self.items=[];self.i=0
        manifest=json.loads((args.offline_kit/'samples/manifest.json').read_text())
        for row in manifest['samples']:
            file=args.offline_kit/'samples'/row['file']
            if digest(file)!=row['sha256']:raise ValueError('Replay sample hash mismatch')
            with np.load(file,allow_pickle=False) as z:
                self.items.append({'images':{c:z[f'observation.images.{c}'].transpose(1,2,0).copy() for c in ('1','3')},
                                   'state':z['observation.state'].copy(),'sample_file':row['file']})
    def start(self):pass
    def snapshot(self):
        row=dict(self.items[self.i%len(self.items)]);self.i+=1;now=time.perf_counter()
        row.update(image_timestamps={'1':now,'3':now},state_timestamp=now,camera_age_ms={'1':0,'3':0},state_age_ms=0,pair_timestamp_skew_ms=0)
        return row
    def close(self):pass


class Engine:
    def __init__(self,args,report):
        import torch
        from huggingface_hub import snapshot_download
        from lerobot.configs import PreTrainedConfig
        from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
        from lerobot.policies.factory import make_pre_post_processors
        from safetensors.torch import load_file
        import numpy as np
        torch.set_num_threads(2)
        if not torch.cuda.is_available():raise RuntimeError('CUDA unavailable')
        if args.precision=='bf16' and not torch.cuda.is_bf16_supported():raise RuntimeError('BF16 unavailable')
        if args.gpu_memory_fraction:torch.cuda.set_per_process_memory_fraction(args.gpu_memory_fraction)
        self.dtype=torch.bfloat16 if args.precision=='bf16' else torch.float16
        folder=args.model_dir or args.offline_kit/'models/010000/pretrained_model'
        expected=json.loads((HERE/'checkpoint_manifest.json').read_text())['checkpoints']['010000']['files']
        for name,ref in expected.items():
            if not (folder/name).is_file() or digest(folder/name)!=ref['sha256']:raise ValueError(f'10k file missing or changed: {folder/name}')
        snapshot=snapshot_download(BASE,revision=REV,local_files_only=True)
        cfg=PreTrainedConfig.from_pretrained(str(folder),local_files_only=True)
        cfg.device='cuda';cfg.pretrained_path=folder;cfg.compile_model=False;cfg.vlm_model_name=snapshot
        torch.cuda.reset_peak_memory_stats();t=time.perf_counter()
        self.policy=SmolVLAPolicy.from_pretrained(str(folder),config=cfg,local_files_only=True,strict=True)
        self.pre,self.post=make_pre_post_processors(policy_cfg=cfg,pretrained_path=str(folder),preprocessor_overrides={
            'device_processor':{'device':'cuda'},'rename_observations_processor':{'rename_map':RENAME},
            'tokenizer_processor':{'tokenizer_name':snapshot}})
        self.stats=load_file(str(folder/'policy_preprocessor_step_5_normalizer_processor.safetensors'))
        self.state_mean=self.stats[next(k for k in self.stats if 'observation.state' in k and k.endswith('mean'))].numpy().reshape(6)
        self.state_std=self.stats[next(k for k in self.stats if 'observation.state' in k and k.endswith('std'))].numpy().reshape(6)
        stats=load_file(str(folder/'policy_postprocessor_step_0_unnormalizer_processor.safetensors'))
        mean=stats[next(k for k in stats if 'action' in k and k.endswith('mean'))].numpy().reshape(6)
        std=stats[next(k for k in stats if 'action' in k and k.endswith('std'))].numpy().reshape(6)
        with torch.inference_mode():probe=self.post(torch.stack([torch.zeros(6),torch.ones(6)]).unsqueeze(0).cuda()).cpu().numpy()[0]
        np.testing.assert_allclose(probe[0],mean,rtol=1e-5,atol=1e-5)
        np.testing.assert_allclose(probe[1],mean+std,rtol=1e-5,atol=1e-5)
        torch.cuda.synchronize()
        report['model']={'checkpoint':'010000','precision':args.precision,'base_revision':REV,
                         'path':str(folder),'load_seconds':time.perf_counter()-t,'gpu':torch.cuda.get_device_name(0),
                         'torch':torch.__version__,'strict_load':True,'unnormalizer_probe':True}
        self.report=report
    def predict(self,row,seed):
        import torch
        import numpy as np
        from lerobot.policies.utils import prepare_observation_for_inference
        self.policy.reset();self.pre.reset();self.post.reset()
        torch.manual_seed(seed);torch.cuda.manual_seed_all(seed)
        torch.cuda.synchronize();t0=time.perf_counter()
        obs={'observation.state':row['state'].copy(),**{f'observation.images.{k}':v.copy() for k,v in row['images'].items()}}
        with torch.inference_mode(),torch.autocast('cuda',dtype=self.dtype):
            batch=self.pre(prepare_observation_for_inference(obs,torch.device('cuda'),TASK,'so101_follower'))
            if {k for k in batch if k.startswith('observation.images.')}!=set(RENAME.values()):raise ValueError('Unexpected camera mapping')
            torch.cuda.synchronize();t1=time.perf_counter()
            chunk=self.policy.predict_action_chunk(batch)
            torch.cuda.synchronize();t2=time.perf_counter()
            action=self.post(chunk).float().cpu().numpy()
            torch.cuda.synchronize();t3=time.perf_counter()
        if action.shape!=(1,50,6) or not np.isfinite(action).all():raise ValueError('Invalid predicted action chunk')
        return action[0],{'pre_ms':(t1-t0)*1000,'chunk_ms':(t2-t1)*1000,'post_ms':(t3-t2)*1000,'pipeline_ms':(t3-t0)*1000,
                'input_age_at_result_ms':(t3-min(row['state_timestamp'],*row['image_timestamps'].values()))*1000,
                'state_zscore':((row['state']-self.state_mean)/(self.state_std+1e-8)).tolist()}
    def memory(self):
        import torch
        return {'peak_allocated_mib':torch.cuda.max_memory_allocated()/2**20,'peak_reserved_mib':torch.cuda.max_memory_reserved()/2**20}


def percentiles(values):
    if not values:return None
    import numpy as np
    return {'count':len(values),'p50':float(np.percentile(values,50)),'p95':float(np.percentile(values,95)),
            'max':float(max(values)),'mean':float(np.mean(values))}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--mode',choices=['inputs','live','replay'],default='inputs')
    p.add_argument('--lerobot-root',type=Path,default=Path('G:/LeRobot'))
    p.add_argument('--offline-kit',type=Path)
    p.add_argument('--model-dir',type=Path)
    p.add_argument('--robot-config',type=Path)
    p.add_argument('--calibration',type=Path)
    p.add_argument('--port')
    p.add_argument('--precision',choices=['bf16','fp16'],default='bf16')
    p.add_argument('--seconds',type=float,default=60)
    p.add_argument('--max-chunks',type=int,default=0)
    p.add_argument('--no-browser',action='store_true')
    p.add_argument('--gpu-memory-fraction',type=float,default=0)
    p.add_argument('--output',type=Path,default=HERE/'results'/datetime.now().strftime('%Y%m%d_%H%M%S_%f'))
    args=p.parse_args()
    if not 0<args.seconds<=600 or args.max_chunks<0:p.error('seconds must be within (0,600], max-chunks >=0')
    args.offline_kit=args.offline_kit or args.lerobot_root/'offline_kit'
    args.robot_config=args.robot_config or args.lerobot_root/'data/robots/lerobot.json'
    args.calibration=args.calibration or args.lerobot_root/'data/calibration/robots/so_follower/lerobot.json'
    args.output=args.output.resolve();args.output.mkdir(parents=True,exist_ok=False)
    os.environ['HF_HUB_OFFLINE']='1';os.environ['TRANSFORMERS_OFFLINE']='1'
    os.environ.setdefault('TOKENIZERS_PARALLELISM','false')
    report={'status':'started','mode':args.mode,'created_utc':datetime.now(timezone.utc).isoformat(),
            'python':sys.version,'executable':sys.executable,'arguments':{k:str(v) for k,v in vars(args).items()},
            'source_sha256':{f.name:digest(f) for f in HERE.glob('*.py')},
            'motor_action_execution':False,'power_start':power_status(),
            'power_scheme':run_command(['powercfg','/getactivescheme']) if os.name=='nt' else None,
            'limits':'Read-only predictions, no action execution. Camera timestamps are software arrival times, not exposure times. Replay input ages are artificial.'}
    preview=None;source=None;engine=None;telemetry=Telemetry(args.output);records=[];predictions=[];states=[];cleanup=[]
    try:
        report['packages']={}
        for name in ('torch','lerobot','opencv-python-headless','feetech-servo-sdk','pyserial'):
            try:report['packages'][name]=importlib.metadata.version(name)
            except importlib.metadata.PackageNotFoundError:report['packages'][name]='not installed'
        telemetry.start()
        if args.mode!='inputs':
            engine=Engine(args,report)
            warm=Replay(args)
            for i in range(5):engine.predict(warm.snapshot(),20260919+i)
            report['model']['warmup_calls']=5
        source=Replay(args) if args.mode=='replay' else Hardware(args,report)
        source.start()
        preview=Preview(args.output,not args.no_browser)
        start=time.perf_counter();mid_saved=False;last=None
        with (args.output/'iterations.jsonl').open('w',encoding='utf-8') as log:
            while time.perf_counter()-start<args.seconds:
                tick=time.perf_counter();row=source.snapshot();index=len(records)
                record={'index':index,'wall_time_ns':time.time_ns(),'monotonic_start':tick,'elapsed_s':tick-start,
                    'state':row['state'].tolist(),'camera_age_ms':row['camera_age_ms'],'state_age_ms':row['state_age_ms'],
                    'pair_timestamp_skew_ms':row['pair_timestamp_skew_ms'],'sample_file':row.get('sample_file'),
                    'image_timestamps':row['image_timestamps'],'state_timestamp':row['state_timestamp']}
                if engine:
                    seed=20260919+(index%33 if args.mode=='replay' else index)
                    pred,timing=engine.predict(row,seed)
                    record.update(timing);record['seed']=seed;record['first_predicted_action']=pred[0].tolist()
                    record['first_action_minus_state']=(pred[0]-row['state']).tolist()
                    predictions.append(pred);states.append(row['state'])
                save='first_preview.jpg' if index==0 else None
                if not mid_saved and tick-start>=args.seconds/2:save='middle_preview.jpg';mid_saved=True
                # Preview is deliberately outside model/pipeline timers; complete loop time is also recorded.
                preview.update(row,record,save)
                record['loop_work_ms']=(time.perf_counter()-tick)*1000
                record['power']=power_status()
                log.write(json.dumps(record,allow_nan=False)+'\n');log.flush();records.append(record);last=row
                if index%10==0:print(f'{args.mode}: {index+1} observations, {time.perf_counter()-start:.1f}s',flush=True)
                if args.max_chunks and len(records)>=args.max_chunks:break
                if args.mode=='inputs':time.sleep(max(0,.1-(time.perf_counter()-tick)))
        if last:preview.update(last,records[-1],'last_preview.jpg')
        report['elapsed_capture_seconds']=time.perf_counter()-start
        report['status']='passed' if records else 'failed_no_observations'
    except KeyboardInterrupt:report['status']='interrupted_by_user'
    except Exception:report['status']='failed';report['error']=traceback.format_exc();print(report['error'],flush=True)
    finally:
        for obj in (source,preview,telemetry):
            if obj is not None:
                try:obj.close()
                except Exception:cleanup.append(traceback.format_exc())
        if engine:report['gpu_memory']=engine.memory()
        report['telemetry']=telemetry.info;report['power_end']=power_status();report['observations']=len(records)
        if predictions:
            import numpy as np
            np.savez_compressed(args.output/'predictions.npz',actions=np.stack(predictions),states=np.stack(states))
        if records:
            keys=('chunk_ms','pipeline_ms','loop_work_ms','state_age_ms','pair_timestamp_skew_ms','input_age_at_result_ms')
            report['timing_summary']={k:percentiles([r[k] for r in records if k in r]) for k in keys}
            report['camera_age_ms']={c:percentiles([r['camera_age_ms'][c] for r in records]) for c in ('1','3')}
        if cleanup:report['cleanup_errors']=cleanup
        if report.get('cleanup_errors') and report['status']=='passed':report['status']='cleanup_failed'
        report['nvidia_smi_after']=run_command(['nvidia-smi','--query-gpu=memory.used,memory.free','--format=csv'])
        write_json(args.output/'summary.json',report)
        (args.output/'SUMMARY.txt').write_text(f"Status: {report['status']}\nMode: {args.mode}\nObservations: {len(records)}\nMotor actions executed: no\nSee summary.json and iterations.jsonl.\n",encoding='utf-8')
        archive=args.output.with_name(args.output.name+'_return.zip')
        with zipfile.ZipFile(archive,'w',zipfile.ZIP_DEFLATED) as z:
            for f in sorted(args.output.rglob('*')):
                if f.is_file():z.write(f,f.relative_to(args.output))
        print(f"STATUS: {report['status']}\nSEND BACK: {archive}",flush=True)
    return 0 if report['status'] in ('passed','interrupted_by_user') else 1

if __name__=='__main__':sys.exit(main())
