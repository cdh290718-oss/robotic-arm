"""Ten-second, gripper-fixed SO101 trial. Default DryRun never writes motors."""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import json
import math
import multiprocessing as mp
import os
from pathlib import Path
import queue
import signal
import sys
import time
import traceback
import zipfile
from control_core import (MOTORS, HZ, MAX_LOOP_GAP, MAX_INPUT_AGE, MAX_RESULT_AGE,
                          NO_PLAN_TIMEOUT, GRASP_TRACKING_ERROR_DEG, BoundedArm, Limiter, GraspLimiter, action_at)
from readonly_bus import validate_calibration
from live_probe import Engine, Replay, Telemetry, digest, write_json, percentiles

HERE = Path(__file__).resolve().parent

class Finished(Exception):
    pass


def choose_plan(active, pending, active_ticks, now, min_ticks, action_hz, max_age, rejected):
    """Select a plan at the actual control tick, rejecting stale waiting results."""
    event=None
    if pending is not None and now-pending['observation_time'] > max_age:
        rejected.append({'id':pending['id'], 'age_s':now-pending['observation_time'],
                         'reason':'expired_while_pending'})
        pending=None
    if pending is not None and (active is None or active_ticks >= min_ticks):
        event={'id':pending['id'], 'age_s':now-pending['observation_time'],
               'reason':'first' if active is None else 'minimum_ticks'}
        active,pending,active_ticks=pending,None,0
    if active is None:
        return active,pending,active_ticks,None,-1,event
    desired,index=action_at(active,now,action_hz)
    if desired is None and pending is not None:
        event={'id':pending['id'], 'age_s':now-pending['observation_time'], 'reason':'active_expired'}
        active,pending,active_ticks=pending,None,0
        desired,index=action_at(active,now,action_hz)
    if event is not None:
        event['first_index']=index
    return active,pending,active_ticks,desired,index,event


def validate_fixed_wrist_distribution(chunk):
    """Reject missing, nonfinite, or strongly out-of-distribution wrist inputs."""
    import math
    values = chunk.get('timings',{}).get('state_zscore',[])
    try:
        value = float(values[4]) if len(values)==6 else float('nan')
    except (TypeError, ValueError):
        value = float('nan')
    if not math.isfinite(value) or abs(value)>5:
        raise RuntimeError(f'Fixed wrist model input is outside 5 training standard deviations: {values}; no goal write')


class SharedObservation:
    """One latest observation, no image serialization in the control loop."""
    def __init__(self, ctx):
        self.images = ctx.RawArray('B', 2*480*640*3)
        self.meta = ctx.RawArray('d', 10)  # state[6], state_t, image_t[2], sequence
        self.lock = ctx.Lock()

    def put(self, row, sequence):
        import numpy as np
        if not self.lock.acquire(False):
            return False
        try:
            view = np.frombuffer(self.images, dtype=np.uint8).reshape(2, 480, 640, 3)
            for i, key in enumerate(('3', '1')):
                view[i] = row['images'][key]
            np.frombuffer(self.meta)[:] = [*row['state'], row['state_timestamp'],
                                          row['image_timestamps']['3'], row['image_timestamps']['1'], sequence]
            return True
        finally:
            self.lock.release()

    def get(self, last_sequence):
        import numpy as np
        if not self.lock.acquire(timeout=.02):
            return None
        try:
            meta = np.frombuffer(self.meta).copy()
            if meta[9] <= last_sequence:
                return None
            images = np.frombuffer(self.images, dtype=np.uint8).reshape(2,480,640,3).copy()
        finally:
            self.lock.release()
        return {'images': {'3': images[0], '1': images[1]}, 'state': meta[:6].astype(np.float32),
                'state_timestamp': float(meta[6]), 'image_timestamps': {'3': float(meta[7]), '1': float(meta[8])},
                'sequence': int(meta[9])}


def inference_worker(args, shared, results, stop):
    # Windows spawn starts fresh; this process never opens cameras or motor ports.
    import numpy as np
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    results.cancel_join_thread()  # never wait indefinitely for a stopped consumer
    report = {'status': 'loading', 'has_hardware_access': False}
    predictions = []
    inputs = []
    try:
        with (args.output/'inference.log').open('w', encoding='utf-8', buffering=1) as log:
            sys.stdout = log
            sys.stderr = log
            engine = Engine(args, report)
            row = None
            while not stop.is_set() and row is None:
                row = shared.get(0)
                time.sleep(.01)
            if row is None:
                return
            for i in range(5):
                if stop.is_set():
                    return
                engine.predict(row, 900+i)
            report['warmup_chunks'] = 5
            results.put({'kind': 'ready'}, timeout=1)
            previous = 0
            with (args.output/'chunks.jsonl').open('w', encoding='utf-8', buffering=1) as log_chunks:
                while not stop.is_set():
                    row = shared.get(previous)
                    if row is None:
                        time.sleep(.005)
                        continue
                    previous = row['sequence']
                    now = time.perf_counter()
                    if now-min(row['state_timestamp'], *row['image_timestamps'].values()) > MAX_INPUT_AGE:
                        continue
                    action, timings = engine.predict(row, 1000+len(predictions))
                    message = {'kind': 'chunk', 'id': len(predictions), 'sequence': previous,
                               'observation_time': min(row['state_timestamp'], *row['image_timestamps'].values()),
                               'finished_time': time.perf_counter(), 'actions': action.tolist(), 'timings': timings}
                    predictions.append(action)
                    inputs.append(row['state'])
                    log_chunks.write(json.dumps(message)+'\n')
                    try:
                        results.put_nowait(message)
                    except queue.Full:
                        report['dropped_output_messages'] = report.get('dropped_output_messages', 0)+1
            report.update(status='stopped', chunks=len(predictions), memory=engine.memory())
    except BaseException:
        report.update(status='failed', error=traceback.format_exc())
        try:
            results.put({'kind':'error', 'error':report['error']}, timeout=.2)
        except queue.Full:
            pass
    finally:
        if predictions:
            np.savez_compressed(args.output/'predictions.npz', actions=np.stack(predictions), states=np.stack(inputs))
        write_json(args.output/'inference_summary.json', report)


class CameraRig:
    def __init__(self, config, report):
        self.cameras = {}
        self.config = config
        self.report = report

    def start(self):
        from lerobot.cameras.opencv.configuration_opencv import OpenCVCameraConfig
        from lerobot.cameras.opencv.camera_opencv import OpenCVCamera
        selected = {str(c['name']): c for c in self.config['cameras'] if str(c['name']) in ('1','3')}
        if len([c for c in self.config['cameras'] if str(c['name']) in ('1','3')]) != 2 or set(selected) != {'1','3'}:
            raise ValueError('Expected exactly cameras 1 and 3')
        if selected['1']['camera_index'] == selected['3']['camera_index']:
            raise ValueError('Camera roles have identical device indices')
        self.report['camera_config'] = selected
        for name in ('3', '1'):
            c = selected[name]
            if (c.get('width'), c.get('height'), c.get('fps'), c.get('fourcc')) != (640,480,30,'MJPG'):
                raise ValueError('Camera settings differ from validated training layout')
            camera = OpenCVCamera(OpenCVCameraConfig(index_or_path=c['camera_index'], width=640,
                                  height=480, fps=30, fourcc='MJPG', color_mode='rgb', backend=c.get('backend',0)))
            self.cameras[name] = camera
            camera.connect()
            cap = camera.videocapture
            fourcc = int(cap.get(6))
            self.report.setdefault('opened_cameras',{})[name] = {
                'index':c['camera_index'], 'backend':cap.getBackendName(),
                'width':cap.get(3), 'height':cap.get(4), 'fps_reported':cap.get(5),
                'fourcc':''.join(chr((fourcc >> (8*i)) & 255) for i in range(4))}

    def row(self, state, stamp):
        import numpy as np
        images = {}
        timestamps = {}
        for name, camera in self.cameras.items():
            if camera.thread is None or not camera.thread.is_alive():
                raise RuntimeError(f'Camera {name} acquisition stopped')
            # Never wait indefinitely for the camera thread while motors are active.
            if not camera.frame_lock.acquire(timeout=.005):
                raise TimeoutError('Camera frame lock timeout')
            try:
                if camera.latest_frame is None or camera.latest_timestamp is None:
                    raise RuntimeError('Camera has no frame')
                images[name] = camera.latest_frame.copy()
                timestamps[name] = camera.latest_timestamp
            finally:
                camera.frame_lock.release()
            if images[name].shape != (480,640,3) or images[name].dtype != np.uint8:
                raise ValueError('Unexpected camera RGB array')
            if float(np.percentile(images[name][::8,::8].mean(axis=2),99)) < 18:
                raise RuntimeError(f'Camera {name} is nearly black; remove cover/check lighting')
        now = time.perf_counter()
        if now-min(stamp, *timestamps.values()) > MAX_INPUT_AGE:
            raise TimeoutError('Camera/state older than 250 ms')
        return {'images':images, 'state':np.asarray(state,dtype=np.float32), 'state_timestamp':stamp,
                'image_timestamps':timestamps}

    def close(self):
        errors = []
        for camera in self.cameras.values():
            try:
                if camera.is_connected:
                    camera.disconnect()
            except Exception:
                errors.append(traceback.format_exc())
        return errors


class SimArm:
    """Only for developer model/IPC tests. Does not represent real robot dynamics."""
    def __init__(self, freeze_wrist=False):
        self.raw = [1964,778,3274,2908,2042,2046]
        self.may_be_enabled = False
        self.writes = 0
        self.freeze_wrist = freeze_wrist
    def raw_read(self): return list(self.raw)
    def write_register(self, name, values):
        assert name == 'Goal_Position'
        self.raw = list(values)
        self.writes += 1
    def enable_aligned(self, raw, stop_requested=lambda:False, expected_d=32):
        self.may_be_enabled = True
        return self.raw_read()
    def hold(self): return {'simulated':True, 'goal_readback_confirmed':True, 'all_torque_enabled':True}
    def cleanup_motion(self): return {'action':'hold','confirmed':True,**self.hold()}
    def close(self): pass
    def report(self): return {'simulated':True, 'hardware_write_packets':0, 'simulated_commands':self.writes}


def save_panel(output, row, name):
    import cv2
    import numpy as np
    # Offline image file only; no preview encoding/HTTP server on the control path.
    panel = cv2.cvtColor(np.hstack([row['images']['3'],row['images']['1']]), cv2.COLOR_RGB2BGR)
    ok,encoded=cv2.imencode(Path(name).suffix,panel)
    if not ok:
        raise IOError('Failed to save camera panel')
    (output/name).write_bytes(encoded.tobytes())


def stop_key():
    if os.name == 'nt':
        import msvcrt
        while msvcrt.kbhit():
            if msvcrt.getwch() in ('\x1b', ' ', 'q', 'Q'):
                return True
    return False


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--mode', choices=['dryrun','execute','release','simulate','grasp','grasp_dryrun','grasp_simulate'], default='dryrun')
    p.add_argument('--seconds', type=float, default=10)
    p.add_argument('--precision', choices=['bf16','fp16'], default='bf16')
    p.add_argument('--lerobot-root', type=Path, default=Path('G:/LeRobot'))
    p.add_argument('--offline-kit', type=Path)
    p.add_argument('--model-dir', type=Path)
    p.add_argument('--robot-config', type=Path)
    p.add_argument('--calibration', type=Path)
    p.add_argument('--port', default='')
    p.add_argument('--gpu-memory-fraction', type=float, default=0)
    p.add_argument('--fixed-wrist', action='store_true', help='Keep wrist_roll passive; six-dimensional observation remains real')
    p.add_argument('--output', type=Path, default=HERE/'results'/datetime.now().strftime('%Y%m%d_%H%M%S_%f'))
    p.add_argument('--min-chunk-ticks', type=int, default=0, help='Keep each action chunk active for at least this many control ticks before replacing it (0 = current behavior).')
    p.add_argument('--action-hz', type=float, default=HZ, help='Action-chunk index rate; HZ = plan timeline, lower = slower playback.')
    p.add_argument('--max-result-age-s', type=float, default=MAX_RESULT_AGE, help='Max age of a model result when accepted as the active plan.')
    p.add_argument('--speed-ticks', type=int, default=8, help='Raw ticks per 30 Hz update for body targets: 8 = 8x baseline (~21.1 deg/s), 4 = 4x baseline (~10.5 deg/s).')
    p.add_argument('--expected-d', type=int, default=32, choices=[0, 32], help='Expected D_Coefficient on the five active motors at startup (32 default; 0 only for the D=0 trial).')
    args = p.parse_args()
    if args.min_chunk_ticks < 0:
        p.error('--min-chunk-ticks must be nonnegative')
    if not math.isfinite(args.action_hz) or not 0 < args.action_hz <= HZ:
        p.error('--action-hz must be finite and in (0, 30]')
    if not math.isfinite(args.max_result_age_s) or not 0 < args.max_result_age_s <= 4.0:
        p.error('--max-result-age-s must be finite and in (0, 4.0]; values above the default 1.4s are only for slow-playback (V2) experiments')
    if not 1 <= args.speed_ticks <= 8:
        p.error('--speed-ticks must be an integer in [1, 8]')
    GraspLimiter.speed_ticks = args.speed_ticks
    requested_mode=args.mode
    grasp=requested_mode in ('grasp','grasp_dryrun','grasp_simulate')
    limiter_type=GraspLimiter if grasp else Limiter
    args.mode={'grasp':'execute','grasp_dryrun':'dryrun','grasp_simulate':'simulate'}.get(args.mode,args.mode)
    maximum = (120 if requested_mode == 'grasp' else 60) if grasp else (120 if args.mode == 'dryrun' else 10)
    if not 0 < args.seconds <= maximum:
        p.error(f'Mode {args.mode} supports 0 < seconds <= {maximum}')
    args.offline_kit = args.offline_kit or args.lerobot_root/'offline_kit'
    args.robot_config = args.robot_config or args.lerobot_root/'data/robots/lerobot.json'
    args.calibration = args.calibration or args.lerobot_root/'data/calibration/robots/so_follower/lerobot.json'
    args.output = args.output.resolve()
    args.output.mkdir(parents=True, exist_ok=False)
    os.environ['HF_HUB_OFFLINE'] = '1'
    os.environ['TRANSFORMERS_OFFLINE'] = '1'
    report = {'status':'starting', 'mode':args.mode, 'created_utc':datetime.now(timezone.utc).isoformat(),
              'python':sys.version, 'executable':sys.executable,
              'arguments':{k:str(v) for k,v in vars(args).items()},
              'code_sha256':{f.name:digest(f) for f in HERE.glob('*.py')},
              'bounds':{'max_seconds':args.seconds, 'max_excursion_deg':8, 'setpoint_speed_upper_deg_s':5,
                        'max_raw_ticks_per_update':1, 'gripper_fixed':True, 'max_loop_gap_ms':150,
                        'max_input_age_ms':250, 'max_new_result_age_ms':1400, 'no_plan_stop_s':2},
              'limits':'Software bounded trial, not hard real-time or a collision-safe planner. Camera timestamps are decode times.'}
    report['requested_mode']=requested_mode
    report['grasp_success_verified']=False
    report['fixed_wrist']={'enabled':args.fixed_wrist,'passive_motor_id':5 if args.fixed_wrist else None,
                           'max_drift_ticks':11 if args.fixed_wrist else None,
                           'model_state':'measured position; never substituted',
                           'goal_write_ids':[1,2,3,4,6] if args.fixed_wrist else [1,2,3,4,5,6]}
    report['consumption_policy']={'min_chunk_ticks':args.min_chunk_ticks,'action_hz':args.action_hz,
                                  'max_result_age_s':args.max_result_age_s,'expected_d':args.expected_d,'speed_ticks':args.speed_ticks}
    if grasp:
        report['bounds'].update(max_excursion_deg=None,workspace='saved training calibration',gripper_fixed=False,
                                setpoint_speed_upper_deg_s=args.speed_ticks*360/4095*30,max_raw_ticks_per_update=args.speed_ticks,
                                speed_reference=f'{args.speed_ticks}x baseline goal rate: {args.speed_ticks} raw ticks per 30 Hz update (~{args.speed_ticks*360/4095*30:.1f} deg/s per active joint)',
                                grasp_tracking_stop_deg=GRASP_TRACKING_ERROR_DEG,
                                gripper_speed_percent_s=40,gripper_tracking_percent=5,
                                goal_tracking_preclamp='within 4deg body and 5 percentage points gripper of latest measured position')
    arm = rig = worker = results = telemetry = None
    ctx = mp.get_context('spawn')
    stop = ctx.Event()
    signal.signal(signal.SIGINT, lambda *_:stop.set())
    signal.signal(signal.SIGTERM, lambda *_:stop.set())
    rows = []
    active_chunk = None
    pending_chunk = None
    active_ticks = 0
    last_chunk_id = -1
    last_observation_time = -1.0
    rejected_chunks = []
    last_row = None
    exit_code = 1
    try:
        calibration = (json.loads((HERE/'reference_calibration.json').read_text()) if args.mode=='simulate'
                       else validate_calibration(args.calibration, HERE/'reference_calibration.json'))
        if any(c['drive_mode'] != 0 for c in calibration.values()):
            raise ValueError('This kit expects the validated zero-drive-mode calibration')
        if args.mode == 'simulate':
            arm = SimArm(freeze_wrist=args.fixed_wrist)
            replay = Replay(args)
            base_row = replay.snapshot()
            def snapshot(state, stamp):
                return {'images':base_row['images'], 'state':state, 'state_timestamp':stamp,
                        'image_timestamps':{'1':stamp,'3':stamp}}
        else:
            config = json.loads(args.robot_config.read_text(encoding='utf-8-sig'))
            report['robot_config_sha256'] = digest(args.robot_config)
            report['calibration_sha256'] = digest(args.calibration)
            report['port'] = args.port or config['follower_port']
            if args.fixed_wrist:
                arm = BoundedArm(report['port'], calibration, args.mode in ('execute','release'), freeze_wrist=True)
            else:
                arm = BoundedArm(report['port'], calibration, args.mode in ('execute','release'))
            arm.connect()
            if args.mode == 'release':
                arm.release()
                report.update(status='released', stop_reason='explicit_release')
                exit_code = 0
                raise Finished()
            if args.mode == 'execute' and any(int(v) != 0 for v in arm.torque_before.values()):
                raise RuntimeError('Torque is already enabled. Support the arm and run Release before another Execute.')
            rig = CameraRig(config, report)
            rig.start()
            snapshot = rig.row
        initial = arm.raw_read()
        report['initial_raw'] = initial
        report['initial_range_check'] = {
            n: {'raw':v,'min':calibration[n]['range_min'],'max':calibration[n]['range_max'],
                'inside':calibration[n]['range_min']<=v<=calibration[n]['range_max'],
                'passive_fixed_wrist':bool(args.fixed_wrist and n=='wrist_roll')}
            for n,v in zip(MOTORS,initial)
        }
        limiter = limiter_type(calibration, initial, freeze_wrist=True) if args.fixed_wrist else limiter_type(calibration, initial)
        report['initial_state'] = limiter.state(initial)
        report['raw_envelope'] = {'min':limiter.lo, 'max':limiter.hi}
        shared = SharedObservation(ctx)
        results = ctx.Queue(maxsize=4)
        telemetry = Telemetry(args.output)
        telemetry.start()
        worker = ctx.Process(target=inference_worker, args=(args,shared,results,stop), name='SmolVLA-inference')
        worker.start()
        print('Loading offline 10k model + five warmup chunks; torque remains unchanged...', flush=True)
        ready = False
        preflight_end = time.perf_counter()+180
        sequence = 0
        def receive_chunks():
            nonlocal ready, pending_chunk, last_chunk_id, last_observation_time
            # Bounded draining: no unbounded queue loop on the controller thread.
            for _ in range(4):
                try:
                    message = results.get_nowait()
                except queue.Empty:
                    break
                if message['kind'] == 'error':
                    raise RuntimeError('Inference process failed: '+message['error'])
                if message['kind'] == 'ready':
                    ready = True
                    continue
                if message['kind'] != 'chunk':
                    raise ValueError('Unknown inference message')
                now = time.perf_counter()
                if message['id'] <= last_chunk_id or message['observation_time'] <= last_observation_time:
                    raise ValueError('Nonmonotonic model result')
                last_chunk_id = message['id']
                last_observation_time = message['observation_time']
                action_at(message, now, args.action_hz)  # includes shape, finite and timestamp validation
                if now-message['observation_time'] <= args.max_result_age_s:
                    pending_chunk = message
                else:
                    rejected_chunks.append({'id':message['id'], 'age_s':now-message['observation_time']})
        while not stop.is_set():
            if stop_key():
                stop.set()
                break
            raw = arm.raw_read()
            if max(abs(a-b) for a,b in zip(raw,initial)) > 2:
                raise RuntimeError('Arm moved during model preflight; rerun from a stable supported position')
            stamp = time.perf_counter()
            last_row = snapshot(limiter.state(raw),stamp)
            sequence += 1
            shared.put(last_row,sequence)
            receive_chunks()
            if ready and (active_chunk is not None or pending_chunk is not None):
                if active_chunk is None and pending_chunk is not None:
                    active_chunk = pending_chunk
                    pending_chunk = None
                    active_ticks = 0
                break
            if not worker.is_alive():
                raise RuntimeError('Inference process exited during preflight; see inference.log')
            if time.perf_counter() > preflight_end:
                raise TimeoutError('No fresh model result within 180 seconds; torque not enabled')
            time.sleep(.033)
        if stop.is_set():
            report.update(status='stopped_before_enable', stop_reason='user_stop')
            exit_code = 0
            raise Finished()
        save_panel(args.output,last_row,'camera_before.jpg')
        # Save/encoding may be slow; require another fresh result after all setup.
        if time.perf_counter()-active_chunk['observation_time'] > args.max_result_age_s:
            raise TimeoutError('Initial plan aged during setup; no torque enabled')
        initial = arm.raw_read()
        limiter = limiter_type(calibration,initial,freeze_wrist=True) if args.fixed_wrist else limiter_type(calibration,initial)
        last_row = snapshot(limiter.state(initial),time.perf_counter())
        if stop_key():
            stop.set()
        if stop.is_set():
            report.update(status='stopped_before_enable',stop_reason='user_stop')
            exit_code = 0
            raise Finished()
        if args.mode in ('execute','simulate'):
            if args.fixed_wrist and args.mode == 'execute':
                validate_fixed_wrist_distribution(active_chunk)
            initial = arm.enable_aligned(initial, stop.is_set, args.expected_d)
            limiter = limiter_type(calibration,initial,freeze_wrist=True) if args.fixed_wrist else limiter_type(calibration,initial)
        report['execution_initial_raw'] = initial
        report['execution_envelope'] = {'min':limiter.lo, 'max':limiter.hi}
        start = previous = time.perf_counter()
        deadline = start+1/HZ
        no_plan_since = None
        print(f'{requested_mode.upper()} started: {args.seconds:g}s; gripper {"model-controlled" if grasp else "fixed"}. Ctrl+C / Esc / Space stops.', flush=True)
        reason = 'duration_complete'
        while time.perf_counter()-start < args.seconds:
            remaining = deadline-time.perf_counter()
            if remaining > 0:
                time.sleep(min(.005,remaining))
                if stop.is_set() or stop_key():
                    reason = 'user_stop'
                    break
                continue
            if stop.is_set() or stop_key():
                reason = 'user_stop'
                break
            tick_start = time.perf_counter()
            dt = tick_start-previous
            previous = tick_start
            if dt > MAX_LOOP_GAP:
                raise TimeoutError(f'Control loop gap {dt*1000:.1f} ms exceeds 150 ms')
            if not worker.is_alive():
                raise RuntimeError('Inference process exited during trial')
            raw = arm.raw_read()
            stamp = time.perf_counter()
            try:
                limiter.check_measured(raw, enforce_tracking=args.mode!='dryrun')
            except BaseException:
                report['failed_measurement'] = {
                    'time': stamp, 'raw_present': raw,
                    'previous_raw_command': list(limiter.last),
                    'error_degrees': [(raw[i]-limiter.last[i])*360/4095 for i in range(5)],
                }
                raise
            last_row = snapshot(limiter.state(raw),stamp)
            sequence += 1
            shared.put(last_row,sequence)
            receive_chunks()
            now = time.perf_counter()
            active_chunk,pending_chunk,active_ticks,desired,index,promotion = choose_plan(
                active_chunk,pending_chunk,active_ticks,now,args.min_chunk_ticks,
                args.action_hz,args.max_result_age_s,rejected_chunks)
            if desired is None:
                if no_plan_since is None:
                    no_plan_since = now
                if now-no_plan_since > NO_PLAN_TIMEOUT:
                    raise TimeoutError('No usable action plan for two seconds')
                command, clipped = list(limiter.last), [False]*6
            else:
                no_plan_since = None
                active_ticks += 1
                command, clipped = limiter.command(desired,dt)
            # Do not issue a new trajectory step if reads/copies stalled this iteration.
            if time.perf_counter()-tick_start > MAX_LOOP_GAP:
                raise TimeoutError('Control work exceeded 150 ms before write')
            if stop.is_set():
                reason = 'user_stop'
                break
            if args.mode in ('execute','simulate'):
                arm.write_register('Goal_Position',command)
            end = time.perf_counter()
            rows.append({'time':tick_start, 'elapsed_s':tick_start-start, 'interval_ms':dt*1000,
                         'work_ms':(end-tick_start)*1000, 'raw_present':raw, 'raw_command':command,
                         'state':limiter.state(raw), 'desired':None if desired is None else desired.tolist(),
                         'clipped':clipped, 'chunk_id':active_chunk['id'] if active_chunk is not None else -1,
                         'chunk_index':index,
                         'chunk_promotion':promotion,
                         'observation_age_ms':(now-active_chunk['observation_time'])*1000 if active_chunk is not None else -1,
                         'camera_age_ms':{k:(now-v)*1000 for k,v in last_row['image_timestamps'].items()},
                         'state_age_ms':(now-stamp)*1000, 'plan_expired_hold':desired is None})
            # Schedule from actual tick start; never catch up with a burst of writes.
            deadline = tick_start+1/HZ
        report.update(status='completed' if reason=='duration_complete' else 'stopped_by_user', stop_reason=reason,
                      active_seconds=time.perf_counter()-start)
        exit_code = 0
    except Finished:
        pass
    except BaseException:
        report.update(status='failed', error=traceback.format_exc(), stop_reason='exception')
        print(report['error'],file=sys.stderr,flush=True)
    finally:
        # Stop/hold BEFORE model joining, image saving, file IO or camera teardown.
        if arm is not None and arm.may_be_enabled:
            try:
                report['motion_cleanup'] = arm.cleanup_motion()
                report['stop_hold'] = (report['motion_cleanup']
                                       if report['motion_cleanup']['action'] == 'hold' else None)
                if not report['motion_cleanup'].get('confirmed'):
                    report['status'] = 'hold_not_confirmed'
                    exit_code = 1
            except BaseException:
                report['motion_cleanup'] = {'confirmed':False, 'error':traceback.format_exc()}
                report['status'] = 'cleanup_not_confirmed'
                exit_code = 1
                if args.fixed_wrist and getattr(arm,'force_release',False):
                    try:
                        arm.release()
                        report['motion_cleanup']['fallback_release_confirmed'] = True
                    except BaseException:
                        report['motion_cleanup']['fallback_release_error'] = traceback.format_exc()
                print('MOTION CLEANUP NOT CONFIRMED: inspect robot locally. Serial/software failure cannot guarantee a stop.',flush=True)
        stop.set()
        cleanup = []
        if arm:
            try:
                arm.close()
            except Exception:
                cleanup.append(traceback.format_exc())
            report['bus'] = arm.report()
        if rig:
            cleanup.extend(rig.close())
        if worker:
            worker.join(timeout=5)
            if worker.is_alive():
                worker.terminate()
                worker.join(timeout=2)
                report['worker_forced_termination'] = True
            report['worker_exitcode'] = worker.exitcode
        if telemetry:
            telemetry.close()
            report['telemetry'] = telemetry.info
        if last_row:
            try:
                save_panel(args.output,last_row,'camera_last_observation.jpg')
            except Exception:
                cleanup.append(traceback.format_exc())
        if results:
            results.close()
        report['control_ticks'] = len(rows)
        report['timing'] = {key:percentiles([r[key] for r in rows]) for key in ('interval_ms','work_ms','observation_age_ms')}
        report['rejected_chunks'] = rejected_chunks
        report['clipped_ticks_per_joint'] = [sum(r['clipped'][i] for r in rows) for i in range(6)]
        report['expired_plan_hold_ticks'] = sum(r['plan_expired_hold'] for r in rows)
        if cleanup:
            report['cleanup_errors'] = cleanup
            if exit_code == 0:
                report['status'] = 'cleanup_failed'
                exit_code = 1
        with (args.output/'control.jsonl').open('w',encoding='utf-8') as f:
            for row in rows:
                f.write(json.dumps(row,allow_nan=False)+'\n')
        write_json(args.output/'summary.json',report)
        archive = args.output.with_name(args.output.name+'_return.zip')
        with zipfile.ZipFile(archive,'w',zipfile.ZIP_DEFLATED) as z:
            for file in sorted(args.output.rglob('*')):
                if file.is_file():
                    z.write(file,file.relative_to(args.output))
        print(f"STATUS: {report['status']}\nSEND BACK: {archive}",flush=True)
    return exit_code


if __name__ == '__main__':
    mp.freeze_support()
    sys.exit(main())
