"""Offline-tested SO101 elbow P16/P20 comparison. Default mode is read-only."""
import argparse
from datetime import datetime
import hashlib
import json
from pathlib import Path
import signal
import sys
import time
import traceback
import zipfile
from control_core import BoundedArm, DEG_PER_TICK, MOTORS
from readonly_bus import validate_calibration

HERE=Path(__file__).resolve().parent
REGISTERS=('Present_Position','Goal_Position','Torque_Enable','Present_Current',
           'Present_Load','Present_Voltage','Present_Temperature','Status')

def gain_packet(value):
    if type(value) is not int or value not in (16,20): raise ValueError('Only elbow P16/P20 allowed')
    b=[255,255,3,4,3,21,value]
    return bytes(b+[(~sum(b[2:]))&255])

def set_gain(arm,value):
    packet=gain_packet(value)
    arm.verify('Torque_Enable',[0]*6)
    if arm.bus.read('Lock','elbow_flex',normalize=False,num_retry=0)!=1:
        raise RuntimeError('Elbow Lock must be 1; will not unlock EEPROM')
    if not arm.guard.writable or arm.guard.expected is not None: raise RuntimeError('Write gate closed')
    arm.guard.expected=packet
    try:
        arm.bus.write('P_Coefficient','elbow_flex',value,normalize=False,num_retry=0)
        if arm.guard.expected is not None: raise RuntimeError('P write not transmitted')
    finally: arm.guard.expected=None
    actual=arm.bus.read('P_Coefficient','elbow_flex',normalize=False,num_retry=0)
    if actual!=value: raise RuntimeError(f'P readback expected {value}, got {actual}; no unlocking attempted')

def preflight(arm,cal,ticks=10):
    arm.verify('Torque_Enable',[0]*6);arm.verify('Operating_Mode',[0]*6)
    arm.verify('P_Coefficient',[16]*6);arm.verify('I_Coefficient',[0]*6);arm.verify('D_Coefficient',[32]*6)
    arm.verify('Lock',[1]*6)
    raw=arm.raw_read()
    bad=[f'{n}: {v}, allowed {cal[n]["range_min"]}..{cal[n]["range_max"]}'
         for n,v in zip(MOTORS,raw) if not cal[n]['range_min']<=v<=cal[n]['range_max']]
    if bad: raise RuntimeError('Starting pose outside saved range: '+'; '.join(bad)+'. Do not recalibrate to bypass.')
    if raw[2]-ticks<cal['elbow_flex']['range_min']: raise RuntimeError('Elbow target outside saved range')
    return raw

def check(raw,reference,command,cal,ticks=10):
    for i,n in enumerate(MOTORS):
        if not cal[n]['range_min']-2<=raw[i]<=cal[n]['range_max']+2: raise RuntimeError(f'{n}: measured outside calibration')
        if i<5 and (abs(raw[i]-reference[i])*DEG_PER_TICK>max(2,(ticks+2)*DEG_PER_TICK) or abs(raw[i]-command[i])*DEG_PER_TICK>2):
            raise RuntimeError(f'{n}: measured excursion/tracking exceeds 2 degrees')
    if abs(raw[5]-reference[5])>15: raise RuntimeError('Gripper moved')

def trajectory(reference,ticks=10):
    if ticks not in (10,33): raise ValueError('Only 10/33 tick trajectories allowed')
    # Identical absolute targets in both trials: ramp out, dwell, ramp back, dwell.
    for offset in [*range(ticks+1),*([ticks]*10),*range(ticks-1,-1,-1),*([0]*5)]:
        command=list(reference);command[2]-=offset
        yield command

def probe(arm,cal,reference,trial,stop=lambda:False,clock=time.perf_counter,sleep=time.sleep,ticks=10):
    arm.verify('Torque_Enable',[0]*6)
    raw=arm.raw_read();trial['starting_raw']=raw
    if max(abs(a-b) for a,b in zip(raw,reference))>2:
        raise RuntimeError('Starting pose differs by more than 2 ticks; comparison aborted without moving back automatically')
    sleep(.05)
    if max(abs(a-b) for a,b in zip(arm.raw_read(),raw))>2: raise RuntimeError('Unstable starting pose')
    if stop(): raise KeyboardInterrupt('Stop requested')
    start=clock();command=list(reference)
    arm.write_register('Goal_Position',command)
    sleep(.02)
    # This device has demonstrated goal-triggered activation. Verify it explicitly.
    arm.verify('Goal_Position',command);arm.verify('Torque_Enable',[1]*6)
    trial['after_activation_raw']=arm.raw_read()
    check(trial['after_activation_raw'],reference,command,cal,ticks)
    for command_next in trajectory(reference,ticks):
        begin=clock()
        if stop(): raise KeyboardInterrupt('Stop requested')
        if begin-start>(10 if ticks==33 else 5): raise TimeoutError('Trial exceeded duration limit')
        row={'elapsed_s':begin-start,'previous_command':command.copy(),'command':command_next.copy(),'registers':{}}
        trial['rows'].append(row)
        raw=arm.raw_read();row['before_write_raw']=raw
        check(raw,reference,command,cal,ticks)
        if clock()-begin>.150: raise TimeoutError('Read delayed; no next goal written')
        arm.write_register('Goal_Position',command_next);command=command_next
        for name in REGISTERS:
            row['registers'][name]=arm.bus.sync_read(name,normalize=False,num_retry=0)
        r=row['registers']
        if [r['Goal_Position'][n] for n in MOTORS]!=command: raise RuntimeError('Goal readback mismatch')
        if [r['Torque_Enable'][n] for n in MOTORS]!=[1]*6: raise RuntimeError('Torque changed during trial')
        if any(r['Status'].values()): raise RuntimeError('Motor status alarm')
        check([r['Present_Position'][n] for n in MOTORS],reference,command,cal,ticks)
        row['work_ms']=(clock()-begin)*1000
        if row['work_ms']>150: raise TimeoutError('Cycle exceeded 150ms')
        sleep(max(0,.1-(clock()-begin)))
    arm.release()
    trial['release_confirmed']=True
    trial['status']='completed'
    # Compare end of outward dwell, before the return-to-start trajectory.
    dwell=trial['rows'][ticks+10]
    actual=dwell['registers']['Present_Position']['elbow_flex']
    trial['metrics']={'dwell_actual':actual,'dwell_target':reference[2]-ticks,
                      'dwell_error_ticks':actual-(reference[2]-ticks),
                      'dwell_error_degrees':(actual-(reference[2]-ticks))*DEG_PER_TICK}

def run_session(arm,cal,mode,report,stop=lambda:False,clock=time.perf_counter,sleep=time.sleep,ticks=10):
    gain_dirty=False
    try:
        arm.connect()
        report['reference']=preflight(arm,cal,ticks)
        report['preflight']='passed'
        if mode=='check':report['status']='checked';return
        for gain in (16,20):
            if stop(): raise KeyboardInterrupt('Stop requested')
            if gain==20:
                gain_dirty=True  # Set before TX, including partial writes.
                set_gain(arm,20)
            trial={'gain':gain,'rows':[],'status':'starting'};report['trials'].append(trial)
            probe(arm,cal,report['reference'],trial,stop,clock,sleep,ticks)
        report['status']='completed'
    except BaseException:
        report['status']='stopped' if stop() else 'failed';report['error']=traceback.format_exc()
    finally:
        # No files/model work before attempting torque OFF.
        if arm.may_be_enabled:
            try:arm.release();report['cleanup_release_confirmed']=True
            except BaseException:report['release_error']=traceback.format_exc();report['status']='cleanup_not_confirmed'
        if gain_dirty:
            try:set_gain(arm,16);report['p_restored_to_16']=True
            except BaseException:report['p_restore_error']=traceback.format_exc();report['status']='cleanup_not_confirmed'
        if arm.connected:
            try:
                report['final_p']=arm.bus.sync_read('P_Coefficient',normalize=False,num_retry=0)
                report['final_lock']=arm.bus.sync_read('Lock',normalize=False,num_retry=0)
            except BaseException:report['final_read_error']=traceback.format_exc();report['status']='cleanup_not_confirmed'
        try:arm.close()
        except BaseException:report['close_error']=traceback.format_exc();report['status']='cleanup_not_confirmed'
        report['bus']=arm.report()

def main():
    p=argparse.ArgumentParser();p.add_argument('--mode',choices=['check','compare','large'],default='check')
    p.add_argument('--root',type=Path,default=Path(r'G:\LeRobot'));p.add_argument('--port',default='COM5')
    args=p.parse_args()
    ticks=33 if args.mode=='large' else 10
    out=HERE/'results'/(datetime.now().strftime('%Y%m%d_%H%M%S_%f')+'_'+args.mode)
    out.mkdir(parents=True)
    report={'mode':args.mode,'status':'starting','trials':[],'port':args.port,'python':sys.version,'limits':{'elbow_goal_ticks':ticks,'tracking_degrees':2,'measured_excursion_degrees':max(2,(ticks+2)*DEG_PER_TICK),'cycle_ms':150,'trial_seconds':10 if ticks==33 else 5}}
    stopped=False
    def on_signal(*_):
        nonlocal stopped
        stopped=True
    def stop():
        nonlocal stopped
        import msvcrt
        if msvcrt.kbhit() and msvcrt.getwch() in ('\x1b',' '):stopped=True
        return stopped
    signal.signal(signal.SIGINT,on_signal);signal.signal(signal.SIGTERM,on_signal)
    try:
        cal=validate_calibration(args.root/'data/calibration/robots/so_follower/lerobot.json',HERE/'reference_calibration.json')
        arm=BoundedArm(args.port,cal,args.mode in ('compare','large'))
        run_session(arm,cal,args.mode,report,stop,ticks=ticks)
    except BaseException:report['status']='failed';report['error']=traceback.format_exc()
    report['source_sha256']={f.name:hashlib.sha256(f.read_bytes()).hexdigest() for f in HERE.glob('*.py')}
    (out/'summary.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    archive=out.with_name(out.name+'_return.zip')
    with zipfile.ZipFile(archive,'w',zipfile.ZIP_DEFLATED) as z:
        z.write(out/'summary.json','summary.json');z.write(HERE/'reference_calibration.json','reference_calibration.json')
        for f in HERE.glob('*.py'):z.write(f,'source/'+f.name)
    print('STATUS:',report['status'])
    if 'error' in report:print(report['error'])
    for trial in report['trials']:print('P=',trial['gain'],trial.get('metrics',trial['status']))
    print('SEND BACK:',archive)
    if report['status']=='cleanup_not_confirmed':print('CLEANUP NOT CONFIRMED: inspect summary; do not assume torque or P restoration succeeded.')
    return 0 if report['status'] in ('checked','completed') else 1

if __name__=='__main__':raise SystemExit(main())

