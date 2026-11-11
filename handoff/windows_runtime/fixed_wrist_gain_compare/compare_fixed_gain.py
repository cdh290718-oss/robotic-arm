"""Identical small-angle P16/P20 elbow trials, wrist motor ID5 passive."""
import argparse
import json
import time
import traceback
import zipfile
from datetime import datetime
from pathlib import Path
from control_core import BoundedArm, DEG_PER_TICK, GraspLimiter, MOTORS
from readonly_bus import validate_calibration
from elbow_probe_fixed import inspect

HERE=Path(__file__).resolve().parent
OFF=[0]*6

def gain_packet(value):
    if type(value) is not int or value not in (16,20):raise ValueError('Only P16/P20 allowed')
    frame=[255,255,3,4,3,21,value]
    return bytes(frame+[(~sum(frame[2:]))&255])

class GainArm(BoundedArm):
    def __init__(self,*args,**kwargs):
        super().__init__(*args,freeze_wrist=True,**kwargs)
        self.expected_gain=16
    def verify(self,name,expected):
        if name=='P_Coefficient':
            if expected!=[16]*6:raise ValueError('Unexpected P verification')
            expected=[16,16,self.expected_gain,16,16,16]
        return super().verify(name,expected)

def set_gain(arm,value):
    packet=gain_packet(value)
    arm.verify('Torque_Enable',OFF)
    if int(arm.bus.read('Lock','elbow_flex',normalize=False,num_retry=0))!=1:
        raise RuntimeError('Elbow EEPROM must be locked; will not unlock')
    if not arm.guard.writable or arm.guard.expected is not None:raise RuntimeError('Serial write gate unavailable')
    arm.guard.expected=packet
    # A partial write or unanticipated device response leaves motor state uncertain.
    arm.may_be_enabled=True
    try:
        arm.bus.write('P_Coefficient','elbow_flex',value,normalize=False,num_retry=0)
        if arm.guard.expected is not None:raise RuntimeError('Allowed gain write did not reach serial boundary')
    finally:arm.guard.expected=None
    arm.verify('Torque_Enable',OFF)
    arm.may_be_enabled=False
    readback=int(arm.bus.read('P_Coefficient','elbow_flex',normalize=False,num_retry=0))
    if readback!=value:raise RuntimeError(f'P readback mismatch: expected {value}, got {readback}')
    arm.expected_gain=value

def trajectory(reference):
    for offset in [*range(11),*([10]*10),*range(9,-1,-1),*([0]*5)]:
        command=reference.copy();command[2]-=offset
        yield command

def run_trial(arm,cal,reference,gain,trial,clock=time.perf_counter,sleep=time.sleep):
    arm.verify('Torque_Enable',OFF)
    actual=arm.raw_read();trial['start_raw']=actual
    if max(abs(a-b) for a,b in zip(actual,reference))>2:
        raise RuntimeError('Starting pose differs by more than two ticks; comparison stopped')
    limiter=GraspLimiter(cal,reference,freeze_wrist=True)
    arm.verify('P_Coefficient',[16]*6)
    trial['p_before']=arm.bus.sync_read('P_Coefficient',normalize=False,num_retry=0)
    arm.enable_aligned(actual)
    previous=actual.copy();start=clock()
    try:
        for command in trajectory(reference):
            begin=clock()
            if begin-start>5:raise TimeoutError('Trial exceeded 5 seconds')
            inspect(arm,limiter,reference,previous)
            arm.write_register('Goal_Position',command)
            row={'elapsed_s':clock()-start,'command':command.copy()};trial['samples'].append(row)
            row.update(inspect(arm,limiter,reference,command))
            row['temperature']=arm.bus.sync_read('Present_Temperature',normalize=False,num_retry=0)
            previous=command.copy()
            elapsed=clock()-begin
            if elapsed>.15:raise TimeoutError('Trial cycle exceeded 150ms')
            sleep(max(0,.1-elapsed))
        end_dwell=trial['samples'][20]
        trial['metrics']={'target_raw':end_dwell['command'][2], 'measured_raw':end_dwell['raw'][2],
            'error_ticks':end_dwell['raw'][2]-end_dwell['command'][2],
            'error_degrees':(end_dwell['raw'][2]-end_dwell['command'][2])*DEG_PER_TICK}
        arm.verify('P_Coefficient',[16]*6)
        trial['p_after']=arm.bus.sync_read('P_Coefficient',normalize=False,num_retry=0)
        arm.verify('Lock',[1]*6)
        trial['status']='completed'
    finally:
        if arm.may_be_enabled:
            arm.release();trial['release_confirmed']=True

def run(arm,cal,report,clock=time.perf_counter,sleep=time.sleep):
    gain_touched=False
    try:
        arm.connect();arm.verify('Torque_Enable',OFF);arm.verify('P_Coefficient',[16]*6)
        arm.verify('I_Coefficient',[0]*6);arm.verify('D_Coefficient',[32]*6);arm.verify('Lock',[1]*6)
        reference=arm.raw_read();report['reference_raw']=reference
        outside=[n for n,v in zip(MOTORS,reference) if not cal[n]['range_min']<=v<=cal[n]['range_max']]
        if outside:raise RuntimeError('Initial position outside saved range: '+', '.join(outside))
        if reference[2]-10<cal['elbow_flex']['range_min']:raise RuntimeError('Elbow target outside saved range')
        sleep(.05)
        if max(abs(a-b) for a,b in zip(reference,arm.raw_read()))>2:raise RuntimeError('Unstable start')
        baseline={'gain':16,'status':'starting','samples':[]};report['trials'].append(baseline)
        run_trial(arm,cal,reference,16,baseline,clock,sleep)
        arm.verify('Torque_Enable',OFF)
        after=arm.raw_read();report['before_p20_raw']=after
        if max(abs(a-b) for a,b in zip(reference,after))>2:
            raise RuntimeError('Pose did not return within two ticks; P20 was not written')
        gain_touched=True
        set_gain(arm,20);report['p20_readback_confirmed']=True
        trial={'gain':20,'status':'starting','samples':[]};report['trials'].append(trial)
        run_trial(arm,cal,reference,20,trial,clock,sleep)
        report['status']='completed'
    except BaseException:
        report['status']='failed';report['error']=traceback.format_exc()
    finally:
        if gain_touched and arm.connected and not arm.may_be_enabled:
            try:
                arm.verify('Torque_Enable',OFF)
            except BaseException:
                # Even a failed read leaves the physical torque state uncertain.
                arm.may_be_enabled=True
                report['cleanup_torque_check_error']=traceback.format_exc()
        if arm.may_be_enabled:
            try:arm.release();report['cleanup_release_confirmed']=True
            except BaseException:report['release_error']=traceback.format_exc();report['status']='cleanup_not_confirmed'
        if gain_touched:
            try:
                arm.verify('Torque_Enable',OFF)
                set_gain(arm,16)
                report['p16_restored']=True
            except BaseException:
                report['p16_restore_error']=traceback.format_exc();report['status']='cleanup_not_confirmed'
        if arm.may_be_enabled:
            try:arm.release();report['final_release_confirmed']=True
            except BaseException:report['final_release_error']=traceback.format_exc();report['status']='cleanup_not_confirmed'
        if arm.connected:
            try:
                report['final_p']=arm.bus.sync_read('P_Coefficient',normalize=False,num_retry=0)
                report['final_lock']=arm.bus.sync_read('Lock',normalize=False,num_retry=0)
            except BaseException:report['final_read_error']=traceback.format_exc();report['status']='cleanup_not_confirmed'
        try:arm.close()
        except BaseException:report['close_error']=traceback.format_exc();report['status']='cleanup_not_confirmed'
        report['bus']=arm.report()

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--run',action='store_true');args=parser.parse_args()
    if not args.run:raise SystemExit('Explicit --run required')
    out=HERE/'results'/datetime.now().strftime('%Y%m%d_%H%M%S_%f_FixedWrist_P16_P20');out.mkdir(parents=True)
    report={'status':'starting','trials':[],'limits':{'elbow_goal_ticks':10,'tracking_degrees':2,'cycle_ms':150,'trial_seconds':5},
            'gain_write_policy':'elbow ID3 P16/P20 only; Lock stays 1; P16 restored'}
    try:
        cal=validate_calibration(Path(r'G:\LeRobot\data\calibration\robots\so_follower\lerobot.json'),HERE/'reference_calibration.json')
        arm=GainArm('COM5',cal,True);sent=[];old=arm.guard.write_port
        def trace(packet):
            data=bytes(packet)
            if len(data)>4 and data[4] not in (1,2,0x82):sent.append(data.hex())
            return old(packet)
        arm.guard.write_port=trace
        run(arm,cal,report);report['write_frames_hex']=sent
    except BaseException:report['status']='failed';report['error']=traceback.format_exc()
    (out/'summary.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    archive=out.with_name(out.name+'_return.zip')
    with zipfile.ZipFile(archive,'w',zipfile.ZIP_DEFLATED) as z:z.write(out/'summary.json','summary.json')
    print(json.dumps({k:v for k,v in report.items() if k not in ('trials','write_frames_hex')},ensure_ascii=False,indent=2))
    for trial in report['trials']:print('P',trial['gain'],trial.get('metrics',trial['status']))
    print('SEND BACK:',archive)
    if report['status']!='completed':raise SystemExit(1)

if __name__=='__main__':main()
