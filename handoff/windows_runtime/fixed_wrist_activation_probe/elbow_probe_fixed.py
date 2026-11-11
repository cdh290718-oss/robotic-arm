"""One bounded P16 elbow diagnostic. Motor 5 is excluded from all goal writes."""
import json
import time
import traceback
import zipfile
from datetime import datetime
from pathlib import Path
from control_core import BoundedArm, DEG_PER_TICK, GraspLimiter, MOTORS
from readonly_bus import validate_calibration

HERE=Path(__file__).resolve().parent

def inspect(arm,limiter,initial,command):
    raw=arm.raw_read()
    limiter.check_measured(raw,enforce_tracking=False)
    for i in (0,1,2,3):
        if abs(raw[i]-initial[i])*DEG_PER_TICK>2 or abs(raw[i]-command[i])*DEG_PER_TICK>2:
            raise RuntimeError(f'{MOTORS[i]} exceeded 2 degree measured/tracking bound')
    if abs(raw[5]-initial[5])>15:raise RuntimeError('Gripper moved')
    registers={}
    for name in ('Torque_Enable','Present_Position','Goal_Position','Present_Current','Present_Load','Present_Voltage','Status'):
        registers[name]=arm.bus.sync_read(name,normalize=False,num_retry=0)
    if [int(registers['Torque_Enable'][n]) for n in MOTORS]!=[1,1,1,1,0,1]:
        arm.force_release=True;raise RuntimeError('Unexpected torque pattern')
    if any(int(v)!=0 for v in registers['Status'].values()):raise RuntimeError('Motor status nonzero')
    if any(int(registers['Goal_Position'][n])!=command[i] for i,n in enumerate(MOTORS) if i!=4):raise RuntimeError('Goal readback mismatch')
    return {'raw':raw,'command':command.copy(),'registers':registers,'monotonic':time.perf_counter()}

def run(arm,cal,report,sleep=time.sleep,clock=time.perf_counter):
    try:
        arm.connect();arm.verify('Torque_Enable',[0]*6)
        initial=arm.raw_read();report['initial_raw']=initial
        outside=[n for n,v in zip(MOTORS,initial) if not cal[n]['range_min']<=v<=cal[n]['range_max']]
        if outside:raise RuntimeError('Initial position outside saved range: '+', '.join(outside))
        if initial[2]-10<cal['elbow_flex']['range_min']:raise RuntimeError('Elbow target outside saved range')
        limiter=GraspLimiter(cal,initial,freeze_wrist=True)
        sleep(.05)
        if max(abs(a-b) for a,b in zip(initial,arm.raw_read()))>2:raise RuntimeError('Unstable start')
        arm.enable_aligned(initial)
        command=initial.copy();start=clock()
        for step in range(21):
            begin=clock()
            if begin-start>3:raise TimeoutError('Three-second diagnostic limit')
            inspect(arm,limiter,initial,command)
            command=initial.copy();command[2]-=min(step,10)
            arm.write_register('Goal_Position',command)
            row={'elapsed_s':clock()-start,'command':command.copy()};report['samples'].append(row)
            row.update(inspect(arm,limiter,initial,command))
            elapsed=clock()-begin
            if elapsed>.15:raise TimeoutError('Diagnostic cycle exceeded 150ms')
            sleep(max(0,.1-elapsed))
        report['status']='completed'
    except BaseException:report['status']='failed';report['error']=traceback.format_exc()
    finally:
        if arm.may_be_enabled:
            try:arm.release();report['release_confirmed']=True
            except BaseException:report['release_error']=traceback.format_exc();report['status']='release_not_confirmed'
        try:arm.close()
        except BaseException:report['close_error']=traceback.format_exc();report['status']='close_not_confirmed'
        report['bus']=arm.report()

def main():
    import argparse
    parser=argparse.ArgumentParser();parser.add_argument('--run',action='store_true');args=parser.parse_args()
    if not args.run:raise SystemExit('Explicit --run required')
    out=HERE/'results'/datetime.now().strftime('%Y%m%d_%H%M%S_%f_Elbow')
    out.mkdir(parents=True)
    report={'status':'starting','samples':[],'max_elbow_goal_ticks':10,'tracking_bound_degrees':2}
    try:
        cal=validate_calibration(Path(r'G:\LeRobot\data\calibration\robots\so_follower\lerobot.json'),HERE/'reference_calibration.json')
        arm=BoundedArm('COM5',cal,True,freeze_wrist=True)
        sent=[];old=arm.guard.write_port
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
    print(json.dumps({k:v for k,v in report.items() if k not in ('samples','write_frames_hex')},ensure_ascii=False,indent=2));print('SEND BACK:',archive)
    if report['status']!='completed':raise SystemExit(1)

if __name__=='__main__':main()
