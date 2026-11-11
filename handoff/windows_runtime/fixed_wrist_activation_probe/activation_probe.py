"""One aligned goal to IDs 1,2,3,4,6; never goal/torque-ON to ID 5."""
import argparse
import json
import time
import traceback
import zipfile
from datetime import datetime
from pathlib import Path
from control_core import BoundedArm, GraspLimiter, MOTORS
from readonly_bus import validate_calibration

HERE=Path(__file__).resolve().parent

def validate_sample(arm,limiter):
    raw=arm.raw_read()
    limiter.check_measured(raw,enforce_tracking=True)
    torque=arm.bus.sync_read('Torque_Enable',normalize=False,num_retry=0)
    expected=[1,1,1,1,0,1]
    if [int(torque[n]) for n in MOTORS]!=expected:
        arm.force_release=True
        raise RuntimeError(f'Unexpected torque during aligned hold: {torque}')
    status=arm.bus.sync_read('Status',normalize=False,num_retry=0)
    if any(int(v)!=0 for v in status.values()):raise RuntimeError(f'Motor status nonzero: {status}')
    return {'monotonic':time.perf_counter(),'raw':raw,'torque':torque,'status':status}

def run(arm,cal,report,sleep=time.sleep):
    try:
        arm.connect()
        arm.verify('Torque_Enable',[0]*6)
        initial=arm.raw_read();report['initial_raw']=initial
        # Check every joint against the saved calibration for this first test.
        outside=[n for n,v in zip(MOTORS,initial) if not cal[n]['range_min']<=v<=cal[n]['range_max']]
        if outside:raise RuntimeError('Initial position outside saved range: '+', '.join(outside))
        limiter=GraspLimiter(cal,initial,freeze_wrist=True)
        sleep(.05)
        if max(abs(a-b) for a,b in zip(initial,arm.raw_read()))>2:raise RuntimeError('Unstable start')
        arm.enable_aligned(initial)
        report['activation_completed']=True
        for _ in range(6):
            report['samples'].append(validate_sample(arm,limiter))
            sleep(.1)
        report['status']='completed'
    except BaseException:
        report['status']='failed';report['error']=traceback.format_exc()
    finally:
        if arm.may_be_enabled:
            try:arm.release();report['release_confirmed']=True
            except BaseException:report['release_error']=traceback.format_exc();report['status']='release_not_confirmed'
        try:arm.close()
        except BaseException:report['close_error']=traceback.format_exc();report['status']='close_not_confirmed'
        report['bus']=arm.report()

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--run',action='store_true');args=parser.parse_args()
    if not args.run:raise SystemExit('Explicit --run required')
    out=HERE/'results'/datetime.now().strftime('%Y%m%d_%H%M%S_%f_Activation')
    out.mkdir(parents=True)
    report={'status':'starting','samples':[],'goals':'current position only, motor IDs 1,2,3,4,6','planned_hold_s':.6}
    try:
        cal=validate_calibration(Path(r'G:\LeRobot\data\calibration\robots\so_follower\lerobot.json'),HERE/'reference_calibration.json')
        arm=BoundedArm('COM5',cal,True,freeze_wrist=True)
        sent=[];old=arm.guard.write_port
        def trace(packet):
            data=bytes(packet)
            if len(data)>4 and data[4] not in (1,2,0x82):sent.append(data.hex())
            return old(packet)
        arm.guard.write_port=trace
        run(arm,cal,report)
        report['write_frames_hex']=sent
    except BaseException:
        report['status']='failed';report['error']=traceback.format_exc()
    (out/'summary.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    archive=out.with_name(out.name+'_return.zip')
    with zipfile.ZipFile(archive,'w',zipfile.ZIP_DEFLATED) as z:z.write(out/'summary.json','summary.json')
    print(json.dumps(report,ensure_ascii=False,indent=2));print('SEND BACK:',archive)
    if report['status'] not in ('completed',):raise SystemExit(1)

if __name__=='__main__':main()
