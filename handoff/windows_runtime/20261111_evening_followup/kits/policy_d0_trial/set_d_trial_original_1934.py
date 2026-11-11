"""Set D on the five active motors, with exact packet whitelist and verification."""
import argparse
import json
import traceback
import zipfile
from datetime import datetime
from pathlib import Path
from control_core import BoundedArm, MOTORS, sync_packet
from readonly_bus import validate_calibration

HERE=Path(__file__).resolve().parent
IDS=[1,2,3,4,6]
NAMES=[MOTORS[i-1] for i in IDS]
CAL=Path(r'G:\LeRobot\data\calibration\robots\so_follower\lerobot.json')

def write_exact(arm,name,address,value):
    if (name,address,value) not in (('Lock',55,0),('Lock',55,1),('D_Coefficient',22,0),('D_Coefficient',22,32)):
        raise ValueError('Write outside D trial allowlist')
    if arm.guard.expected is not None: raise RuntimeError('Gate busy')
    arm.guard.expected=sync_packet(address,1,[value]*5,IDS)
    try:
        arm.bus.sync_write(name,{n:value for n in NAMES},normalize=False,num_retry=0)
        if arm.guard.expected is not None: raise RuntimeError('Expected frame not transmitted')
    finally: arm.guard.expected=None

def change(arm,value,report):
    if value not in (0,32): raise ValueError('Only 0 or 32 supported')
    arm.verify('Torque_Enable',[0]*6)
    arm.verify('P_Coefficient',[16]*6)
    arm.verify('Lock',[1]*6)
    before=arm.bus.sync_read('D_Coefficient',normalize=False,num_retry=0)
    report['d_before']=before
    expected_before=[32]*6 if value==0 else [0,0,0,0,32,0]
    if [int(before[n]) for n in MOTORS]!=expected_before:
        raise RuntimeError('Unexpected starting D values; no writes')
    needs_relock=False
    try:
        needs_relock=True # arm cleanup before transmission, including partial TX
        write_exact(arm,'Lock',55,0)
        arm.verify('Lock',[0,0,0,0,1,0])
        write_exact(arm,'D_Coefficient',22,value)
        arm.verify('D_Coefficient',[value,value,value,value,32,value])
    finally:
        if needs_relock:
            write_exact(arm,'Lock',55,1)
            arm.verify('Lock',[1]*6)
    arm.verify('Torque_Enable',[0]*6)
    arm.verify('P_Coefficient',[16]*6)
    report['d_after']=arm.bus.sync_read('D_Coefficient',normalize=False,num_retry=0)
    report['lock_after']=arm.bus.sync_read('Lock',normalize=False,num_retry=0)

def main():
    p=argparse.ArgumentParser();p.add_argument('--value',type=int,choices=[0,32],required=True);args=p.parse_args()
    out=HERE/'results'/datetime.now().strftime('%Y%m%d_%H%M%S_%f_SetD');out.mkdir(parents=True)
    report={'status':'failed','requested_d':args.value,'write_frames_hex':[]};arm=None
    try:
        cal=validate_calibration(CAL,HERE/'reference_calibration.json')
        arm=BoundedArm('COM5',cal,True,freeze_wrist=True)
        original=arm.guard.write_port
        def trace(packet):
            data=bytes(packet)
            if len(data)>4 and data[4] not in (1,2,0x82):report['write_frames_hex'].append(data.hex())
            return original(packet)
        arm.guard.write_port=trace
        arm.connect();change(arm,args.value,report);report['status']='completed'
    except BaseException:report['error']=traceback.format_exc()
    finally:
        if arm:
            try:arm.close()
            except BaseException:report['status']='close_not_confirmed';report['close_error']=traceback.format_exc()
            report['bus']=arm.report()
        (out/'summary.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
        archive=out.with_name(out.name+'_return.zip')
        with zipfile.ZipFile(archive,'w',zipfile.ZIP_DEFLATED) as z:z.write(out/'summary.json','summary.json')
        print(json.dumps(report,ensure_ascii=False,indent=2));print('SEND BACK:',archive)
    return 0 if report['status']=='completed' else 1
if __name__=='__main__':raise SystemExit(main())
