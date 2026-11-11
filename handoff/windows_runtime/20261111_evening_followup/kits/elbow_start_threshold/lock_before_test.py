"""Explicitly authorized one-time Lock=1 write; no motion commands."""
import json
import traceback
import zipfile
from datetime import datetime
from pathlib import Path
from control_core import BoundedArm, MOTORS, sync_packet
from readonly_bus import validate_calibration
from threshold import CAL_PATH

HERE=Path(__file__).resolve().parent
def main():
    out=HERE/'results'/datetime.now().strftime('%Y%m%d_%H%M%S_%f_LockRestore')
    out.mkdir(parents=True)
    report={'status':'failed','write_frames_hex':[]}
    arm=None
    try:
        cal=validate_calibration(CAL_PATH,HERE/'reference_calibration.json')
        arm=BoundedArm('COM5',cal,True,freeze_wrist=True)
        original=arm.guard.write_port
        def trace(packet):
            b=bytes(packet)
            if len(b)>4 and b[4] not in (1,2,0x82): report['write_frames_hex'].append(b.hex())
            return original(packet)
        arm.guard.write_port=trace
        arm.connect()
        arm.verify('Torque_Enable',[0]*6)
        arm.verify('P_Coefficient',[16]*6)
        pos=arm.raw_read(); report['before_position']=pos
        if min(pos[2]-cal['elbow_flex']['range_min'],cal['elbow_flex']['range_max']-pos[2])<48:
            raise RuntimeError('Elbow outside test starting range')
        report['lock_before']=arm.bus.sync_read('Lock',normalize=False,num_retry=0)
        # Whitelist only this exact single sync write: ID 1..6, address 55, one byte, value 1.
        if arm.guard.expected is not None: raise RuntimeError('Write gate busy')
        arm.guard.expected=sync_packet(55,1,[1]*6)
        try:
            arm.bus.sync_write('Lock',{n:1 for n in MOTORS},normalize=False,num_retry=0)
            if arm.guard.expected is not None: raise RuntimeError('Lock frame was not transmitted')
        finally:
            arm.guard.expected=None
        arm.verify('Lock',[1]*6)
        arm.verify('Torque_Enable',[0]*6)
        arm.verify('P_Coefficient',[16]*6)
        report['lock_after']=arm.bus.sync_read('Lock',normalize=False,num_retry=0)
        report['status']='completed'
    except BaseException:
        report['error']=traceback.format_exc()
    finally:
        if arm is not None:
            try: arm.close()
            except BaseException:
                report['status']='close_not_confirmed';report['close_error']=traceback.format_exc()
            report['bus']=arm.report()
        (out/'summary.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
        archive=out.with_name(out.name+'_return.zip')
        with zipfile.ZipFile(archive,'w',zipfile.ZIP_DEFLATED) as z:z.write(out/'summary.json','summary.json')
        print(json.dumps(report,ensure_ascii=False,indent=2));print('SEND BACK:',archive)
    return 0 if report['status']=='completed' else 1

if __name__=='__main__': raise SystemExit(main())
