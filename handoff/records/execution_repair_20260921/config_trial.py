"""Explicit allowlist for saved calibration restoration and elbow P 16/20.
Never sends a position target or torque-enable command.
"""
import argparse
from contextlib import contextmanager
import dataclasses
import json
from pathlib import Path
import time
import traceback
from urllib.request import urlopen
from control_core import BoundedArm, MOTORS
from readonly_bus import validate_calibration

def packet_for(joint, name, value, cal):
    registers={'Homing_Offset':(31,2,'homing_offset'), 'Min_Position_Limit':(9,2,'range_min'), 'Max_Position_Limit':(11,2,'range_max')}
    if joint not in MOTORS or type(value) is not int: raise ValueError('Invalid joint/value')
    if name in registers:
        address,length,key=registers[name]
        if value!=cal[joint][key]: raise ValueError('Only archived calibration allowed')
        encoded=abs(value)|(0x0800 if value<0 else 0)
    elif name=='Lock' and value in (0,1): address,length,encoded=55,1,value
    elif name=='P_Coefficient' and joint=='elbow_flex' and value in (16,20): address,length,encoded=21,1,value
    else: raise ValueError('Configuration write not allowed')
    data=[255,255,MOTORS.index(joint)+1,length+3,3,address,*encoded.to_bytes(length,'little')]
    return bytes(data+[(~sum(data[2:]))&255])

def write_config(arm,joint,name,value,cal):
    packet=packet_for(joint,name,value,cal)
    if not arm.guard.writable or arm.guard.expected is not None: raise RuntimeError('Gate unavailable')
    arm.guard.expected=packet
    try:
        arm.bus.write(name,joint,value,normalize=False,num_retry=0)
        if arm.guard.expected is not None: raise RuntimeError('Expected packet not transmitted')
    finally: arm.guard.expected=None
    actual=arm.bus.read(name,joint,normalize=False,num_retry=0)
    if actual!=value: raise RuntimeError(f'{joint} {name}: expected {value}, got {actual}')

def main():
    p=argparse.ArgumentParser();p.add_argument('--mode',choices=['restore','p20','p16'],required=True);args=p.parse_args()
    for url,key in [('calibration-status','calibration_active'),('so101-readonly/api/status','active')]:
        if json.load(urlopen('http://localhost:8000/'+url,timeout=3)).get(key): raise RuntimeError('UI task active')
    cal=validate_calibration(Path(r'G:\LeRobot\data\calibration\robots\so_follower\lerobot.json'),Path(r'G:\LeRobot\motor_diagnostic_kit\reference_calibration.json'))
    arm=BoundedArm('COM5',cal,True);report={'mode':args.mode,'status':'starting'}
    out=Path(__file__).with_name('config_'+args.mode+'_'+time.strftime('%Y%m%d_%H%M%S')+'.json')
    def save(): out.write_text(json.dumps(report,indent=2),encoding='utf-8')
    try:
        arm.bus.connect(handshake=True);arm.connected=True
        arm.torque_before=arm.bus.sync_read('Torque_Enable',normalize=False)
        arm.verify('Torque_Enable',[0]*6)
        report['before_calibration']={n:dataclasses.asdict(v) for n,v in arm.bus.read_calibration().items()}
        report['before_p']=arm.bus.sync_read('P_Coefficient',normalize=False)
        report['before_lock']=arm.bus.sync_read('Lock',normalize=False)
        save()
        if args.mode=='restore':
            for n in MOTORS:
                write_config(arm,n,'Lock',0,cal)
                for register,key in [('Homing_Offset','homing_offset'),('Min_Position_Limit','range_min'),('Max_Position_Limit','range_max')]:
                    write_config(arm,n,register,cal[n][key],cal)
                write_config(arm,n,'Lock',1,cal)
            if not arm.bus.is_calibrated: raise RuntimeError('Restoration mismatch')
        else:
            if not arm.bus.is_calibrated: raise RuntimeError('Calibration mismatch; no gain change')
            if report['before_lock']['elbow_flex']!=1: raise RuntimeError('Expected locked EPROM')
            if report['before_p']['elbow_flex'] not in (16,20): raise RuntimeError('Unexpected P baseline')
            # Keep Lock=1; test RAM effect only. Never unlock to persist trial gain.
            write_config(arm,'elbow_flex','P_Coefficient',20 if args.mode=='p20' else 16,cal)
        report['after_calibration']={n:dataclasses.asdict(v) for n,v in arm.bus.read_calibration().items()}
        report['after_p']=arm.bus.sync_read('P_Coefficient',normalize=False)
        report['after_lock']=arm.bus.sync_read('Lock',normalize=False)
        arm.verify('Torque_Enable',[0]*6)
        report['position']=arm.raw_read();report['status']='completed'
    except BaseException: report['status']='failed';report['error']=traceback.format_exc()
    finally:
        try:arm.close()
        except BaseException:report['close_error']=traceback.format_exc()
        report['bus']=arm.report();save();print(json.dumps(report,indent=2));print(out)
    return 0 if report['status']=='completed' else 1

if __name__=='__main__': raise SystemExit(main())
