"""One bounded elbow probe, no model/camera/configuration writes. Explicit --run required."""
import argparse
import json
import sys
import time
import traceback
from pathlib import Path

sys.path.insert(0, r'G:\LeRobot\execution_kit')
from control_core import BoundedArm, DEG_PER_TICK, MOTORS
from readonly_bus import validate_calibration

def target(initial, step):
    if not 0 <= step <= 10:
        raise ValueError('step outside 0..10')
    result = list(initial)
    result[2] -= step
    return result

def check(raw, initial, command):
    if any(abs(a-b)*DEG_PER_TICK > 2 for a,b in zip(raw[:5], initial[:5])):
        raise RuntimeError('Measured excursion exceeds 2 degrees')
    if any(abs(a-b)*DEG_PER_TICK > 2 for a,b in zip(raw[:5], command[:5])):
        raise RuntimeError('Tracking exceeds 2 degrees')
    if abs(raw[5]-initial[5]) > 15:
        raise RuntimeError('Gripper changed')

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--run',action='store_true');args=parser.parse_args()
    if not args.run:
        initial=[2000,800,3270,2700,2045,2046]
        assert target(initial,10)==[2000,800,3260,2700,2045,2046]
        check(target(initial,10),initial,target(initial,10))
        for bad in (-1,11):
            try: target(initial,bad)
            except ValueError: pass
            else: raise AssertionError('target boundary')
        try: check([2000,800,3200,2700,2045,2046],initial,initial)
        except RuntimeError: pass
        else: raise AssertionError('motion bound')
        print('Offline target/bound checks passed; no hardware accessed');return
    out=Path(__file__).parent / ('micro_probe_'+time.strftime('%Y%m%d_%H%M%S')+'.json')
    report={'status':'starting','rows':[],'limit_ticks':10,'note':'raw SDK register values; no unit inference'}
    arm=None;may_have_written=False
    def save():out.write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    def snap(phase,command):
        row={'phase':phase,'time':time.time(),'command':command,'registers':{}}
        report['rows'].append(row)
        try:
            for name in ('Present_Position','Goal_Position','Torque_Enable','Present_Current','Present_Load','Present_Voltage','Present_Temperature','Status'):
                row['registers'][name]=arm.bus.sync_read(name,normalize=False,num_retry=0)
        finally:save()
        return row
    try:
        cal=validate_calibration(Path(r'G:\LeRobot\data\calibration\robots\so_follower\lerobot.json'),Path(r'G:\LeRobot\motor_diagnostic_kit\reference_calibration.json'))
        arm=BoundedArm('COM5',cal,True);arm.connect()
        arm.verify('Torque_Enable',[0]*6)
        arm.verify('Operating_Mode',[0]*6)
        initial=arm.raw_read();report['initial']=initial
        for n,v in zip(MOTORS,initial):
            if not cal[n]['range_min'] <= v <= cal[n]['range_max']:raise RuntimeError('Initial out of calibration')
        if target(initial,10)[2] < cal['elbow_flex']['range_min']:raise RuntimeError('Target out of calibration')
        snap('before',initial)
        if max(abs(a-b) for a,b in zip(arm.raw_read(),initial))>2:raise RuntimeError('Unstable initial position')
        may_have_written=True
        arm.write_register('Goal_Position',initial)
        arm.verify('Goal_Position',initial)
        arm.verify('Torque_Enable',[0]*6)
        # Only the elbow is enabled. Other joints remain at their initial OFF state.
        torques=[0,0,1,0,0,0]
        arm.write_register('Torque_Enable',torques);arm.verify('Torque_Enable',torques)
        command=list(initial);start=time.perf_counter()
        for i in range(21):
            begin=time.perf_counter()
            if begin-start>3:raise RuntimeError('Probe time exceeded')
            raw=arm.raw_read();check(raw,initial,command)
            command=target(initial,min(i,10))
            arm.write_register('Goal_Position',command)
            row=snap('ramp' if i<=10 else 'dwell',command)
            if list(row['registers']['Goal_Position'].values()) != command:
                # Compare by explicit joint order, independent of dict insertion order.
                if [row['registers']['Goal_Position'][n] for n in MOTORS] != command:raise RuntimeError('Goal mismatch')
            if [row['registers']['Torque_Enable'][n] for n in MOTORS] != torques:raise RuntimeError('Torque changed')
            if any(row['registers']['Status'].values()):raise RuntimeError('Nonzero status')
            check([row['registers']['Present_Position'][n] for n in MOTORS],initial,command)
            elapsed=time.perf_counter()-begin
            if elapsed>.15:raise RuntimeError('Probe iteration exceeded 150ms')
            time.sleep(max(0,.1-elapsed))
        report['status']='completed'
    except BaseException:
        report['status']='failed';report['error']=traceback.format_exc()
    finally:
        if arm:
            if may_have_written:
                try:
                    arm.release();report['release_confirmed']=True
                except BaseException:report['release_error']=traceback.format_exc()
            try:arm.close()
            except BaseException:report['close_error']=traceback.format_exc()
            report['bus']=arm.report()
        save();print('RESULT',out);print(json.dumps(report,ensure_ascii=False))

if __name__=='__main__':main()
