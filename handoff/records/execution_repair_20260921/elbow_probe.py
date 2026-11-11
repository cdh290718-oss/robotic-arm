"""Diagnostic only: all joints hold current pose, elbow decreases at most 10 ticks.
2 s nominal, 3 s deadline, <=2 degree measured/tracking bound, always release.
Does not call enable_aligned or any model/EEPROM/configuration write.
"""
import argparse
import json
import time
import traceback
from pathlib import Path
from urllib.request import urlopen
from control_core import BoundedArm, DEG_PER_TICK, MOTORS
from readonly_bus import validate_calibration

def command_for(initial,step):
    if type(step) is not int or not 0<=step<=10: raise ValueError('Invalid step')
    values=list(initial);values[2]-=step
    return values

def check(raw,initial,command,cal):
    for i,n in enumerate(MOTORS):
        if not cal[n]['range_min']-2<=raw[i]<=cal[n]['range_max']+2: raise RuntimeError('Calibration envelope')
        if i<5 and (abs(raw[i]-initial[i])*DEG_PER_TICK>2 or abs(raw[i]-command[i])*DEG_PER_TICK>2): raise RuntimeError('2 degree bound')
    if abs(raw[5]-initial[5])>15: raise RuntimeError('Gripper movement')

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--run',action='store_true');args=parser.parse_args()
    if not args.run: raise SystemExit('Explicit --run required')
    if json.load(urlopen('http://localhost:8000/so101-readonly/api/status',timeout=3)).get('active'): raise RuntimeError('UI busy')
    cal=validate_calibration(Path(r'G:\LeRobot\data\calibration\robots\so_follower\lerobot.json'),Path(r'G:\LeRobot\motor_diagnostic_kit\reference_calibration.json'))
    arm=BoundedArm('COM5',cal,True);report={'rows':[],'status':'starting','max_elbow_goal_ticks':10}
    try:
        arm.connect();arm.verify('Torque_Enable',[0]*6);arm.verify('Operating_Mode',[0]*6)
        initial=arm.raw_read();report['initial']=initial
        for n,v in zip(MOTORS,initial):
            if not cal[n]['range_min']<=v<=cal[n]['range_max']: raise RuntimeError('Initial outside calibration')
        if command_for(initial,10)[2]<cal['elbow_flex']['range_min']: raise RuntimeError('Elbow target outside calibration')
        time.sleep(.05)
        if max(abs(a-b) for a,b in zip(initial,arm.raw_read()))>2: raise RuntimeError('Unstable start')
        started=time.perf_counter();command=list(initial)
        arm.write_register('Goal_Position',command)
        time.sleep(.02)
        # Observe known goal-triggered activation; never send a torque-ON packet.
        arm.verify('Goal_Position',command);arm.verify('Torque_Enable',[1]*6)
        check(arm.raw_read(),initial,command,cal)
        for step in range(21):
            begin=time.perf_counter()
            if begin-started>3: raise TimeoutError('3 second deadline')
            import msvcrt
            if msvcrt.kbhit() and msvcrt.getwch() in ('\x1b',' '): raise KeyboardInterrupt('Stop key')
            raw=arm.raw_read();check(raw,initial,command,cal)
            command=command_for(initial,min(step,10))
            arm.write_register('Goal_Position',command)
            row={'elapsed':time.perf_counter()-started,'command':list(command),'registers':{}}
            report['rows'].append(row)
            for name in ('Present_Position','Goal_Position','Torque_Enable','Present_Current','Present_Load','Present_Voltage','Status'):
                row['registers'][name]=arm.bus.sync_read(name,normalize=False,num_retry=0)
            reg=row['registers']
            if [reg['Goal_Position'][n] for n in MOTORS]!=command: raise RuntimeError('Goal mismatch')
            if [reg['Torque_Enable'][n] for n in MOTORS]!=[1]*6: raise RuntimeError('Torque changed')
            if any(reg['Status'].values()): raise RuntimeError('Motor status error')
            check([reg['Present_Position'][n] for n in MOTORS],initial,command,cal)
            elapsed=time.perf_counter()-begin
            if elapsed>.15: raise TimeoutError('Iteration exceeds 150ms')
            time.sleep(max(0,.1-elapsed))
        report['status']='completed'
    except BaseException:
        report['status']='failed';report['error']=traceback.format_exc()
    finally:
        try: report['cleanup']=arm.cleanup_motion()
        except BaseException: report['cleanup_error']=traceback.format_exc()
        try: arm.close()
        except BaseException: report['close_error']=traceback.format_exc()
        report['bus']=arm.report()
        out=Path(__file__).with_name('elbow_probe_'+time.strftime('%Y%m%d_%H%M%S')+'.json')
        out.write_text(json.dumps(report,indent=2),encoding='utf-8')
        print(json.dumps({k:v for k,v in report.items() if k!='rows'},indent=2));print(out)

if __name__=='__main__': main()
