"""One current-position write, raw serial audit, then verified torque OFF.

No model actions, PID, EEPROM, calibration or explicit torque-ON writes.
"""
import argparse
import json
import time
import traceback
from pathlib import Path
from urllib.request import urlopen
from control_core import BoundedArm, MOTORS
from readonly_bus import validate_calibration

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--run',action='store_true');args=parser.parse_args()
    if not args.run: raise SystemExit('Explicit --run required; current-position write may enable torque.')
    status=json.load(urlopen('http://localhost:8000/so101-readonly/api/status',timeout=3))
    if status.get('active'): raise RuntimeError('Readonly UI busy')
    cal=validate_calibration(Path(r'G:\LeRobot\data\calibration\robots\so_follower\lerobot.json'),Path(r'G:\LeRobot\motor_diagnostic_kit\reference_calibration.json'))
    arm=BoundedArm('COM5',cal,True)
    report={'serial':[],'status':'starting'}
    tx=arm.guard.write_port;rx=arm.bus.port_handler.readPort
    def traced_tx(packet):
        report['serial'].append({'t':time.perf_counter(),'direction':'TX','hex':bytes(packet).hex()})
        return tx(packet)
    def traced_rx(length):
        data=rx(length)
        if data: report['serial'].append({'t':time.perf_counter(),'direction':'RX','hex':bytes(data).hex()})
        return data
    arm.guard.write_port=traced_tx;arm.bus.port_handler.readPort=traced_rx
    try:
        arm.connect();arm.verify('Torque_Enable',[0]*6);arm.verify('Operating_Mode',[0]*6)
        initial=arm.raw_read();report['initial']=initial
        for n,v in zip(MOTORS,initial):
            if not cal[n]['range_min'] <= v <= cal[n]['range_max']: raise RuntimeError('Outside calibration')
        time.sleep(.05)
        if max(abs(a-b) for a,b in zip(initial,arm.raw_read()))>2: raise RuntimeError('Position unstable')
        report['goal_write_time']=time.perf_counter()
        arm.write_register('Goal_Position',initial)
        report['after_goal']={}
        for name in ('Torque_Enable','Goal_Position','Present_Position','Present_Current','Present_Load','Status'):
            report['after_goal'][name]=arm.bus.sync_read(name,normalize=False,num_retry=0)
        report['individual_torque']={n:arm.bus.read('Torque_Enable',n,normalize=False,num_retry=0) for n in MOTORS}
        report['status']='observed'
    except BaseException:
        report['status']='failed';report['error']=traceback.format_exc()
    finally:
        try: report['cleanup']=arm.cleanup_motion()
        except BaseException: report['cleanup_error']=traceback.format_exc()
        try: arm.close()
        except BaseException: report['close_error']=traceback.format_exc()
        report['bus']=arm.report()
        out=Path(__file__).with_name('trace_alignment_'+time.strftime('%Y%m%d_%H%M%S')+'.json')
        out.write_text(json.dumps(report,indent=2),encoding='utf-8')
        print(json.dumps({k:v for k,v in report.items() if k!='serial'},indent=2));print(out)

if __name__=='__main__': main()
