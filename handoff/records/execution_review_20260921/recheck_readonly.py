import json
import sys
import time
import traceback
from pathlib import Path
from urllib.request import urlopen

sys.path.insert(0, r'G:\LeRobot\execution_kit')
from readonly_bus import ReadOnlyArm, MOTORS, validate_calibration

def main():
    status=json.load(urlopen('http://localhost:8000/so101-readonly/api/status',timeout=3))
    if status.get('active'): raise RuntimeError('Readonly UI is busy')
    cal=validate_calibration(Path(r'G:\LeRobot\data\calibration\robots\so_follower\lerobot.json'),Path(r'G:\LeRobot\motor_diagnostic_kit\reference_calibration.json'))
    arm=ReadOnlyArm('COM5',cal)
    result={'time':time.strftime('%Y-%m-%d %H:%M:%S'),'samples':[]}
    try:
        arm.connect()
        for _ in range(3):
            sample={}
            result['samples'].append(sample)
            for name in ('Torque_Enable','Lock','Present_Position','Goal_Position','Present_Current','Present_Load','Status','Present_Voltage','Operating_Mode'):
                sample[name]=arm.bus.sync_read(name,normalize=False,num_retry=0)
            sample['individual_torque']={n:arm.bus.read('Torque_Enable',n,normalize=False,num_retry=0) for n in MOTORS}
            time.sleep(.1)
    except BaseException:
        result['error']=traceback.format_exc()
    finally:
        try: arm.close()
        except BaseException: result['close_error']=traceback.format_exc()
        result['bus']=arm.report()
        output=Path(__file__).with_name('recheck_readonly_'+time.strftime('%Y%m%d_%H%M%S')+'.json')
        output.write_text(json.dumps(result,indent=2),encoding='utf-8')
        print(json.dumps(result,indent=2));print(output)

if __name__=='__main__': main()
