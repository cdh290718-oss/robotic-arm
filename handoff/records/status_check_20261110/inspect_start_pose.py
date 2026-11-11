import json
import sys
import traceback
from datetime import datetime
from pathlib import Path
sys.path.insert(0,r'G:\LeRobot\policy_trial_kit')
from readonly_bus import ReadOnlyArm, MOTORS, validate_calibration

def main():
    report={'checked_at':datetime.now().isoformat(),'motor_writes':0}
    arm=None
    try:
        cal=validate_calibration(Path(r'G:\LeRobot\data\calibration\robots\so_follower\lerobot.json'),Path(r'G:\LeRobot\policy_trial_kit\reference_calibration.json'))
        arm=ReadOnlyArm('COM5',cal);arm.connect()
        raw=arm.bus.sync_read('Present_Position',normalize=False,num_retry=0)
        report['positions']={n:{'raw':raw[n],'min':cal[n]['range_min'],'max':cal[n]['range_max'],
            'inside':cal[n]['range_min']<=raw[n]<=cal[n]['range_max'],
            'nearest_boundary_delta_ticks':max(cal[n]['range_min'],min(cal[n]['range_max'],raw[n]))-raw[n]} for n in MOTORS}
        report['status']='read_completed'
    except BaseException:report['status']='failed';report['error']=traceback.format_exc()
    finally:
        if arm:
            try:arm.close()
            except BaseException:report['close_error']=traceback.format_exc()
            report['bus']=arm.report()
        output=Path(__file__).with_name('start_pose_'+datetime.now().strftime('%Y%m%d_%H%M%S')+'.json')
        output.write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
        print(json.dumps(report,ensure_ascii=False,indent=2));print(output)

if __name__=='__main__':main()
