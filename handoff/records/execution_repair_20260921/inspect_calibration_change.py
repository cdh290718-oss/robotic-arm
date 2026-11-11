import dataclasses
import json
import sys
import time
import traceback
from pathlib import Path
from readonly_bus import ReadOnlyArm, validate_calibration

cal=validate_calibration(Path(r'G:\LeRobot\data\calibration\robots\so_follower\lerobot.json'),Path(r'G:\LeRobot\motor_diagnostic_kit\reference_calibration.json'))
arm=ReadOnlyArm('COM5',cal)
report={'expected':cal,'registers':{}}
try:
    # Diagnostic reads only: intentionally inspect mismatched calibration without
    # permitting motor commands. The read-only serial packet guard stays active.
    arm.bus.connect(handshake=True);arm.connected=True
    arm.torque_before=arm.bus.sync_read('Torque_Enable',normalize=False)
    report['device']={n:dataclasses.asdict(v) for n,v in arm.bus.read_calibration().items()}
    report['differences']={n:{k:{'expected':cal[n][k],'device':v} for k,v in row.items() if cal[n][k]!=v} for n,row in report['device'].items() if row!=cal[n]}
    for name in ('P_Coefficient','I_Coefficient','D_Coefficient','Torque_Limit','Max_Torque_Limit','Present_Position','Goal_Position','Torque_Enable','Lock','Status'):
        report['registers'][name]=arm.bus.sync_read(name,normalize=False,num_retry=0)
except BaseException:report['error']=traceback.format_exc()
finally:
    try:arm.close()
    except BaseException:report['close_error']=traceback.format_exc()
    report['bus']=arm.report()
    out=Path(__file__).with_name('calibration_change_'+time.strftime('%Y%m%d_%H%M%S')+'.json')
    out.write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps(report,indent=2));print(out)
