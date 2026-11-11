"""Read-only SO101 register capture. Never load a policy or command a motor."""
import argparse
from datetime import datetime,timezone
import hashlib
import importlib.metadata
import json
from pathlib import Path
import signal
import sys
import time
import traceback
import zipfile
from readonly_bus import ReadOnlyArm, validate_calibration
HERE=Path(__file__).resolve().parent
STATIC=('Operating_Mode','P_Coefficient','I_Coefficient','D_Coefficient','Max_Torque_Limit',
        'Torque_Limit','Acceleration','Goal_Time','Goal_Velocity','Protection_Current',
        'Protection_Time','Over_Current_Protection_Time','Min_Position_Limit','Max_Position_Limit')
DYNAMIC=('Torque_Enable','Present_Position','Goal_Position','Present_Velocity','Present_Load',
         'Present_Current','Present_Voltage','Present_Temperature','Status')


def read_registers(arm,names):
    data={}
    for name in names:
        begin=time.perf_counter()
        try:
            values=arm.bus.sync_read(name,normalize=False,num_retry=0)
            data[name]={'values':values,'read_ms':(time.perf_counter()-begin)*1000}
        except Exception:
            data[name]={'error':traceback.format_exc(),'read_ms':(time.perf_counter()-begin)*1000}
    return data


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--lerobot-root',type=Path,default=Path('G:/LeRobot'))
    p.add_argument('--port',default='')
    p.add_argument('--seconds',type=float,default=15)
    p.add_argument('--output',type=Path,default=HERE/'results'/datetime.now().strftime('%Y%m%d_%H%M%S_%f'))
    args=p.parse_args()
    if not 0<args.seconds<=60:p.error('seconds must be within (0,60]')
    args.output=args.output.resolve();args.output.mkdir(parents=True,exist_ok=False)
    stopping=False
    def stop(*_):
        nonlocal stopping
        stopping=True
    handlers={s:signal.getsignal(s) for s in (signal.SIGINT,signal.SIGTERM)}
    for s in handlers:signal.signal(s,stop)
    report={'status':'starting','created_utc':datetime.now(timezone.utc).isoformat(),
        'python':sys.version,'executable':sys.executable,
        'source_sha256':{f.name:hashlib.sha256(f.read_bytes()).hexdigest() for f in HERE.glob('*.py')},
        'note':'All register values are SDK raw values (sign decoding may apply). No inferred voltage/current units. No position, torque, calibration or PID writes.'}
    arm=None;rows=[];code=1
    try:
        config_path=args.lerobot_root/'data/robots/lerobot.json'
        calibration_path=args.lerobot_root/'data/calibration/robots/so_follower/lerobot.json'
        config=json.loads(config_path.read_text(encoding='utf-8-sig'))
        calibration=validate_calibration(calibration_path,HERE/'reference_calibration.json')
        report['port']=args.port or config['follower_port']
        report['calibration']=calibration
        report['packages']={}
        for name in ('lerobot','feetech-servo-sdk','pyserial'):
            try:report['packages'][name]=importlib.metadata.version(name)
            except importlib.metadata.PackageNotFoundError:report['packages'][name]='unknown'
        arm=ReadOnlyArm(report['port'],calibration)
        arm.connect()
        report['static_registers']=read_registers(arm,STATIC)
        print('READ ONLY: keep current supported setup; no movement commands. Ctrl+C stops.',flush=True)
        start=time.perf_counter()
        with (args.output/'registers.jsonl').open('w',encoding='utf-8',buffering=1) as f:
            while not stopping and time.perf_counter()-start<args.seconds:
                begin=time.perf_counter()
                row={'elapsed_seconds':begin-start,'wall_time_ns':time.time_ns(),'registers':read_registers(arm,DYNAMIC)}
                rows.append(row);f.write(json.dumps(row)+'\n')
                deadline=begin+.2
                while not stopping and time.perf_counter()<deadline:time.sleep(min(.02,max(0,deadline-time.perf_counter())))
        report['status']='interrupted' if stopping else 'completed';code=0
    except BaseException:
        report.update(status='failed',error=traceback.format_exc())
        print(report['error'],file=sys.stderr)
    finally:
        if arm is not None:
            try:arm.close()
            except Exception:report.update(status='cleanup_failed',cleanup_error=traceback.format_exc());code=1
            report['read_only_bus']=arm.report()
        report['samples']=len(rows)
        report['static_read_errors']=sum('error' in r for r in report.get('static_registers',{}).values())
        report['dynamic_read_errors']=sum('error' in reg for r in rows for reg in r['registers'].values())
        if (report['static_read_errors'] or report['dynamic_read_errors']) and report['status']=='completed':
            report['status']='completed_with_read_errors';code=1
        (args.output/'summary.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
        archive=args.output.with_name(args.output.name+'_return.zip')
        with zipfile.ZipFile(archive,'w',zipfile.ZIP_DEFLATED) as z:
            for file in args.output.iterdir():
                if file.is_file():z.write(file,file.name)
        for s,h in handlers.items():signal.signal(s,h)
        print(f"STATUS: {report['status']}\nSEND BACK: {archive}",flush=True)
    return code

if __name__=='__main__':sys.exit(main())
