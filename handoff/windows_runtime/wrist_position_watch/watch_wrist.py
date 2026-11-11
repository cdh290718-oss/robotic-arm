"""Display wrist_roll position without issuing any motor register writes."""
import argparse
import json
import math
import sys
import time
import traceback
import zipfile
from datetime import datetime
from pathlib import Path
from readonly_bus import MOTORS, ReadOnlyArm, validate_calibration

HERE=Path(__file__).resolve().parent

def assess(raw,torque,cal,mean,std):
    if set(raw)!=set(MOTORS) or set(torque)!=set(MOTORS): raise ValueError('Missing motor readings')
    if any(int(torque[n])!=0 for n in MOTORS): raise RuntimeError('Torque is not OFF on all motors; stop manual adjustment')
    if not math.isfinite(mean) or not math.isfinite(std) or std<=0: raise ValueError('Invalid model normalization statistics')
    v=int(raw['wrist_roll']);c=cal['wrist_roll']
    if not 0<=v<=4095: raise ValueError('Wrist reading outside valid 0..4095')
    degrees=(v-(c['range_min']+c['range_max'])/2)*360/4095
    return {'raw':v,'training_range':[c['range_min'],c['range_max']],
            'in_training_range':c['range_min']<=v<=c['range_max'],
            'model_state_degrees':degrees,'training_zscore':(degrees-mean)/std,
            'position_raw':{n:int(raw[n]) for n in MOTORS},'torque':{n:int(torque[n]) for n in MOTORS}}

def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,default=Path(r'G:\LeRobot'));p.add_argument('--port',default='COM5');p.add_argument('--seconds',type=int,default=120)
    args=p.parse_args()
    if not 1<=args.seconds<=300:p.error('seconds must be 1..300')
    output=HERE/'results'/datetime.now().strftime('%Y%m%d_%H%M%S_%f_WristWatch')
    output.mkdir(parents=True)
    report={'status':'starting','samples':[],'motor_writes':0};arm=None;exit_code=1
    try:
        from safetensors.numpy import load_file
        cal=validate_calibration(args.root/'data/calibration/robots/so_follower/lerobot.json',HERE/'reference_calibration.json')
        stats=load_file(str(args.root/'offline_kit/models/010000/pretrained_model/policy_preprocessor_step_5_normalizer_processor.safetensors'))
        mean=float(stats['observation.state.mean'][4]);std=float(stats['observation.state.std'][4])
        report['normalizer']={'wrist_roll_mean_degrees':mean,'wrist_roll_std_degrees':std}
        arm=ReadOnlyArm(args.port,cal);arm.connect()
        if any(int(v)!=0 for v in arm.torque_before.values()):raise RuntimeError('Torque is ON; no adjustment. Program does not disable it.')
        print('READ ONLY. No motor writes. Keep the terminal focused; Ctrl+C, Esc, or Q stops.',flush=True)
        print('Support the arm. Only move the wrist gently by hand when all torque readings are OFF.',flush=True)
        end=time.monotonic()+args.seconds
        while time.monotonic()<end:
            if sys.platform=='win32':
                import msvcrt
                if msvcrt.kbhit() and msvcrt.getwch() in ('\x1b','q','Q'):break
            torque=arm.bus.sync_read('Torque_Enable',normalize=False,num_retry=0)
            raw=arm.bus.sync_read('Present_Position',normalize=False,num_retry=0)
            sample=assess(raw,torque,cal,mean,std);sample['time']=datetime.now().isoformat(timespec='seconds')
            report['samples'].append(sample)
            print('wrist_roll raw={:4d} | training range {}..{} | z={:+.2f} | {}'.format(sample['raw'],*sample['training_range'],sample['training_zscore'],'IN RANGE' if sample['in_training_range'] else 'OUTSIDE RANGE'),flush=True)
            time.sleep(.5)
        report['status']='completed';exit_code=0
    except KeyboardInterrupt:
        report['status']='stopped';exit_code=0
    except BaseException:
        report['status']='failed';report['error']=traceback.format_exc();print(report['error'],file=sys.stderr,flush=True)
    finally:
        if arm:
            try:arm.close()
            except BaseException:report['close_error']=traceback.format_exc();report['status']='cleanup_failed';exit_code=1
            report['bus']=arm.report()
        (output/'summary.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
        archive=output.with_name(output.name+'_return.zip')
        with zipfile.ZipFile(archive,'w',zipfile.ZIP_DEFLATED) as z:z.write(output/'summary.json','summary.json')
        print('STATUS:',report['status']);print('RESULT:',archive)
    return exit_code

if __name__=='__main__':raise SystemExit(main())
