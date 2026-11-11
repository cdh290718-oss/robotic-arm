"""One baseline and one P20 small-angle probe, always restore P16."""
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback

HERE=Path(__file__).parent

def run(script,args,pattern):
    before=set(HERE.glob(pattern))
    proc=subprocess.run([sys.executable,'-B',str(HERE/script),*args],capture_output=True,text=True,
                        encoding='utf-8',env={**os.environ,'PYTHONUTF8':'1'},timeout=30)
    new=set(HERE.glob(pattern))-before
    if len(new)!=1: raise RuntimeError(f'Expected exactly one report: {proc.stdout} {proc.stderr}')
    path=new.pop();data=json.loads(path.read_text(encoding='utf-8'))
    if proc.returncode or data['status']!='completed': raise RuntimeError(f'{path}: {data}')
    return {'path':str(path),'report':data}

def main():
    result={};gain_may_change=False
    try:
        result['baseline_config']=run('config_trial.py',['--mode','p16'],'config_p16_*.json')
        result['p16']=run('elbow_probe.py',['--run'],'elbow_probe_*.json')
        gain_may_change=True
        result['gain_config']=run('config_trial.py',['--mode','p20'],'config_p20_*.json')
        result['p20']=run('elbow_probe.py',['--run'],'elbow_probe_*.json')
        result['status']='completed'
    except BaseException:
        result['status']='failed';result['error']=traceback.format_exc()
    finally:
        if gain_may_change:
            try:result['restore']=run('config_trial.py',['--mode','p16'],'config_p16_*.json')
            except BaseException:result['restore_error']=traceback.format_exc();result['status']='restore_not_confirmed'
        for key in ('p16','p20'):
            if key in result:
                report=result[key]['report'];last=report['rows'][-1]
                result[key]['metrics']={'initial':report['initial'][2], 'target':last['command'][2],
                    'actual':last['registers']['Present_Position']['elbow_flex'],
                    'final_error_ticks':last['registers']['Present_Position']['elbow_flex']-last['command'][2],
                    'peak_raw_load':max(r['registers']['Present_Load']['elbow_flex'] for r in report['rows']),
                    'peak_raw_current':max(r['registers']['Present_Current']['elbow_flex'] for r in report['rows'])}
        out=HERE/('gain_comparison_'+time.strftime('%Y%m%d_%H%M%S')+'.json')
        out.write_text(json.dumps(result,indent=2),encoding='utf-8')
        print(out);print(json.dumps({k:v for k,v in result.items() if k in ('status','error','restore_error')},indent=2))
        for key in ('p16','p20'):
            if key in result:print(key,result[key]['metrics'])

if __name__=='__main__':main()
