import json
from pathlib import Path
import numpy as np
import pyarrow.parquet as pq
from safetensors.numpy import load_file

root=Path(r'G:\LeRobot\data\ChomCUI')
report={'datasets':[],'note':'Raw position reconstructed from saved degree normalization; this is not a torque measurement.'}
all_a=[];all_s=[];episode_spans=[]
for folder in sorted(root.glob('*_20260918_*')):
    arrays=[]
    for file in sorted((folder/'data').rglob('*.parquet')):
        t=pq.read_table(file,columns=['action','observation.state','episode_index'])
        a=np.array(t['action'].to_pylist());s=np.array(t['observation.state'].to_pylist());e=np.array(t['episode_index'].to_pylist())
        all_a.append(a);all_s.append(s);arrays.append(s[:,4])
        for ep in np.unique(e):
            values=s[e==ep,4]
            episode_spans.append({'dataset':folder.name,'episode':int(ep),'wrist_roll_span_degrees':float(np.ptp(values))})
    v=np.concatenate(arrays)
    report['datasets'].append({'name':folder.name,'frames':len(v),'wrist_roll_state_min_deg':float(v.min()),'wrist_roll_state_max_deg':float(v.max())})
def stats(v):
    return {'min':float(v.min()),'max':float(v.max()),'median':float(np.median(v)),'std':float(np.std(v)),'p01':float(np.percentile(v,1)),'p99':float(np.percentile(v,99))}
a=np.concatenate(all_a);s=np.concatenate(all_s)
report['frames']=len(s)
for i,n in [(3,'wrist_flex'),(4,'wrist_roll')]:
    report[n]={'state_degrees':stats(s[:,i]),'action_degrees':stats(a[:,i])}
report['wrist_roll']['reconstructed_state_raw_ticks']=stats(np.rint(s[:,4]*4095/360+2046.5))
report['wrist_roll']['episode_state_span_degrees']=stats(np.array([x['wrist_roll_span_degrees'] for x in episode_spans]))
normalizer=load_file(r'G:\LeRobot\offline_kit\models\010000\pretrained_model\policy_preprocessor_step_5_normalizer_processor.safetensors')
report['state_normalizer']={k:v.tolist() for k,v in normalizer.items() if 'observation.state' in k and (k.endswith('mean') or k.endswith('std'))}
report['episodes']=episode_spans
out=Path(__file__).with_name('wrist_usage_audit.json')
out.write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
print(json.dumps({k:v for k,v in report.items() if k not in ('episodes','datasets')},ensure_ascii=False,indent=2));print(out)
