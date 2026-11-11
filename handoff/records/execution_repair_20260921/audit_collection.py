import json
from pathlib import Path
import numpy as np
import pyarrow.parquet as pq

root=Path(r'G:\LeRobot\data\ChomCUI')
report={'datasets':[],'note':'Previous-frame action compared with next observation within episode; not simultaneous hardware servo-error telemetry.'}
errors=[];steps=[]
for folder in sorted(root.glob('*_20260918_*')):
    info=json.loads((folder/'meta/info.json').read_text(encoding='utf-8'))
    entry={'path':str(folder),'features':list(info['features']),'frames_read':0}
    for file in sorted((folder/'data').rglob('*.parquet')):
        table=pq.read_table(file,columns=['action','observation.state','episode_index','frame_index'])
        a=np.array(table['action'].to_pylist());s=np.array(table['observation.state'].to_pylist())
        e=np.array(table['episode_index'].to_pylist());f=np.array(table['frame_index'].to_pylist())
        valid=(e[1:]==e[:-1])&(f[1:]==f[:-1]+1)
        errors.extend(np.abs(a[:-1,2]-s[1:,2])[valid].tolist())
        steps.extend(np.abs(np.diff(a[:,2]))[valid].tolist());entry['frames_read']+=len(a)
    report['datasets'].append(entry)
for name,values in [('previous_target_vs_next_state_elbow_deg',errors),('target_step_elbow_deg',steps)]:
    v=np.array(values)
    report[name]={'count':len(v),'p50':float(np.percentile(v,50)),'p95':float(np.percentile(v,95)),'p99':float(np.percentile(v,99)),'max':float(v.max()),'fraction_above_4':float(np.mean(v>4))}
out=Path(__file__).with_name('collection_control_audit.json')
out.write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
print(json.dumps(report,ensure_ascii=False,indent=2))
