from pathlib import Path
import os
os.sched_setaffinity(0, sorted(os.sched_getaffinity(0))[:4])
import json
import numpy as np
import pyarrow.parquet as pq
import pyarrow as pa
pa.set_cpu_count(2)
pa.set_io_thread_count(2)
import av
from lerobot.datasets import LeRobotDataset, merge_datasets, modify_tasks

base=Path('/vepfs-mlp2/mlp-public/huyaoqing/so101_vla_20260918')
dst=base/'data/ChomCUI/yellow_cube_0918_v1'
if dst.exists():
    raise RuntimeError('Merged directory exists; inspect it before rerunning, do not overwrite.')
sources=[]; manifest=[]; next_episode=0
for p in sorted((base/'data/ChomCUI').iterdir()):
    if '20260918' not in p.name: continue
    info=json.loads((p/'meta/info.json').read_text())
    n=0; ids=set()
    for q in (p/'data').rglob('*.parquet'):
        d=pq.read_table(q).to_pandas();n+=len(d);ids.update(d.episode_index.tolist())
        for col in ['action','observation.state']:
            assert np.isfinite(np.stack(d[col].to_numpy())).all(),(p,col)
    assert n==info['total_frames'] and len(ids)==info['total_episodes'],p
    for v in (p/'videos').rglob('*.mp4'):
        with av.open(str(v)) as c:
            c.streams.video[0].codec_context.thread_count=2
            assert sum(1 for _ in c.decode(video=0))==n,v
    repo='ChomCUI/'+p.name
    sources.append(LeRobotDataset(repo,root=p,video_backend='pyav'))
    for ep in range(info['total_episodes']):
        manifest.append({'merged_episode':next_episode,'source_repo':repo,'source_episode':ep})
        next_episode+=1
    print('AUDIT_OK',repo,n,flush=True)
assert next_episode==67,next_episode
ds=merge_datasets(sources,output_repo_id='ChomCUI/yellow_cube_0918_v1',output_dir=dst,concatenate_videos=False,concatenate_data=False)
modify_tasks(ds,new_task='Pick up the yellow cube and place it into the mesh container.')
(base/'data/episode_provenance.json').write_text(json.dumps(manifest,indent=2))
print('DATA_READY',next_episode,str(dst),flush=True)
