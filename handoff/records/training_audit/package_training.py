from pathlib import Path
import tarfile, json
base=Path(r'G:\LeRobot\data\ChomCUI')
dest=Path(r'D:\桌面\萌芽杯\training_audit\so101_training_inputs.tar')
manifest=[]
with tarfile.open(dest,'w') as tar:
 src=Path(r'G:\LeRobot\source\lerobot-0.6.0')
 for f in src.rglob('*'):
  if f.is_file() and not any(x in f.parts for x in ['.git','__pycache__','.venv']):tar.add(f,arcname='source/lerobot-0.6.0/'+f.relative_to(src).as_posix())
 for p in sorted(base.iterdir()):
  if not p.is_dir() or '20260918' not in p.name or not (p/'meta/info.json').exists():continue
  info=json.loads((p/'meta/info.json').read_text())
  manifest.append({'repo_id':'ChomCUI/'+p.name,'episodes':info['total_episodes'],'frames':info['total_frames']})
  for f in p.rglob('*'):
   if f.is_file() and '.cache' not in f.parts:tar.add(f,arcname='data/ChomCUI/'+p.name+'/'+f.relative_to(p).as_posix())
Path(r'D:\桌面\萌芽杯\training_audit\training_sources.json').write_text(json.dumps(manifest,indent=2),encoding='utf8')
print('Packaged',sum(x['episodes'] for x in manifest),'episodes',dest.stat().st_size,'bytes',flush=True)
