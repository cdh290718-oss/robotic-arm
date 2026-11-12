"""Verify public files and decoded JSONL hashes, without hardware access."""
import gzip
import hashlib
import json
from pathlib import Path

here=Path(__file__).resolve().parent
root=here.parents[2]
manifest=json.loads((here/'manifest.json').read_text(encoding='utf8'))
errors=[]
def digest(stream):
    h=hashlib.sha256()
    for block in iter(lambda:stream.read(1024*1024),b''):h.update(block)
    return h.hexdigest()
for e in manifest['files']:
    p=root/e['path']
    if not p.is_file():errors.append([e['path'],'missing']);continue
    with p.open('rb') as f:stored=digest(f)
    if stored!=e['sha256'] or p.stat().st_size!=e['bytes']:errors.append([e['path'],'stored mismatch'])
    if e['encoding']=='gzip':
        try:
            with gzip.open(p,'rb') as f:decoded=digest(f)
            if decoded!=e['source_sha256']:errors.append([e['path'],'decoded mismatch'])
        except (OSError,EOFError) as ex:errors.append([e['path'],str(ex)])
print(json.dumps({'checked_files':len(manifest['files']),'errors':errors},ensure_ascii=False,indent=2))
raise SystemExit(1 if errors else 0)
