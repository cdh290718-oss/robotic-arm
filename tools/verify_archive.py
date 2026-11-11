"""Verify the 2026-11-11 staged assets after `git lfs pull`; no hardware access."""
import hashlib
import json
import sys
from pathlib import Path

root=Path(__file__).resolve().parent.parent
manifest=json.loads((root/'handoff/manifest_20261111.json').read_text(encoding='utf-8'))
errors=[]
for entry in manifest['files']:
    path=root/entry['path']
    if not path.is_file():
        errors.append({'path':entry['path'],'reason':'missing'})
        continue
    if path.stat().st_size!=entry['bytes']:
        errors.append({'path':entry['path'],'reason':'size mismatch; check git lfs pull'})
        continue
    digest=hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda:stream.read(8*1024*1024),b''):digest.update(block)
    if digest.hexdigest()!=entry['sha256']:errors.append({'path':entry['path'],'reason':'SHA256 mismatch'})
print(json.dumps({'checked_files':len(manifest['files']),'errors':errors},ensure_ascii=False,indent=2))
raise SystemExit(1 if errors else 0)
