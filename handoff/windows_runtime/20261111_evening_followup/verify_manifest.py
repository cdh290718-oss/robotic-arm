"""Verify public evening handoff files and decoded JSONL hashes; no hardware access."""
import gzip
import hashlib
import json
from pathlib import Path

here = Path(__file__).resolve().parent
root = here.parents[2]
manifest = json.loads((here/'manifest.json').read_text(encoding='utf-8'))
errors = []

def digest(stream):
    h = hashlib.sha256()
    for block in iter(lambda: stream.read(1024*1024), b''):
        h.update(block)
    return h.hexdigest()

for entry in manifest['files']:
    p = root/entry['path']
    if not p.is_file():
        errors.append([entry['path'], 'missing'])
        continue
    with p.open('rb') as f:
        actual = digest(f)
    if actual != entry['sha256'] or p.stat().st_size != entry['bytes']:
        errors.append([entry['path'], 'stored size or SHA256 mismatch'])
    if entry['encoding'] == 'gzip':
        try:
            with gzip.open(p, 'rb') as f:
                decoded = digest(f)
            if decoded != entry['source_sha256']:
                errors.append([entry['path'], 'decoded SHA256 mismatch'])
        except (OSError, EOFError) as e:
            errors.append([entry['path'], str(e)])

print(json.dumps({'checked_files': len(manifest['files']), 'errors': errors}, ensure_ascii=False, indent=2))
raise SystemExit(1 if errors else 0)
