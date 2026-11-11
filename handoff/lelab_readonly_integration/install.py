"""Install additive LeLab read-only UI and keep exact pre-change backups."""
import hashlib
import json
from pathlib import Path
import shutil
from datetime import datetime

stage=Path(__file__).resolve().parent
root=Path(r'G:\LeRobot\source\lelab-9a182993a2a9a3b04312c13ff8365f68d379fe80')
backup=Path(r'G:\LeRobot\logs')/('readonly_integration_backup_'+datetime.now().strftime('%Y%m%d_%H%M%S'))
server=root/'lelab/server.py';index=root/'frontend/dist/index.html'
text=server.read_text(encoding='utf-8')
marker='# Serve the built frontend at /. Must be mounted last so API routes win.'
assert text.count(marker)==1
html=index.read_text(encoding='utf-8')
assert '</body>' in html
backup.mkdir(parents=True,exist_ok=False)
files={}
def keep(path):
    if path.exists():
        relative=path.relative_to(root);target=backup/relative;target.parent.mkdir(parents=True,exist_ok=True)
        shutil.copy2(path,target)
        files[str(relative)]={'before_sha256':hashlib.sha256(path.read_bytes()).hexdigest()}
    else:files[str(path.relative_to(root))]={'before_sha256':None}
for name in ['readonly_bridge.py','readonly_worker.py','readonly.html','readonly-link.js']:
    dest=root/'lelab'/name;keep(dest);shutil.copy2(stage/name,dest)
keep(server);keep(index)
if 'from .readonly_bridge import install as install_readonly' not in text:
    text=text.replace(marker,'# Local SO101 read-only integration (2026-09-21).\nfrom .readonly_bridge import install as install_readonly\ninstall_readonly(app)\n\n'+marker)
    server.write_text(text,encoding='utf-8')
if 'so101-readonly-link.js' not in html:
    html=html.replace('</body>','<script src="/so101-readonly-link.js" defer></script></body>')
    index.write_text(html,encoding='utf-8')
for relative,details in files.items():details['after_sha256']=hashlib.sha256((root/relative).read_bytes()).hexdigest()
record={'backup':str(backup),'root':str(root),'files':files}
(backup/'manifest.json').write_text(json.dumps(record,indent=2),encoding='utf-8')
(stage/'installation.json').write_text(json.dumps(record,indent=2),encoding='utf-8')
print(json.dumps(record,indent=2))
