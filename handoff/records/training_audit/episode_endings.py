from pathlib import Path
import av
import pyarrow.parquet as pq
from PIL import Image,ImageDraw
root=Path(r'G:\LeRobot\data\ChomCUI');tiles=[]
for p in sorted(root.iterdir()):
 if '20260918' not in p.name or not (p/'meta/info.json').exists():continue
 d=pq.read_table(next((p/'meta/episodes').rglob('*.parquet'))).to_pandas()
 wanted={round(float(row['videos/observation.images.3/to_timestamp'])*30)-1:int(row.episode_index) for _,row in d.iterrows()}
 with av.open(str(next((p/'videos/observation.images.3').rglob('*.mp4')))) as c:
  for i,f in enumerate(c.decode(video=0)):
   if i not in wanted:continue
   im=f.to_image();im.thumbnail((240,180));tile=Image.new('RGB',(250,205),'white');tile.paste(im,(5,22));ImageDraw.Draw(tile).text((5,3),p.name.split('_')[0]+' episode '+str(wanted[i]),fill='black');tiles.append(tile)
for offset in range(0,len(tiles),24):
 part=tiles[offset:offset+24];sheet=Image.new('RGB',(1000,205*((len(part)+3)//4)),'#ddd')
 for i,t in enumerate(part):sheet.paste(t,((i%4)*250,(i//4)*205))
 sheet.save(Path(r'D:\桌面\萌芽杯\training_audit')/f'endings_{offset//24}.jpg')
print('End frames:',len(tiles))
