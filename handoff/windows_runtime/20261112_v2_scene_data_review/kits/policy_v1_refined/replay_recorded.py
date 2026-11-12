"""Causal open-loop replay on recorded 8x chunk and control timestamps; no robot."""
import json
from pathlib import Path
from bounded_run import choose_plan

SOURCE=Path(r'D:\桌面\机械臂\policy_demo8x_trial\results\20261111_213457_906_Grasp_bf16')

def replay(rows,chunks,min_ticks):
    chunks=sorted(chunks,key=lambda x:x['finished_time'])
    active=pending=None;used=0;j=0;rejected=[];results=[];promotions=[]
    for row in rows:
        now=row['time']
        while j<len(chunks) and chunks[j]['finished_time']<=now:
            candidate=chunks[j];j+=1
            if now-candidate['observation_time']<=1.4:
                pending=candidate
        active,pending,used,desired,index,event=choose_plan(active,pending,used,now,min_ticks,30,1.4,rejected)
        if event:promotions.append(event)
        if desired is not None:used+=1
        results.append({'id':None if active is None else active['id'],'index':index,
                        'gripper':None if desired is None else float(desired[5])})
    elapsed=0;longest=0;current=0
    for i in range(len(rows)-1):
        dt=rows[i+1]['time']-rows[i]['time']
        if results[i]['gripper'] is not None and results[i]['gripper']<10:
            elapsed+=dt;current+=dt;longest=max(longest,current)
        else:current=0
    valid=[x for x in results if x['gripper'] is not None]
    return {'closed_goal_below10_s':round(elapsed,3),'longest_goal_below10_s':round(longest,3),
            'late_frame_fraction':sum(x['index']>=25 for x in valid)/len(valid),
            'valid_ticks':len(valid),'hold_ticks':len(results)-len(valid),
            'promotions':len(promotions),'expired_pending':len(rejected),
            'promoted_over_age_limit':sum(x['age_s']>1.4 for x in promotions),
            'baseline_matching_chunk_ids':sum(x['id']==r['chunk_id'] for x,r in zip(results,rows))}

def main():
    rows=[json.loads(x) for x in (SOURCE/'control.jsonl').open(encoding='utf8')]
    chunks=[json.loads(x) for x in (SOURCE/'chunks.jsonl').open(encoding='utf8')]
    report={'caveat':'Recorded chunks and timestamps held fixed; does not simulate changed observations, motor motion, or success.',
            'source':str(SOURCE),'control_ticks':len(rows),
            'baseline':replay(rows,chunks,0),'v1_25':replay(rows,chunks,25)}
    out=Path(__file__).resolve().parent/'offline_replay_v1.json'
    out.write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
    print(json.dumps(report,ensure_ascii=False,indent=2))

if __name__=='__main__':main()
