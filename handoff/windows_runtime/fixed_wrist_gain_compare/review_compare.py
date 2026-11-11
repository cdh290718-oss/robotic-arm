import json
import zipfile
from pathlib import Path

path=Path(__file__).parent/'results/20261111_180428_366659_FixedWrist_P16_P20'
r=json.loads((path/'summary.json').read_text(encoding='utf-8'))
trials=r['trials'];frames=[bytes.fromhex(x) for x in r['write_frames_hex']]
goals=[x for x in frames if x[4]==0x83 and x[5]==42]
out={'status':r['status'],'reference':r['reference_raw'],'before_p20':r['before_p20_raw'],
     'same_commands':[x['command'] for x in trials[0]['samples']]==[x['command'] for x in trials[1]['samples']],
     'goal_packets':len(goals),'all_goal_ids_1_2_3_4_6':all(list(x[7:-1:3])==[1,2,3,4,6] for x in goals),
     'config_write_frames':[x.hex() for x in frames if x[4]==3],
     'p16_restored':r.get('p16_restored'),'final_p':r['final_p'],'final_lock':r['final_lock'],
     'final_torque':r['bus']['torque_after'],'zip_crc_error':zipfile.ZipFile(str(path)+'_return.zip').testzip(),'trials':[]}
for trial in trials:
    rows=trial['samples']
    def values(register):return [x['registers'][register]['elbow_flex'] for x in rows]
    loads=values('Present_Load');currents=values('Present_Current');voltages=values('Present_Voltage')
    out['trials'].append({'gain':trial['gain'],'samples':len(rows),'start_raw':trial['start_raw'],'metrics':trial['metrics'],
        'elbow_actual_minmax':[min(x['raw'][2] for x in rows),max(x['raw'][2] for x in rows)],
        'wrist_raw_unique':sorted({x['raw'][4] for x in rows}),
        'wrist_torque_unique':sorted({x['registers']['Torque_Enable']['wrist_roll'] for x in rows}),
        'elbow_load_raw_minmax':[min(loads),max(loads)],'elbow_current_raw_minmax':[min(currents),max(currents)],
        'elbow_voltage_raw_minmax':[min(voltages),max(voltages)],
        'p_before':trial['p_before']['elbow_flex'],'p_after':trial['p_after']['elbow_flex']})
(path/'review.json').write_text(json.dumps(out,indent=2),encoding='utf-8')
print(json.dumps(out,indent=2))
