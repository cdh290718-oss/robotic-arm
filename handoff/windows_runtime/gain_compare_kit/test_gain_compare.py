import json
from pathlib import Path
import unittest
from control_core import BoundedArm,MOTORS,sync_packet,ExecutionGate
from gain_compare import run_session,trajectory,gain_packet

CAL=json.loads((Path(__file__).parent/'reference_calibration.json').read_text(encoding='utf-8'))
INITIAL=[2000,1000,3000,2700,2045,2100]

class Port:
    is_open=True
    is_using=False
    def __init__(self):self.sent=[]
    def writePort(self,p):self.sent.append(bytes(p));return len(p)
    def closePort(self):self.is_open=False
    def clearPort(self):pass

class Bus:
    is_calibrated=True
    def __init__(self,fault=None):
        self.port_handler=Port();self.fault=fault;self.fired=False;self.writes=[]
        self.data={'Present_Position':INITIAL.copy(),'Goal_Position':INITIAL.copy(),'Torque_Enable':[0]*6,
            'Operating_Mode':[0]*6,'P_Coefficient':[16]*6,'I_Coefficient':[0]*6,'D_Coefficient':[32]*6,
            'Lock':[1]*6,'Present_Current':[0]*6,'Present_Load':[0]*6,'Present_Voltage':[125]*6,
            'Present_Temperature':[30]*6,'Status':[0]*6}
    def connect(self,**kw):pass
    def disconnect(self,disable_torque=True):assert not disable_torque;self.port_handler.closePort()
    def sync_read(self,name,**kw):return dict(zip(MOTORS,self.data[name]))
    def read(self,name,joint,**kw):return self.data[name][MOTORS.index(joint)]
    def sync_write(self,name,values,**kw):
        raw=[values[n] for n in MOTORS]
        address,length=(42,2) if name=='Goal_Position' else (40,1)
        self.port_handler.writePort(sync_packet(address,length,raw));self.writes.append((name,raw.copy()))
        self.data[name]=raw.copy()
        if name=='Goal_Position':
            self.data['Torque_Enable']=[1]*6
            if self.fault=='goal' and not self.fired:self.fired=True;raise ConnectionError('partial goal TX')
            if self.fault=='tracking':self.data['Present_Position'][2]=INITIAL[2]-24
            else:self.data['Present_Position']=raw.copy()
        if name=='Torque_Enable' and self.fault=='release':raise ConnectionError('release unavailable')
    def write(self,name,joint,value,**kw):
        self.port_handler.writePort(gain_packet(value));self.writes.append((name,value));self.data[name][2]=value
        if self.fault=='p20' and value==20:raise ConnectionError('partial P20 TX')
        if self.fault=='restore' and value==16:raise ConnectionError('restore unavailable')

class Clock:
    def __init__(self):self.t=0
    def now(self):return self.t
    def sleep(self,s):self.t+=s

class TestComparison(unittest.TestCase):
    def run_case(self,mode='compare',fault=None,ticks=10,edit=None):
        bus=Bus(fault)
        if edit:edit(bus)
        arm=BoundedArm('fake',CAL,mode!='check',lambda:bus);clock=Clock()
        report={'trials':[],'status':'starting'}
        run_session(arm,CAL,mode,report,clock=clock.now,sleep=clock.sleep,ticks=ticks)
        return report,bus
    def test_check_never_writes(self):
        r,b=self.run_case('check');self.assertEqual(r['status'],'checked');self.assertEqual(b.writes,[])
    def test_small_and_large_success_identical_ab_targets_restored(self):
        for ticks in (10,33):
            r,b=self.run_case(ticks=ticks)
            self.assertEqual(r['status'],'completed',r.get('error'))
            self.assertTrue(r['p_restored_to_16']);self.assertEqual(b.data['Torque_Enable'],[0]*6)
            self.assertEqual([x['command'] for x in r['trials'][0]['rows']],[x['command'] for x in r['trials'][1]['rows']])
            self.assertEqual(b.data['P_Coefficient'],[16]*6)
    def test_goal_failure_releases(self):
        r,b=self.run_case(fault='goal');self.assertEqual(r['status'],'failed');self.assertEqual(b.data['Torque_Enable'],[0]*6)
        self.assertEqual(len(r['trials']),1)
    def test_partial_p20_write_restores(self):
        r,b=self.run_case(fault='p20');self.assertEqual(r['status'],'failed');self.assertTrue(r['p_restored_to_16'])
    def test_restore_failure_not_claimed_success(self):
        r,b=self.run_case(fault='restore');self.assertEqual(r['status'],'cleanup_not_confirmed');self.assertIn('p_restore_error',r)
    def test_release_failure_not_claimed_success(self):
        r,b=self.run_case(fault='release');self.assertEqual(r['status'],'cleanup_not_confirmed')
    def test_tracking_fault_saved_no_next_trial(self):
        r,b=self.run_case(fault='tracking');self.assertEqual(r['status'],'failed');self.assertEqual(len(r['trials']),1)
        self.assertEqual(b.data['Torque_Enable'],[0]*6)
    def test_outside_pose_has_zero_writes(self):
        r,b=self.run_case(edit=lambda b:b.data['Present_Position'].__setitem__(4,2031))
        self.assertEqual(r['status'],'failed');self.assertEqual(b.writes,[])
    def test_calibration_mismatch_has_zero_writes(self):
        r,b=self.run_case(edit=lambda b:setattr(b,'is_calibrated',False))
        self.assertEqual(r['status'],'failed');self.assertEqual(b.writes,[])
    def test_existing_torque_has_zero_writes(self):
        r,b=self.run_case(edit=lambda b:b.data.__setitem__('Torque_Enable',[1]*6))
        self.assertEqual(r['status'],'failed');self.assertEqual(b.writes,[])
    def test_trajectory_bounds(self):
        for ticks in (10,33):
            cmds=list(trajectory(INITIAL,ticks));self.assertEqual(cmds[-1],INITIAL)
            self.assertEqual(min(c[2] for c in cmds),INITIAL[2]-ticks)
            for a,b in zip(cmds,cmds[1:]):self.assertLessEqual(abs(a[2]-b[2]),1)
            for c in cmds:
                self.assertEqual(c[:2]+c[3:],INITIAL[:2]+INITIAL[3:])
        with self.assertRaises(ValueError):list(trajectory(INITIAL,100))
    def test_stop_after_p20_restores_without_second_motion(self):
        bus=Bus();arm=BoundedArm('fake',CAL,True,lambda:bus);clock=Clock();r={'trials':[]}
        run_session(arm,CAL,'compare',r,stop=lambda:bus.data['P_Coefficient'][2]==20,clock=clock.now,sleep=clock.sleep)
        self.assertEqual(r['status'],'stopped')
        self.assertTrue(r['p_restored_to_16'])
        self.assertEqual(bus.data['Torque_Enable'],[0]*6)
        self.assertEqual(r['trials'][1]['rows'],[])
    def test_sdk_gain_packet_and_unapproved_write(self):
        import scservo_sdk as sdk
        port=Port();gate=ExecutionGate(port.writePort,True);port.writePort=gate
        gate.expected=gain_packet(20);sdk.PacketHandler(0).write1ByteTxOnly(port,3,21,20)
        self.assertIsNone(gate.expected)
        with self.assertRaises(RuntimeError):gate(gain_packet(16))
        for value in (32,255,-1):
            with self.assertRaises(ValueError):gain_packet(value)

if __name__=='__main__':unittest.main()
