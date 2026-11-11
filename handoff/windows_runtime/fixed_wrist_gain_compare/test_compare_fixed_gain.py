import unittest
from compare_fixed_gain import GainArm,gain_packet,run,set_gain,trajectory
from test_control import CAL,INITIAL,FakePort
from test_elbow_probe_fixed import Clock,FollowingBus
from control_core import ExecutionGate,MOTORS

class GainBus(FollowingBus):
    def __init__(self,fault=None):
        super().__init__();self.fault=fault;self.data['Present_Temperature']=[30]*6;self.injected=False
    def read(self,name,motor,**kwargs):
        value=super().read(name,motor,**kwargs)
        if self.fault=='late_torque' and name=='P_Coefficient' and value==20 and not self.injected:
            self.data['Torque_Enable'][4]=1;self.injected=True
        return value
    def write(self,name,motor,value,**kwargs):
        assert name=='P_Coefficient' and motor=='elbow_flex'
        self.port_handler.writePort(gain_packet(value))
        self.written.append((name,[3],[value]))
        self.data[name][2]=value
        if self.fault=='p20_partial' and value==20:raise ConnectionError('after P20 transmitted')
        if self.fault=='restore_failure' and value==16:raise ConnectionError('P16 restore failed')

class CompareTests(unittest.TestCase):
    def setup_case(self,fault=None):
        bus=GainBus(fault);arm=GainArm('fake',CAL,True,lambda:bus);clock=Clock()
        return arm,bus,clock,{'status':'starting','trials':[]}
    def test_both_trials_identical_targets_and_restore(self):
        arm,bus,c,r=self.setup_case();run(arm,CAL,r,c.now,c.sleep)
        self.assertEqual(r['status'],'completed',r.get('error'))
        self.assertEqual([x['command'] for x in r['trials'][0]['samples']],[x['command'] for x in r['trials'][1]['samples']])
        self.assertEqual(bus.data['P_Coefficient'],[16]*6);self.assertTrue(r['p16_restored'])
        self.assertEqual(bus.data['Torque_Enable'],[0]*6)
        self.assertTrue(all(5 not in ids for name,ids,_ in bus.written if name=='Goal_Position'))
        self.assertTrue(all(v==[20] or v==[16] for name,ids,v in bus.written if name=='P_Coefficient'))
    def test_partial_gain_write_restored_and_no_p20_motion(self):
        arm,bus,c,r=self.setup_case('p20_partial');run(arm,CAL,r,c.now,c.sleep)
        self.assertEqual(r['status'],'failed');self.assertTrue(r['p16_restored']);self.assertEqual(len(r['trials']),1)
        self.assertEqual(bus.data['Torque_Enable'],[0]*6);self.assertEqual(bus.data['P_Coefficient'][2],16)
    def test_failed_restore_is_reported(self):
        arm,bus,c,r=self.setup_case('restore_failure');run(arm,CAL,r,c.now,c.sleep)
        self.assertEqual(r['status'],'cleanup_not_confirmed');self.assertIn('p16_restore_error',r)
    def test_delayed_torque_after_gain_write_is_released_and_restored(self):
        arm,bus,c,r=self.setup_case('late_torque');run(arm,CAL,r,c.now,c.sleep)
        self.assertEqual(r['status'],'failed');self.assertTrue(r['cleanup_release_confirmed'])
        self.assertTrue(r['p16_restored']);self.assertEqual(bus.data['Torque_Enable'],[0]*6)
    def test_locked_gain_mismatch_aborts_before_any_goal(self):
        arm,bus,c,r=self.setup_case();bus.data['P_Coefficient'][2]=17;run(arm,CAL,r,c.now,c.sleep)
        self.assertEqual(r['status'],'failed');self.assertEqual(arm.guard.write_attempts,0)
    def test_stalled_elbow_produces_measurable_error_in_both_trials(self):
        class StalledBus(GainBus):
            def sync_write(self,name,values,**kwargs):
                elbow=self.data['Present_Position'][2]
                super().sync_write(name,values,**kwargs)
                if name=='Goal_Position':self.data['Present_Position'][2]=elbow
        bus=StalledBus();arm=GainArm('fake',CAL,True,lambda:bus);c=Clock();r={'status':'starting','trials':[]}
        run(arm,CAL,r,c.now,c.sleep)
        self.assertEqual(r['status'],'completed',r.get('error'))
        self.assertEqual([t['metrics']['error_ticks'] for t in r['trials']],[10,10])
        self.assertEqual(bus.data['P_Coefficient'][2],16);self.assertEqual(bus.data['Torque_Enable'],[0]*6)
    def test_p20_ignored_fails_before_p20_motion_and_restores(self):
        class IgnoreP20(GainBus):
            def write(self,name,motor,value,**kwargs):
                super().write(name,motor,value,**kwargs)
                if value==20:self.data[name][2]=16
        bus=IgnoreP20();arm=GainArm('fake',CAL,True,lambda:bus);c=Clock();r={'status':'starting','trials':[]}
        run(arm,CAL,r,c.now,c.sleep)
        self.assertEqual(r['status'],'failed');self.assertEqual(len(r['trials']),1)
        self.assertTrue(r['p16_restored']);self.assertEqual(bus.data['Torque_Enable'],[0]*6)
    def test_real_sdk_single_motor_packet(self):
        import scservo_sdk as sdk
        port=FakePort();gate=ExecutionGate(port.writePort,True,True);port.writePort=gate
        gate.expected=gain_packet(20);sdk.PacketHandler(0).write1ByteTxOnly(port,3,21,20)
        self.assertIsNone(gate.expected);self.assertEqual(gate.write_packets,1)
        for x in (15,21,0):
            with self.assertRaises(ValueError):gain_packet(x)
    def test_trajectory_never_moves_other_joint(self):
        for command in trajectory(INITIAL):
            self.assertEqual([command[i] for i in (0,1,3,4,5)],[INITIAL[i] for i in (0,1,3,4,5)])
            self.assertTrue(INITIAL[2]-10<=command[2]<=INITIAL[2])

if __name__=='__main__':unittest.main()
