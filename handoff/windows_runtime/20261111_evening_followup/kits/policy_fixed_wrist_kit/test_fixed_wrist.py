import unittest
from control_core import BoundedArm, ExecutionGate, GraspLimiter, Limiter, MOTORS, sync_packet
from test_control import CAL, INITIAL, FakeBus, FakePort

class SubsetBus(FakeBus):
    def __init__(self,unexpected_enable=False):
        super().__init__(); self.unexpected_enable=unexpected_enable; self.written=[]
    def sync_write(self,name,values,**kwargs):
        ids=[MOTORS.index(n)+1 for n in values]
        raw=[values[n] for n in values]
        addr,length=(42,2) if name=='Goal_Position' else (40,1)
        self.port_handler.writePort(sync_packet(addr,length,raw,ids))
        self.written.append((name,ids,raw));self.events.append('write_'+name)
        for n,v in values.items(): self.data[name][MOTORS.index(n)]=v
        if name=='Goal_Position':
            for n in values: self.data['Torque_Enable'][MOTORS.index(n)]=1
            if self.unexpected_enable:self.data['Torque_Enable'][4]=1

class FixedWristTests(unittest.TestCase):
    def test_out_of_distribution_wrist_rejected_before_motion(self):
        from bounded_run import validate_fixed_wrist_distribution
        validate_fixed_wrist_distribution({'timings':{'state_zscore':[0,0,0,0,4.9,0]}})
        for z in (-23.484941482543945, 5.01, float('nan'), float('inf')):
            with self.assertRaisesRegex(RuntimeError,'no goal write'):
                validate_fixed_wrist_distribution({'timings':{'state_zscore':[0,0,0,0,z,0]}})
    def test_missing_wrist_distribution_rejected(self):
        from bounded_run import validate_fixed_wrist_distribution
        with self.assertRaisesRegex(RuntimeError,'no goal write'):validate_fixed_wrist_distribution({})
    def arm(self,unexpected=False):
        bus=SubsetBus(unexpected);arm=BoundedArm('fake',CAL,True,lambda:bus,freeze_wrist=True)
        arm.connect();return arm,bus
    def test_packet_boundary_blocks_goal_with_id5_and_torque_on(self):
        port=FakePort();gate=ExecutionGate(port.writePort,True,True)
        with self.assertRaises(RuntimeError):
            with gate.permit('Goal_Position',INITIAL):gate(sync_packet(42,2,INITIAL))
        with self.assertRaises(ValueError):
            with gate.permit('Torque_Enable',[1]*6):pass
        self.assertEqual(port.sent,[])
    def test_sdk_goal_packet_excludes_motor_5(self):
        import scservo_sdk as sdk
        port=FakePort();gate=ExecutionGate(port.writePort,True,True);port.writePort=gate
        ids=[1,2,3,4,6];params=[]
        for i in ids:
            v=INITIAL[i-1];params.extend([i,v&255,v>>8])
        with gate.permit('Goal_Position',INITIAL):
            sdk.PacketHandler(0).syncWriteTxOnly(port,42,2,params,len(params))
        self.assertEqual(gate.write_packets,1)
        self.assertEqual(port.sent[0],sync_packet(42,2,[INITIAL[i-1] for i in ids],ids))
    def test_active_goal_and_hold_keep_wrist_passive(self):
        arm,bus=self.arm();arm.enable_aligned(INITIAL)
        self.assertEqual(bus.data['Torque_Enable'],[1,1,1,1,0,1])
        goal=INITIAL.copy();goal[0]+=1;goal[4]=1966
        arm.write_register('Goal_Position',goal)
        result=arm.cleanup_motion()
        self.assertTrue(result['confirmed']);self.assertTrue(result['fixed_wrist_torque_off'])
        for name,ids,_ in bus.written:
            if name=='Goal_Position': self.assertNotIn(5,ids)
    def test_unexpected_wrist_torque_releases_all(self):
        arm,bus=self.arm(unexpected=True)
        with self.assertRaisesRegex(RuntimeError,'Fixed wrist torque'):arm.enable_aligned(INITIAL)
        result=arm.cleanup_motion()
        self.assertEqual(result['action'],'release_fixed_wrist_fault')
        self.assertEqual(bus.data['Torque_Enable'],[0]*6)
        self.assertEqual([ids for name,ids,_ in bus.written if name=='Goal_Position'],[[1,2,3,4,6]])
    def test_late_wrist_torque_fault_releases_without_another_goal(self):
        arm,bus=self.arm();arm.enable_aligned(INITIAL)
        bus.data['Torque_Enable'][4]=1
        with self.assertRaisesRegex(RuntimeError,'Fixed wrist torque'):arm.write_register('Goal_Position',INITIAL)
        result=arm.cleanup_motion()
        self.assertEqual(result['action'],'release_fixed_wrist_fault')
        self.assertEqual(bus.data['Torque_Enable'],[0]*6)
        self.assertEqual([ids for name,ids,_ in bus.written if name=='Goal_Position'],[[1,2,3,4,6]])
    def test_implicit_wrist_activation_during_running_is_released(self):
        arm,bus=self.arm();arm.enable_aligned(INITIAL)
        bus.unexpected_enable=True
        with self.assertRaisesRegex(RuntimeError,'Fixed wrist torque'):arm.write_register('Goal_Position',INITIAL)
        self.assertTrue(arm.cleanup_motion()['confirmed'])
        self.assertEqual(bus.data['Torque_Enable'],[0]*6)
    def test_excluded_wrist_position_remains_real_and_is_monitored(self):
        raw=INITIAL.copy();raw[4]=1966
        for cls in (Limiter,GraspLimiter):
            lim=cls(CAL,raw,freeze_wrist=True)
            self.assertAlmostEqual(lim.state(raw)[4],(1966-2046.5)*360/4095)
            action=lim.state(raw);action[4]=49
            goal,clipped=lim.command(action,1/30)
            self.assertEqual(goal[4],1966);self.assertTrue(clipped[4])
            still=raw.copy();still[4]+=11;lim.check_measured(still,enforce_tracking=False)
            still[4]+=1
            with self.assertRaisesRegex(RuntimeError,'Fixed wrist moved'):lim.check_measured(still,enforce_tracking=False)
    def test_other_joint_bounds_remain_enforced(self):
        raw=INITIAL.copy();raw[4]=1966;raw[3]=CAL['wrist_flex']['range_max']+1
        with self.assertRaisesRegex(ValueError,'wrist_flex'):GraspLimiter(CAL,raw,freeze_wrist=True)

if __name__=='__main__':unittest.main()
