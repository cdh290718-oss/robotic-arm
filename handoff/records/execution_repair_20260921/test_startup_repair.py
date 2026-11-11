import unittest
from control_core import BoundedArm
from test_control import FakeBus, CAL, INITIAL

class StartupFaults(unittest.TestCase):
    def arm(self, bus):
        arm=BoundedArm('fake',CAL,True,bus_factory=lambda:bus)
        arm.connect()
        return arm

    def test_goal_implicitly_enables_abort_and_release_without_second_goal(self):
        class Bus(FakeBus):
            def sync_write(self,name,values,**kw):
                super().sync_write(name,values,**kw)
                if name=='Goal_Position': self.data['Torque_Enable']=[1]*6
        bus=Bus(); arm=self.arm(bus)
        with self.assertRaisesRegex(RuntimeError,'Torque_Enable'): arm.enable_aligned(INITIAL)
        self.assertTrue(arm.may_be_enabled)
        self.assertFalse(arm.startup_complete)
        self.assertEqual(arm.cleanup_motion()['action'],'release_failed_startup')
        self.assertEqual(bus.data['Torque_Enable'],[0]*6)
        self.assertEqual(bus.events.count('write_Goal_Position'),1)

    def test_partial_first_write_still_releases(self):
        class Bus(FakeBus):
            def sync_write(self,name,values,**kw):
                super().sync_write(name,values,**kw)
                if name=='Goal_Position':
                    self.data['Torque_Enable']=[1]*6
                    raise ConnectionError('partial TX')
        bus=Bus(); arm=self.arm(bus)
        with self.assertRaises(ConnectionError): arm.enable_aligned(INITIAL)
        self.assertTrue(arm.cleanup_motion()['confirmed'])
        self.assertEqual(bus.data['Torque_Enable'],[0]*6)

    def test_release_readback_failure_not_claimed_success(self):
        class Bus(FakeBus):
            def sync_write(self,name,values,**kw):
                super().sync_write(name,values,**kw)
                self.data['Torque_Enable']=[1]*6
        bus=Bus(); arm=self.arm(bus)
        with self.assertRaises(RuntimeError): arm.enable_aligned(INITIAL)
        with self.assertRaises(RuntimeError): arm.cleanup_motion()
        self.assertTrue(arm.may_be_enabled)

    def test_preflight_failure_has_no_cleanup_write(self):
        bus=FakeBus(); bus.data['Operating_Mode']=[1]*6; arm=self.arm(bus)
        with self.assertRaises(ValueError): arm.enable_aligned(INITIAL)
        self.assertEqual(arm.cleanup_motion()['action'],'none')
        self.assertEqual(arm.guard.write_attempts,0)

    def test_successful_execution_retains_hold_semantics(self):
        bus=FakeBus(); arm=self.arm(bus); arm.enable_aligned(INITIAL)
        self.assertEqual(arm.cleanup_motion()['action'],'hold')
        self.assertEqual(bus.data['Torque_Enable'],[1]*6)

if __name__=='__main__': unittest.main()
