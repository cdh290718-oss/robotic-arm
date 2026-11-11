import unittest
from control_core import BoundedArm
from test_control import CAL,INITIAL,FakeBus

class Startup(unittest.TestCase):
    def test_snapshot_handles_unicode_directory(self):
        import tempfile
        import numpy as np
        from pathlib import Path
        from bounded_run import save_panel
        with tempfile.TemporaryDirectory(prefix='截图_') as directory:
            path=Path(directory)
            save_panel(path,{'images':{'1':np.zeros((8,8,3),dtype=np.uint8),'3':np.zeros((8,8,3),dtype=np.uint8)}},'camera.jpg')
            self.assertTrue((path/'camera.jpg').read_bytes().startswith(b'\xff\xd8'))
    def test_no_goal_activation_fails_and_releases(self):
        class Bus(FakeBus):
            def sync_write(self,name,values,**kw):
                super().sync_write(name,values,**kw)
                if name=='Goal_Position':self.data['Torque_Enable']=[0]*6
        bus=Bus();arm=BoundedArm('fake',CAL,True,lambda:bus);arm.connect()
        with self.assertRaises(RuntimeError):arm.enable_aligned(INITIAL)
        self.assertFalse(arm.startup_complete)
        self.assertEqual(arm.cleanup_motion()['action'],'release_failed_startup')
    def test_changed_gain_refused_before_goal(self):
        bus=FakeBus();bus.data['P_Coefficient'][2]=20
        arm=BoundedArm('fake',CAL,True,lambda:bus);arm.connect()
        with self.assertRaises(RuntimeError):arm.enable_aligned(INITIAL)
        self.assertEqual(arm.guard.write_attempts,0)
    def test_partial_goal_write_cleanup(self):
        class Bus(FakeBus):
            def sync_write(self,name,values,**kw):
                super().sync_write(name,values,**kw)
                if name=='Goal_Position':raise ConnectionError('after TX')
        bus=Bus();arm=BoundedArm('fake',CAL,True,lambda:bus);arm.connect()
        with self.assertRaises(ConnectionError):arm.enable_aligned(INITIAL)
        self.assertTrue(arm.cleanup_motion()['confirmed'])
        self.assertEqual(bus.data['Torque_Enable'],[0]*6)

if __name__=='__main__':unittest.main()
