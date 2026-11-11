import unittest
from activation_probe import run
from control_core import BoundedArm
from test_control import CAL,INITIAL
from test_fixed_wrist import SubsetBus

class ProbeTests(unittest.TestCase):
    def test_success_sends_one_five_id_goal_then_release(self):
        bus=SubsetBus();bus.data['Status']=[0]*6;arm=BoundedArm('fake',CAL,True,lambda:bus,freeze_wrist=True)
        r={'status':'starting','samples':[]};run(arm,CAL,r,sleep=lambda _:None)
        self.assertEqual(r['status'],'completed',r.get('error'))
        self.assertEqual([ids for name,ids,_ in bus.written if name=='Goal_Position'],[[1,2,3,4,6]])
        self.assertEqual(bus.data['Torque_Enable'],[0]*6);self.assertTrue(r['release_confirmed'])
    def test_unexpected_wrist_activation_releases_without_second_goal(self):
        bus=SubsetBus(unexpected_enable=True);arm=BoundedArm('fake',CAL,True,lambda:bus,freeze_wrist=True)
        r={'status':'starting','samples':[]};run(arm,CAL,r,sleep=lambda _:None)
        self.assertEqual(r['status'],'failed');self.assertTrue(r['release_confirmed'])
        self.assertEqual(bus.data['Torque_Enable'],[0]*6)
        self.assertEqual([ids for name,ids,_ in bus.written if name=='Goal_Position'],[[1,2,3,4,6]])
    def test_preflight_outside_range_never_writes(self):
        bus=SubsetBus();bus.data['Present_Position'][4]=1966
        arm=BoundedArm('fake',CAL,True,lambda:bus,freeze_wrist=True)
        r={'status':'starting','samples':[]};run(arm,CAL,r,sleep=lambda _:None)
        self.assertEqual(r['status'],'failed');self.assertEqual(arm.guard.write_attempts,0)

if __name__=='__main__':unittest.main()
