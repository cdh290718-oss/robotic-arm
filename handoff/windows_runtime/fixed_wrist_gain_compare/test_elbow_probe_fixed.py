import unittest
from control_core import BoundedArm, MOTORS
from test_control import CAL
from test_fixed_wrist import SubsetBus
from elbow_probe_fixed import run

class Clock:
    def __init__(self):self.t=0
    def now(self):return self.t
    def sleep(self,v):self.t+=v

class FollowingBus(SubsetBus):
    def __init__(self,unexpected=False):
        super().__init__(unexpected)
        for n in ('Status','Present_Current','Present_Load','Present_Voltage'):self.data[n]=[0]*6
    def sync_write(self,name,values,**kwargs):
        super().sync_write(name,values,**kwargs)
        if name=='Goal_Position':
            for n,v in values.items():self.data['Present_Position'][MOTORS.index(n)]=v

class ElbowTests(unittest.TestCase):
    def test_bounded_trajectory_and_passive_wrist(self):
        bus=FollowingBus();arm=BoundedArm('fake',CAL,True,lambda:bus,freeze_wrist=True);clock=Clock()
        report={'status':'starting','samples':[]};run(arm,CAL,report,sleep=clock.sleep,clock=clock.now)
        self.assertEqual(report['status'],'completed',report.get('error'))
        goals=[(ids,values) for name,ids,values in bus.written if name=='Goal_Position']
        self.assertTrue(all(5 not in ids for ids,_ in goals))
        self.assertEqual(max(abs(v[2]-goals[0][1][2]) for _,v in goals),10)
        self.assertEqual(bus.data['Torque_Enable'],[0]*6)
    def test_implicit_wrist_activation_releases(self):
        bus=FollowingBus(True);arm=BoundedArm('fake',CAL,True,lambda:bus,freeze_wrist=True);clock=Clock()
        report={'status':'starting','samples':[]};run(arm,CAL,report,sleep=clock.sleep,clock=clock.now)
        self.assertEqual(report['status'],'failed');self.assertEqual(bus.data['Torque_Enable'],[0]*6)

if __name__=='__main__':unittest.main()
