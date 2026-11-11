import unittest
from elbow_probe import command_for, check
from test_control import INITIAL,CAL

class ProbeBounds(unittest.TestCase):
    def test_fixed_other_joints_and_subdegree_elbow(self):
        for step in range(11):
            cmd=command_for(INITIAL,step)
            self.assertEqual([cmd[i] for i in (0,1,3,4,5)],[INITIAL[i] for i in (0,1,3,4,5)])
            self.assertEqual(cmd[2],INITIAL[2]-step)
            check(cmd,INITIAL,cmd,CAL)
        for step in (-1,11,1.5):
            with self.assertRaises(ValueError): command_for(INITIAL,step)
    def test_measured_tracking_and_gripper_guards(self):
        for i,offset in ((2,-24),(5,16),(4,24)):
            raw=INITIAL.copy();raw[i]+=offset
            with self.assertRaises(RuntimeError): check(raw,INITIAL,INITIAL,CAL)

if __name__=='__main__': unittest.main()
