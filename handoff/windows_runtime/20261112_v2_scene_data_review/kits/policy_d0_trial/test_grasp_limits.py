import unittest
from control_core import GraspLimiter,DEG_PER_TICK
from test_control import CAL,INITIAL

class GraspBounds(unittest.TestCase):
    def test_gripper_rate_and_calibration_range(self):
        limit=GraspLimiter(CAL,INITIAL)
        state=limit.state(INITIAL);state[5]=100
        previous=INITIAL.copy()
        for _ in range(1000):
            command,_=limit.command(state,1/30)
            self.assertLessEqual(command[5]-previous[5],2)
            self.assertLessEqual(command[5],CAL['gripper']['range_max'])
            previous=command.copy()
        self.assertEqual(command[5],CAL['gripper']['range_max'])
    def test_body_steps_and_tracking_unchanged(self):
        limit=GraspLimiter(CAL,INITIAL);state=limit.state(INITIAL);state[0]+=100
        command,_=limit.command(state,1/30)
        self.assertEqual(command[0]-INITIAL[0],1)
        raw=command.copy();raw[0]+=46
        with self.assertRaisesRegex(RuntimeError,'4 degrees'):limit.check_measured(raw)
    def test_gripper_contact_error_stops(self):
        limit=GraspLimiter(CAL,INITIAL);raw=INITIAL.copy();raw[5]+=80
        with self.assertRaisesRegex(RuntimeError,'5 percentage'):limit.check_measured(raw)
    def test_nonfinite_and_out_of_range(self):
        limit=GraspLimiter(CAL,INITIAL)
        with self.assertRaises(ValueError):limit.command([float('nan')]*6,1/30)
        raw=INITIAL.copy();raw[4]=CAL['wrist_roll']['range_max']+3
        with self.assertRaises(RuntimeError):limit.check_measured(raw)

if __name__=='__main__':unittest.main()
