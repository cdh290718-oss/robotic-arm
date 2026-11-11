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
            self.assertLessEqual(command[5]-previous[5],27)
            self.assertLessEqual(command[5],CAL['gripper']['range_max'])
            limit.check_measured(command)
            previous=command.copy()
        self.assertEqual(command[5],CAL['gripper']['range_max'])
    def test_body_steps_and_tracking_unchanged(self):
        limit=GraspLimiter(CAL,INITIAL);state=limit.state(INITIAL);state[0]+=100
        command,_=limit.command(state,1/30)
        self.assertEqual(command[0]-INITIAL[0],16)
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

    def test_fixed_wrist_does_not_move(self):
        limit=GraspLimiter(CAL,INITIAL,freeze_wrist=True)
        state=limit.state(INITIAL);state[4]+=20
        cmd,_=limit.command(state,1/30)
        self.assertEqual(cmd[4],INITIAL[4])

    def test_stalled_joint_never_gets_goal_beyond_tracking_bound(self):
        limit=GraspLimiter(CAL,INITIAL,freeze_wrist=True)
        state=limit.state(INITIAL);state[0]+=100;state[5]=100
        for _ in range(40):
            limit.check_measured(INITIAL)
            cmd,_=limit.command(state,1/30)
            self.assertLessEqual(abs(cmd[0]-INITIAL[0])*DEG_PER_TICK,4)
            self.assertLessEqual(abs(cmd[5]-INITIAL[5])*100/(CAL['gripper']['range_max']-CAL['gripper']['range_min']),5)

    def test_per_joint_demo_rate_and_measured_guard(self):
        limit=GraspLimiter(CAL,INITIAL,freeze_wrist=True)
        state=limit.state(INITIAL)
        state[:4]=[x+90 for x in state[:4]]
        state[5]=100
        cmd,_=limit.command(state,1/30)
        # ID3 begins near its saved upper limit; it may be limited by calibration.
        self.assertEqual(cmd[0]-INITIAL[0],16)
        self.assertLessEqual(cmd[1]-INITIAL[1],45)
        self.assertLessEqual(cmd[2]-INITIAL[2],45)
        self.assertEqual(cmd[3]-INITIAL[3],21)
        self.assertEqual(cmd[4],INITIAL[4])
        self.assertEqual(cmd[5]-INITIAL[5],27)

if __name__=='__main__':unittest.main()
