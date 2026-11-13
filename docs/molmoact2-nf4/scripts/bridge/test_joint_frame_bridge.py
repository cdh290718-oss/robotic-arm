import unittest
import numpy as np
from joint_frame_bridge import arm_to_model, model_to_arm, action_bounds_in_arm


class BridgeTests(unittest.TestCase):
    def test_documented_equations(self):
        local = np.array([-12, -102.5, 96.6, 74, 0, 1.2])
        np.testing.assert_allclose(arm_to_model(local), [-12, 192.5, 186.6, 74, 0, 1.2])
        np.testing.assert_allclose(model_to_arm([-12, 185, 173, 74, 0, 1.2]), [-12, -95, 83, 74, 0, 1.2])

    def test_roundtrip_chunks_no_mutation(self):
        values = np.random.default_rng(42).uniform(-270, 270, (4, 30, 6))
        before = values.copy()
        np.testing.assert_allclose(model_to_arm(arm_to_model(values)), values, atol=1e-12)
        np.testing.assert_array_equal(values, before)
        np.testing.assert_array_equal(arm_to_model(values)[..., 5], values[..., 5])

    def test_negative_sign_reverses_bounds(self):
        low, high = action_bounds_in_arm([0, 45, 35, 0, 0, 0], [1, 186, 174, 1, 1, 45])
        np.testing.assert_array_equal(low, [0, -96, -55, 0, 0, 0])
        np.testing.assert_array_equal(high, [1, 45, 84, 1, 1, 45])

    def test_invalid_input(self):
        for bad in ([1, 2], 3, [0, 0, float('nan'), 0, 0, 0], [0, 0, float('inf'), 0, 0, 0]):
            with self.assertRaises(ValueError):
                arm_to_model(bad)


if __name__ == '__main__':
    unittest.main()
