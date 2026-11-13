#!/usr/bin/env python
"""Gripper-axis mapping for THIS rig (SO-101 follower, LeRobot 0.6.0).

Measured on the hardware (wrist camera, jaws visible):

    arm gripper   0  -> jaws wide open
    arm gripper 100  -> jaws closed

The training convention is the opposite way round: the official sample frame
(`Beegbrain/pick_lemon_and_drop_in_bowl`, episode 0 frame 0) has
`observation.state[5] = 1.097` and the jaws are **closed**, and the dataset's
gripper values only span about 1.1 .. 51.3.

So for this rig the value has to be reversed *and* rescaled into the model's
practical span.  Constants come from the checkpoint's own statistics:

    model state gripper q01 = 0.944  (closed end, as seen in the sample frame)
    model state gripper q99 = 44.138 (open end)

    model = OPEN - (OPEN - CLOSED) * arm / 100
    arm   = (OPEN - model) / (OPEN - CLOSED) * 100

This module is deliberately separate from `joint_frame_bridge.py`: the
sign/offset bridge is official and applies to every arm, while this reversal is
a property of *this* machine's assembly and must be re-measured if the gripper
is rebuilt.

`--check` prints a table so the mapping can be eyeballed against the jaws
before it is used in a control loop.
"""

from __future__ import annotations

import argparse

import numpy as np

# From model/norm_stats.json metadata_by_tag.so100_so101_molmoact2.state_stats
MODEL_CLOSED = 0.944
MODEL_OPEN = 44.138
SPAN = MODEL_OPEN - MODEL_CLOSED


def arm_gripper_to_model(arm_value):
    """arm 0 (open) -> MODEL_OPEN, arm 100 (closed) -> MODEL_CLOSED."""
    a = np.asarray(arm_value, dtype=np.float64)
    return MODEL_OPEN - SPAN * np.clip(a, 0.0, 100.0) / 100.0


def model_gripper_to_arm(model_value):
    """inverse of arm_gripper_to_model, clipped to 0..100."""
    g = np.asarray(model_value, dtype=np.float64)
    a = (MODEL_OPEN - g) / SPAN * 100.0
    return np.clip(a, 0.0, 100.0)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true", help="print the mapping table")
    args = ap.parse_args()
    if not args.check:
        raise SystemExit("use --check")
    print(f"model closed={MODEL_CLOSED:.3f}  open={MODEL_OPEN:.3f}  span={SPAN:.3f}")
    print(f"{'arm 0-100':>10s}{'model':>10s}{'arm back':>10s}")
    for a in (0, 6.57, 25, 50, 75, 100):
        m = arm_gripper_to_model(a)
        print(f"{a:10.2f}{float(m):10.3f}{float(model_gripper_to_arm(m)):10.3f}")
    # round-trip over the full arm span
    xs = np.linspace(0, 100, 101)
    err = np.abs(model_gripper_to_arm(arm_gripper_to_model(xs)) - xs)
    print(f"round-trip max error over 0..100: {err.max():.6f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
