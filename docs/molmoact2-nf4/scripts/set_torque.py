#!/usr/bin/env python
"""Turn torque on or off for the five actuated joints (never wrist_roll).

`--off` releases the arm: it will go slack and may fall, so hold it first.
`--on` primes the goal with the current position before enabling, so the arm
holds where it is instead of jumping.

  python set_torque.py --off
  python set_torque.py --on
  python set_torque.py --status
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
LEROBOT_SRC = Path(r"G:\LeRobot\source\lerobot-0.6.0\src")
CAL_PATH = Path(r"G:\LeRobot\data\calibration\robots\so_follower\lerobot.json")
NAMES = ("shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper")
ACTUATED = (0, 1, 2, 3, 5)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", default="COM5")
    ap.add_argument("--on", action="store_true")
    ap.add_argument("--off", action="store_true")
    ap.add_argument("--status", action="store_true")
    args = ap.parse_args()
    if not (args.on or args.off or args.status):
        raise SystemExit("choose --on / --off / --status")

    sys.path.insert(0, str(LEROBOT_SRC))
    from lerobot.motors import Motor, MotorCalibration, MotorNormMode
    from lerobot.motors.feetech import FeetechMotorsBus

    cal = json.loads(CAL_PATH.read_text(encoding="utf-8-sig"))
    motors = {
        n: Motor(i + 1, "sts3215",
                 MotorNormMode.RANGE_0_100 if n == "gripper" else MotorNormMode.DEGREES)
        for i, n in enumerate(NAMES)
    }
    bus = FeetechMotorsBus(port=args.port, motors=motors,
                           calibration={n: MotorCalibration(**v) for n, v in cal.items()})
    bus.connect()
    try:
        state = np.asarray([float(bus.sync_read("Present_Position")[n]) for n in NAMES])
        torque = bus.sync_read("Torque_Enable", normalize=False)
        print("state :", [round(float(x), 2) for x in state])
        print("torque:", torque)
        if args.status:
            return 0
        if args.off:
            for idx in ACTUATED:
                bus.write("Torque_Enable", NAMES[idx], 0)
            print("torque disabled on", [NAMES[i] for i in ACTUATED],
                  "-> the arm is now free and may fall")
        if args.on:
            for idx in ACTUATED:
                bus.write("Goal_Position", NAMES[idx], float(state[idx]))
            time.sleep(0.1)
            for idx in ACTUATED:
                bus.write("Torque_Enable", NAMES[idx], 1)
            time.sleep(0.3)
            after = np.asarray([float(bus.sync_read("Present_Position")[n]) for n in NAMES])
            print("torque enabled on", [NAMES[i] for i in ACTUATED],
                  "jump:", round(float(np.max(np.abs(after - state))), 3), "deg")
        print("torque now:", bus.sync_read("Torque_Enable", normalize=False))
    finally:
        bus.disconnect(disable_torque=False)
    return 0


if __name__ == "__main__":
    sys.exit(main())
