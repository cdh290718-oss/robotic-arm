#!/usr/bin/env python
"""Bounded small-range motion test for the SO-101 follower (MolmoAct2 targets).

This is the ONLY script here that writes to the motors, and it is deliberately
minimal:

  * it actuates five joints (pan, lift, elbow, wrist_flex, gripper) and never
    touches wrist_roll (passive joint, per this project's rule);
  * before enabling torque it writes the *current* position as the goal, so
    enabling torque cannot produce a jump;
  * each commanded step is clamped (`--max-step`, default 2 deg) and the pace is
    rate-limited (`--speed-deg-s`, default 5 deg/s);
  * the final target is clamped to a fraction of the model delta
    (`--fraction`, default 0.25) and to the calibrated travel;
  * after every step it reads back and aborts if tracking error exceeds
    `--max-error` (default 5 deg) or if any joint leaves the travel range;
  * torque is left ENABLED at the end so the arm holds position (it is safer
    than dropping a free arm); the log says so explicitly.

Use --dry-run to print the plan without writing anything.

Example:
  python execute_small_move.py --target 7.56 -96.0 83.0 73.0 1.23 10.25 --dry-run
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
REF_CAL_PATH = Path(r"D:\桌面\萌芽杯\policy_v1_refined\reference_calibration.json")

NAMES = ("shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper")
ACTUATED = (0, 1, 2, 3, 5)  # indices; wrist_roll (4) stays passive
MAX_RES = 4095


def travel_limits(cal: dict) -> tuple[np.ndarray, np.ndarray]:
    lo, hi = np.zeros(6), np.zeros(6)
    for i, name in enumerate(NAMES):
        c = cal[name]
        if name == "gripper":
            lo[i], hi[i] = 0.0, 100.0
        else:
            mid = (c["range_min"] + c["range_max"]) / 2
            lo[i] = (c["range_min"] - mid) * 360 / MAX_RES
            hi[i] = (c["range_max"] - mid) * 360 / MAX_RES
    return lo, hi


def make_bus(port: str, cal: dict):
    sys.path.insert(0, str(LEROBOT_SRC))
    from lerobot.motors import Motor, MotorCalibration, MotorNormMode
    from lerobot.motors.feetech import FeetechMotorsBus

    motors = {
        n: Motor(
            i + 1,
            "sts3215",
            MotorNormMode.RANGE_0_100 if n == "gripper" else MotorNormMode.DEGREES,
        )
        for i, n in enumerate(NAMES)
    }
    bus = FeetechMotorsBus(
        port=port,
        motors=motors,
        calibration={n: MotorCalibration(**v) for n, v in cal.items()},
    )
    bus.connect()
    if not bus.is_calibrated:
        raise RuntimeError("on-device calibration differs from the saved calibration; refusing to move")
    return bus


def read_state(bus) -> np.ndarray:
    values = bus.sync_read("Present_Position")
    return np.asarray([float(values[n]) for n in NAMES])


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", default="COM5")
    ap.add_argument("--target", nargs=6, type=float, required=True,
                    help="model first action in ARM coordinates (after the bridge)")
    ap.add_argument("--fraction", type=float, default=0.25,
                    help="fraction of (target - current) to attempt")
    ap.add_argument("--max-step", type=float, default=2.0, help="max deg per step")
    ap.add_argument("--speed-deg-s", type=float, default=5.0)
    ap.add_argument("--max-error", type=float, default=5.0, help="abort above this tracking error")
    ap.add_argument("--hold-after", type=float, default=1.5)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--run-name", default="small_move")
    ap.add_argument("--run-root", default=str(HERE / "tmp" / "runs"))
    args = ap.parse_args()

    stamp = time.strftime("%Y%m%d_%H%M%S")
    run_dir = Path(args.run_root) / f"{stamp}_{args.run_name}"
    run_dir.mkdir(parents=True, exist_ok=True)
    report: dict = {
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "argv": sys.argv,
        "mode": "bounded small-range motion test",
        "status": "started",
    }
    log_path = run_dir / "report.json"

    def flush():
        log_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    cal = json.loads(CAL_PATH.read_text(encoding="utf-8-sig"))
    ref = json.loads(REF_CAL_PATH.read_text(encoding="utf-8-sig"))
    if cal != ref:
        raise SystemExit("calibration file differs from the training reference; not moving")
    lo, hi = travel_limits(cal)

    bus = make_bus(args.port, cal)
    try:
        state = read_state(bus)
        torque = bus.sync_read("Torque_Enable", normalize=False)
        target_model = np.asarray(args.target, dtype=np.float64)

        wanted = state + args.fraction * (target_model - state)
        wanted = np.clip(wanted, lo + 1.0, hi - 1.0)
        wanted[4] = state[4]  # passive joint: never commanded

        plan = []
        for i in NAMES:
            idx = NAMES.index(i)
            plan.append({
                "joint": i,
                "current": round(float(state[idx]), 3),
                "model_target": round(float(target_model[idx]), 3),
                "wanted": round(float(wanted[idx]), 3),
                "delta": round(float(wanted[idx] - state[idx]), 3),
                "travel": [round(float(lo[idx]), 2), round(float(hi[idx]), 2)],
                "actuated": idx in ACTUATED,
            })
        report.update({
            "torque_before": torque,
            "state_before": [round(float(x), 3) for x in state],
            "target_model": [round(float(x), 3) for x in target_model],
            "fraction": args.fraction,
            "max_step_deg": args.max_step,
            "speed_deg_s": args.speed_deg_s,
            "plan": plan,
        })
        print(json.dumps({"torque": torque,
                          "state": [round(float(x), 2) for x in state],
                          "wanted": [round(float(x), 2) for x in wanted],
                          "plan": plan}, indent=2))

        if args.dry_run:
            report["status"] = "dry_run"
            flush()
            print(f"dry run only; report: {log_path}")
            return 0

        # ---- 1. goal = current, BEFORE torque, so enabling cannot jump
        for idx in ACTUATED:
            bus.write("Goal_Position", NAMES[idx], float(state[idx]))
        report["goal_primed"] = True

        # ---- 2. enable torque on the five actuated joints only
        for idx in ACTUATED:
            bus.write("Torque_Enable", NAMES[idx], 1)
        report["torque_enabled_on"] = [NAMES[i] for i in ACTUATED]
        time.sleep(0.5)
        after_enable = read_state(bus)
        report["state_after_torque_enable"] = [round(float(x), 3) for x in after_enable]
        jump = float(np.max(np.abs(after_enable - state)))
        report["max_jump_on_enable_deg"] = round(jump, 3)
        if jump > 3.0:
            raise RuntimeError(f"jump of {jump:.2f} deg when enabling torque; aborting")

        # ---- 3. step towards the wanted pose
        current = after_enable.copy()
        steps = []
        remaining = np.abs(wanted - current)
        n_steps = int(np.ceil(remaining[list(ACTUATED)].max() / args.max_step)) if remaining.any() else 0
        n_steps = max(n_steps, 0)
        report["planned_steps"] = n_steps
        for k in range(n_steps):
            nxt = current.copy()
            for idx in ACTUATED:
                delta = wanted[idx] - current[idx]
                nxt[idx] = current[idx] + float(np.clip(delta, -args.max_step, args.max_step))
            for idx in ACTUATED:
                bus.write("Goal_Position", NAMES[idx], float(nxt[idx]))
            time.sleep(args.max_step / max(args.speed_deg_s, 0.5))
            got = read_state(bus)
            err = np.abs(got - nxt)
            steps.append({
                "step": k,
                "commanded": [round(float(nxt[i]), 3) for i in range(6)],
                "measured": [round(float(got[i]), 3) for i in range(6)],
                "max_error_deg": round(float(err[list(ACTUATED)].max()), 3),
            })
            print(f"step {k}: cmd={[round(float(nxt[i]),2) for i in range(6)]} "
                  f"meas={[round(float(got[i]),2) for i in range(6)]} "
                  f"err={err[list(ACTUATED)].max():.2f}", flush=True)
            if err[list(ACTUATED)].max() > args.max_error:
                report["abort_reason"] = f"tracking error {err.max():.2f} > {args.max_error}"
                break
            out_of_range = [NAMES[i] for i in range(6) if got[i] < lo[i] - 0.5 or got[i] > hi[i] + 0.5]
            if out_of_range:
                report["abort_reason"] = f"outside travel: {out_of_range}"
                break
            current = got
        report["steps"] = steps

        # ---- 4. hold and finish (torque stays ON)
        time.sleep(args.hold_after)
        final = read_state(bus)
        report["state_final"] = [round(float(x), 3) for x in final]
        report["total_delta_deg"] = [
            round(float(final[i] - state[i]), 3) for i in range(6)
        ]
        report["torque_enabled_after"] = bus.sync_read("Torque_Enable", normalize=False)
        report["status"] = "completed"
        report["note"] = (
            "torque is left ENABLED so the arm holds the final pose; "
            "disable it explicitly if you want the arm released"
        )
    except Exception:
        import traceback

        report["status"] = "failed"
        report["error"] = traceback.format_exc()
        raise
    finally:
        try:
            report["torque_at_exit"] = bus.sync_read("Torque_Enable", normalize=False)
        except Exception:  # noqa: BLE001
            pass
        try:
            bus.disconnect(disable_torque=False)
        except Exception:  # noqa: BLE001
            pass
        flush()
        print(f"report: {log_path} status={report.get('status')}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
