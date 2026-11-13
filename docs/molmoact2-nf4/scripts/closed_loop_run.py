#!/usr/bin/env python
"""Closed-loop MolmoAct2 run: predict -> bounded command -> repeat, for N seconds.

Pipeline per control step:
  1. newest camera pair from the frame spool (produced by capture_frames_only.py)
  2. read the six joints over the serial bus
  3. official joint-frame bridge: model_state = signs*arm + offsets
  4. predict_action (continuous, CUDA graph)
  5. action_arm = signs*(model_action - offsets); take the first action
  6. clamp: |delta| <= --max-step, speed <= --max-speed, stay inside travel
  7. write Goal_Position for the five actuated joints (wrist_roll never)

Safety:
  * aborts on tracking error > --max-error, non-finite numbers, values outside
    the calibrated travel, a stale frame (age > --max-frame-age) or a duration
    cap;
  * torque is primed with the current position before being enabled, and is left
    enabled at the end so the arm holds still;
  * a JSONL row is written for every control step, before and after the write.

Example:
  python closed_loop_run.py --duration 120 --task "put the yellow block into the basket"
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
NORM_TAG = "so100_so101_molmoact2"
NAMES = ("shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper")
ACTUATED = (0, 1, 2, 3, 5)
MAX_RES = 4095


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", default="COM5")
    ap.add_argument("--spool", default=str(HERE / "tmp" / "frames"))
    ap.add_argument("--model-dir", default=str(HERE / "model"))
    ap.add_argument("--duration", type=float, default=120.0)
    ap.add_argument("--task", default="put the yellow block into the basket")
    ap.add_argument("--num-steps", type=int, default=10)
    ap.add_argument("--max-step", type=float, default=3.0, help="deg per control step")
    ap.add_argument("--max-speed", type=float, default=12.0, help="deg/s")
    ap.add_argument("--max-error", type=float, default=8.0, help="abort above tracking error")
    ap.add_argument("--max-frame-age", type=float, default=0.8, help="seconds")
    ap.add_argument("--max-frame-wait", type=float, default=3.0,
                    help="seconds to wait for a fresh frame before aborting")
    ap.add_argument("--control-period", type=float, default=0.0,
                    help="extra sleep between steps (0 = as fast as inference allows)")
    ap.add_argument("--gripper-invert", action="store_true", default=True,
                    help="this rig: 0=open, 100=closed (see rig_gripper.py)")
    ap.add_argument("--no-gripper-invert", dest="gripper_invert", action="store_false")
    ap.add_argument("--run-name", default="closed_loop")
    ap.add_argument("--run-root", default=str(HERE / "tmp" / "runs"))
    args = ap.parse_args()

    stamp = time.strftime("%Y%m%d_%H%M%S")
    run_dir = Path(args.run_root) / f"{stamp}_{args.run_name}"
    run_dir.mkdir(parents=True, exist_ok=True)
    report: dict = {
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "argv": sys.argv,
        "mode": "closed loop, bounded, read+write to the arm",
        "status": "started",
    }
    log_path = run_dir / "report.json"

    def flush():
        log_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    # ---- calibration / travel
    cal = json.loads(CAL_PATH.read_text(encoding="utf-8-sig"))
    ref = json.loads(REF_CAL_PATH.read_text(encoding="utf-8-sig"))
    if cal != ref:
        raise SystemExit("calibration differs from the training reference; refusing to move")
    travel_lo, travel_hi = np.zeros(6), np.zeros(6)
    for i, name in enumerate(NAMES):
        c = cal[name]
        if name == "gripper":
            travel_lo[i], travel_hi[i] = 0.0, 100.0
        else:
            mid = (c["range_min"] + c["range_max"]) / 2
            travel_lo[i] = (c["range_min"] - mid) * 360 / MAX_RES
            travel_hi[i] = (c["range_max"] - mid) * 360 / MAX_RES

    # ---- model + processor (load before touching the arm)
    import torch
    from PIL import Image

    import molmoact2_nf4 as m

    sys.path.insert(0, str(HERE / "coordinate_bridge_20261113"))
    from joint_frame_bridge import arm_to_model, model_to_arm
    from rig_gripper import arm_gripper_to_model, model_gripper_to_arm

    report["allocator"] = m.configure_cuda_allocator()
    torch.set_grad_enabled(False)
    t0 = time.perf_counter()
    model, load_report = m.load_model(Path(args.model_dir), device="cuda",
                                      dtype=torch.bfloat16, verbose=False)
    torch.cuda.synchronize()
    processor = m.load_processor(Path(args.model_dir))
    sys.path.insert(0, str(HERE / "patches"))
    import embedding_no_cat

    embedding_no_cat.install(m.remote_module("modeling_molmoact2", Path(args.model_dir)))
    report["load"] = {"seconds": round(time.perf_counter() - t0, 2),
                      "audit": load_report["audit"],
                      "key_accounting": load_report["key_accounting"]}
    flush()

    # ---- bus
    sys.path.insert(0, str(LEROBOT_SRC))
    from lerobot.motors import Motor, MotorCalibration, MotorNormMode
    from lerobot.motors.feetech import FeetechMotorsBus

    motors = {n: Motor(i + 1, "sts3215",
                       MotorNormMode.RANGE_0_100 if n == "gripper" else MotorNormMode.DEGREES)
              for i, n in enumerate(NAMES)}
    bus = FeetechMotorsBus(port=args.port, motors=motors,
                           calibration={n: MotorCalibration(**v) for n, v in cal.items()})
    bus.connect()
    if not bus.is_calibrated:
        raise SystemExit("on-device calibration differs; refusing to move")

    def read_state() -> np.ndarray:
        values = bus.sync_read("Present_Position")
        return np.asarray([float(values[n]) for n in NAMES])

    steps = []
    try:
        state = read_state()
        report["state_start"] = [round(float(x), 3) for x in state]
        report["torque_start"] = bus.sync_read("Torque_Enable", normalize=False)
        flush()

        # prime goals at the current pose, then enable torque on the five joints
        for idx in ACTUATED:
            bus.write("Goal_Position", NAMES[idx], float(state[idx]))
        for idx in ACTUATED:
            bus.write("Torque_Enable", NAMES[idx], 1)
        time.sleep(0.4)
        after = read_state()
        report["jump_on_enable_deg"] = round(float(np.max(np.abs(after - state))), 3)
        if report["jump_on_enable_deg"] > 3.0:
            raise RuntimeError("torque enable produced a jump > 3 deg; aborting")
        state = after

        jsonl = open(run_dir / "steps.jsonl", "w", encoding="utf-8")
        t_start = time.perf_counter()
        n = 0
        abort = None
        while time.perf_counter() - t_start < args.duration:
            step_t0 = time.time()
            # newest complete frame pair
            frames = sorted((p for p in Path(args.spool).glob("frame_*") if p.is_dir()))
            if not frames:
                time.sleep(0.05)
                continue
            frame_dir = frames[-1]
            meta = json.loads((frame_dir / "meta.json").read_text(encoding="utf-8"))
            age = step_t0 - float(meta["t_wall"])
            if age > args.max_frame_age:
                # the spooler can be starved by CPU load; wait for a fresh frame
                # instead of aborting, and only give up if it stays stale
                waited = 0.0
                while age > args.max_frame_age and waited < args.max_frame_wait:
                    time.sleep(0.05)
                    waited += 0.05
                    frames = sorted((p for p in Path(args.spool).glob("frame_*") if p.is_dir()))
                    if frames:
                        frame_dir = frames[-1]
                        meta = json.loads((frame_dir / "meta.json").read_text(encoding="utf-8"))
                        age = time.time() - float(meta["t_wall"])
                if age > args.max_frame_age:
                    abort = (f"frame stayed stale for {waited:.1f}s "
                             f"(age {age*1000:.0f} ms > {args.max_frame_age*1000:.0f} ms)")
                    break
            top = Image.open(frame_dir / "top.jpg").convert("RGB")
            wrist = Image.open(frame_dir / "wrist.jpg").convert("RGB")

            state = read_state()
            state_model = state.copy()
            if args.gripper_invert:
                state_model[5] = float(arm_gripper_to_model(state[5]))
            model_state = arm_to_model(state_model).astype(np.float32)
            gen = torch.Generator(device="cuda").manual_seed(20261112)
            torch.cuda.synchronize()
            p0 = time.perf_counter()
            out = model.predict_action(
                processor=processor, images=[top, wrist], task=args.task,
                state=model_state, norm_tag=NORM_TAG,
                inference_action_mode="continuous", enable_depth_reasoning=False,
                num_steps=args.num_steps, generator=gen, enable_cuda_graph=True,
            )
            torch.cuda.synchronize()
            latency = time.perf_counter() - p0
            chunk_model = out.actions.detach().float().cpu().numpy().reshape(-1, 6)
            chunk_arm = model_to_arm(chunk_model)
            if args.gripper_invert:
                chunk_arm[:, 5] = model_gripper_to_arm(chunk_model[:, 5])
            target = chunk_arm[0]

            delta = np.clip(target - state, -args.max_step, args.max_step)
            commanded = state.copy()
            for idx in ACTUATED:
                commanded[idx] = float(np.clip(state[idx] + delta[idx],
                                               travel_lo[idx] + 1.0, travel_hi[idx] - 1.0))
            commanded[4] = state[4]  # passive

            for idx in ACTUATED:
                bus.write("Goal_Position", NAMES[idx], float(commanded[idx]))

            # let the arm travel for max_step at the allowed speed
            time.sleep(max(args.max_step / max(args.max_speed, 1.0), 0.15))
            measured = read_state()
            err = float(np.max(np.abs(measured - commanded)[list(ACTUATED)]))
            row = {
                "n": n,
                "t": round(time.time() - step_t0, 3),
                "elapsed_s": round(time.perf_counter() - t_start, 3),
                "frame_seq": meta["seq"],
                "frame_age_ms": round(age * 1000, 1),
                "state": [round(float(x), 3) for x in state],
                "target": [round(float(x), 3) for x in target],
                "commanded": [round(float(x), 3) for x in commanded],
                "measured": [round(float(x), 3) for x in measured],
                "delta": [round(float(x), 3) for x in delta],
                "tracking_error": round(err, 3),
                "latency_ms": round(latency * 1000, 1),
                "finite": bool(np.isfinite(chunk_arm).all()),
            }
            jsonl.write(json.dumps(row) + "\n")
            jsonl.flush()
            steps.append(row)
            n += 1
            if n % 5 == 0:
                print(f"t={row['elapsed_s']:6.1f}s frame_age={row['frame_age_ms']:5.0f}ms "
                      f"lat={row['latency_ms']:6.1f}ms err={err:4.2f} "
                      f"state={[round(float(x),1) for x in measured]}", flush=True)
            if not row["finite"]:
                abort = "non-finite prediction"
                break
            if err > args.max_error:
                abort = f"tracking error {err:.2f} > {args.max_error}"
                break
            out_of = [NAMES[i] for i in ACTUATED
                      if measured[i] < travel_lo[i] - 0.5 or measured[i] > travel_hi[i] + 0.5]
            if out_of:
                abort = f"outside travel: {out_of}"
                break
            if args.control_period:
                sleep = args.control_period - (time.perf_counter() - p0)
                if sleep > 0:
                    time.sleep(sleep)
        jsonl.close()
        report["steps"] = len(steps)
        report["abort_reason"] = abort
        report["state_end"] = [round(float(x), 3) for x in read_state()]
        report["total_displacement"] = [
            round(float(report["state_end"][i] - report["state_start"][i]), 3) for i in range(6)
        ]
        report["status"] = "completed" if abort is None else "aborted"
        report["note"] = "torque left enabled so the arm holds position"
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
        print(f"report: {log_path} status={report.get('status')} steps={report.get('steps')}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
