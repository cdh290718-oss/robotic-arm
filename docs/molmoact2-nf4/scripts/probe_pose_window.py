#!/usr/bin/env python
"""Is a given start pose inside the checkpoint's representable window?

Takes one real synchronised frame pair and evaluates a few *pose candidates*
through the official joint-frame bridge:

    model_state = signs * local_state + offsets      (before normalisation)
    local_action = signs * (model_action - offsets)  (after un-normalisation)

For each candidate it reports
  * which dimensions are still clipped on the model side,
  * the first target in local coordinates,
  * whether the target sits on the action-range boundary (a boundary artifact
    rather than a motion proposal),
  * the mechanical travel from the calibration file.

Nothing is sent to the robot and the current pose is not changed.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
BRIDGE = HERE / "coordinate_bridge_20261113"
sys.path.insert(0, str(BRIDGE))
sys.path.insert(0, str(HERE))

from joint_frame_bridge import arm_to_model, model_to_arm  # noqa: E402

NORM_TAG = "so100_so101_molmoact2"
NAMES = (
    "shoulder_pan",
    "shoulder_lift",
    "elbow_flex",
    "wrist_flex",
    "wrist_roll",
    "gripper",
)
MAX_RES = 4095


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sample", required=True)
    ap.add_argument("--model-dir", default=str(HERE / "model"))
    ap.add_argument("--calibration", default=r"G:\LeRobot\data\calibration\robots\so_follower\lerobot.json")
    ap.add_argument("--num-steps", type=int, default=10)
    ap.add_argument("--seed", type=int, default=20261112)
    ap.add_argument("--task", default="put the yellow block into the basket")
    ap.add_argument("--out", default=str(HERE / "tmp" / "pose_window_probe.json"))
    args = ap.parse_args()

    import torch
    from PIL import Image

    import molmoact2_nf4 as m

    report: dict = {"created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "status": "started"}
    try:
        m.configure_cuda_allocator()
        torch.set_grad_enabled(False)
        model, load_report = m.load_model(
            Path(args.model_dir), device="cuda", dtype=torch.bfloat16, verbose=False
        )
        processor = m.load_processor(Path(args.model_dir))
        sys.path.insert(0, str(HERE / "patches"))
        import embedding_no_cat

        embedding_no_cat.install(m.remote_module("modeling_molmoact2", Path(args.model_dir)))

        sample = Path(args.sample)
        meta = json.loads((sample / "meta.json").read_text(encoding="utf-8"))
        top = Image.open(sample / "top.jpg").convert("RGB")
        wrist = Image.open(sample / "wrist.jpg").convert("RGB")
        measured = np.asarray(meta["state"], dtype=np.float64)

        norm = json.loads((Path(args.model_dir) / "norm_stats.json").read_text("utf-8"))
        meta_tag = norm["metadata_by_tag"][NORM_TAG]
        st, at = meta_tag["state_stats"], meta_tag["action_stats"]
        q01 = np.asarray(st["q01"], dtype=np.float64)
        q99 = np.asarray(st["q99"], dtype=np.float64)
        a01 = np.asarray(at["q01"], dtype=np.float64)
        a99 = np.asarray(at["q99"], dtype=np.float64)

        # model-space windows converted back to local coordinates
        s_lo, s_hi = np.minimum(model_to_arm(q01), model_to_arm(q99)), np.maximum(
            model_to_arm(q01), model_to_arm(q99)
        )
        a_lo, a_hi = np.minimum(model_to_arm(a01), model_to_arm(a99)), np.maximum(
            model_to_arm(a01), model_to_arm(a99)
        )

        cal = json.loads(Path(args.calibration).read_text(encoding="utf-8-sig"))
        travel = {}
        for i, name in enumerate(NAMES):
            c = cal[name]
            if name == "gripper":
                travel[name] = [0.0, 100.0]
            else:
                mid = (c["range_min"] + c["range_max"]) / 2
                travel[name] = [
                    round((c["range_min"] - mid) * 360 / MAX_RES, 2),
                    round((c["range_max"] - mid) * 360 / MAX_RES, 2),
                ]

        candidates = {
            "current_measured": measured.copy(),
            # inside every window with a margin, still a plausible pre-grasp pose
            "candidate_a": np.asarray([0.0, -60.0, 60.0, 60.0, 0.0, 10.0]),
            "candidate_b": np.asarray([0.0, -40.0, 30.0, 55.0, 0.0, 15.0]),
            # deliberately outside on one joint, to show the failure mode
            "outside_lift_only": np.asarray([0.0, -101.0, 60.0, 60.0, 0.0, 10.0]),
        }

        results = {}
        for name, local in candidates.items():
            model_state = arm_to_model(local).astype(np.float32)
            z = np.clip(2.0 * (model_state.astype(np.float64) - q01) / (q99 - q01) - 1.0, -1, 1)
            clipped = np.abs(z) >= 0.999

            gen = torch.Generator(device="cuda").manual_seed(args.seed)
            torch.cuda.synchronize()
            out = model.predict_action(
                processor=processor,
                images=[top, wrist],
                task=args.task,
                state=model_state,
                norm_tag=NORM_TAG,
                inference_action_mode="continuous",
                enable_depth_reasoning=False,
                num_steps=args.num_steps,
                generator=gen,
                enable_cuda_graph=True,
            )
            torch.cuda.synchronize()
            chunk_model = out.actions.detach().float().cpu().numpy().reshape(-1, 6)
            chunk_local = model_to_arm(chunk_model)
            first = chunk_local[0]
            delta = first - local

            per_joint = {}
            for i, joint in enumerate(NAMES):
                per_joint[joint] = {
                    "input_local": round(float(local[i]), 3),
                    "input_model": round(float(model_state[i]), 3),
                    "normalized": round(float(z[i]), 4),
                    "clipped": bool(clipped[i]),
                    "first_action_local": round(float(first[i]), 3),
                    "delta": round(float(delta[i]), 3),
                    "at_action_min": bool(abs(first[i] - a_lo[i]) <= 0.75),
                    "at_action_max": bool(abs(first[i] - a_hi[i]) <= 0.75),
                    "mechanical_travel": travel[joint],
                }
            results[name] = {
                "input_local": [round(float(x), 3) for x in local],
                "clipped_dims": [NAMES[i] for i in range(6) if clipped[i]],
                "clipped_count": int(clipped.sum()),
                "first_action_local": [round(float(x), 3) for x in first],
                "delta": [round(float(x), 3) for x in delta],
                "max_abs_delta": round(float(np.abs(delta).max()), 3),
                "pinned_joints": [
                    j for j in NAMES
                    if per_joint[j]["at_action_min"] or per_joint[j]["at_action_max"]
                ],
                "per_joint": per_joint,
            }
            print(
                f"{name:20s} clipped={results[name]['clipped_count']}/6 "
                f"first={results[name]['first_action_local']} "
                f"max|d|={results[name]['max_abs_delta']:.2f} pinned={results[name]['pinned_joints']}",
                flush=True,
            )

        report.update(
            {
                "sample": str(sample),
                "windows": {
                    "state_window_local": {
                        NAMES[i]: [round(float(s_lo[i]), 2), round(float(s_hi[i]), 2)]
                        for i in range(6)
                    },
                    "action_window_local": {
                        NAMES[i]: [round(float(a_lo[i]), 2), round(float(a_hi[i]), 2)]
                        for i in range(6)
                    },
                    "mechanical_travel_local": travel,
                    "note": "state/action windows are the model q01..q99 converted to local coordinates",
                },
                "candidates": results,
                "images": "identical in every candidate",
                "seed": args.seed,
                "status": "completed",
            }
        )
    except Exception:
        import traceback

        report["status"] = "failed"
        report["error"] = traceback.format_exc()
        raise
    finally:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(f"report: {out_path} status={report['status']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
