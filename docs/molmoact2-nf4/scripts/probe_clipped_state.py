#!/usr/bin/env python
"""Counterfactual probe: does the clipped state dimension drive the target gap?

Takes one real synchronised frame pair and the measured pose, then predicts
with a few counterfactual states in which only the clipped dimension changes.
The images never change, so any difference in the predicted chunk is caused by
the state input.

Nothing is sent to the robot; this is a pure offline re-run of the model on
already captured frames.

Usage:
  python probe_clipped_state.py --sample tmp/live_spool2/sample_000285
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
NORM_TAG = "so100_so101_molmoact2"
MOTORS = (
    "shoulder_pan",
    "shoulder_lift",
    "elbow_flex",
    "wrist_flex",
    "wrist_roll",
    "gripper",
)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sample", required=True, help="capture sample directory")
    ap.add_argument("--model-dir", default=str(HERE / "model"))
    ap.add_argument("--num-steps", type=int, default=10)
    ap.add_argument("--seed", type=int, default=20261112)
    ap.add_argument("--task", default="put the yellow block into the basket")
    ap.add_argument("--out", default=str(HERE / "tmp" / "clipped_state_probe.json"))
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
        measured = np.asarray(meta["state"], dtype=np.float32)

        norm = json.loads((Path(args.model_dir) / "norm_stats.json").read_text("utf-8"))
        st = norm["metadata_by_tag"][NORM_TAG]["state_stats"]
        q01 = np.asarray(st["q01"], dtype=np.float64)
        q50 = np.asarray(st["q50"], dtype=np.float64)
        q99 = np.asarray(st["q99"], dtype=np.float64)

        lift = MOTORS.index("shoulder_lift")
        variants = {
            "measured": measured.copy(),
            "lift_at_q01": _with(measured, lift, q01[lift]),
            "lift_at_q25": _with(measured, lift, q01[lift] + 0.25 * (q99[lift] - q01[lift])),
            "lift_at_q50": _with(measured, lift, q50[lift]),
            "lift_at_q99": _with(measured, lift, q99[lift]),
        }

        results = {}
        for name, state in variants.items():
            gen = torch.Generator(device="cuda").manual_seed(args.seed)
            torch.cuda.synchronize()
            out = model.predict_action(
                processor=processor,
                images=[top, wrist],
                task=args.task,
                state=state,
                norm_tag=NORM_TAG,
                inference_action_mode="continuous",
                enable_depth_reasoning=False,
                num_steps=args.num_steps,
                generator=gen,
                enable_cuda_graph=True,
            )
            torch.cuda.synchronize()
            chunk = out.actions.detach().float().cpu().numpy().reshape(-1, 6)
            results[name] = {
                "state": [round(float(x), 3) for x in state],
                "first_action": [round(float(x), 3) for x in chunk[0]],
                "first_minus_measured_state": [
                    round(float(a - s), 3) for a, s in zip(chunk[0], measured)
                ],
                "elbow_first": round(float(chunk[0][MOTORS.index("elbow_flex")]), 3),
            }
            print(f"{name:12s} state={results[name]['state']} first={results[name]['first_action']}", flush=True)

        base = np.asarray(results["measured"]["first_action"])
        for name, r in results.items():
            r["first_action_delta_vs_measured_state"] = [
                round(float(a - b), 3) for a, b in zip(np.asarray(r["first_action"]), base)
            ]

        report.update(
            {
                "sample": str(sample),
                "measured_state": [round(float(x), 3) for x in measured],
                "training_state_stats": {
                    "q01": st["q01"],
                    "q50": st["q50"],
                    "q99": st["q99"],
                },
                "variants": results,
                "images": "identical in every variant",
                "seed": args.seed,
                "note": (
                    "the only changing input is the shoulder_lift value; any change in the "
                    "predicted chunk is attributable to that dimension"
                ),
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


def _with(state: np.ndarray, index: int, value: float) -> np.ndarray:
    out = state.copy()
    out[index] = value
    return out


if __name__ == "__main__":
    sys.exit(main())
