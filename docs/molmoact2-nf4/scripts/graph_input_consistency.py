#!/usr/bin/env python
"""Does the captured CUDA graph consume *new* inputs on replay?

The 30-chunk benchmark repeats one observation, so it cannot distinguish
"replay executes the new observation" from "replay silently reuses the buffers
of the first capture".  This script answers that inside a single process:

  1. builds several observations (different saved images and/or different
     joint states) that share the same tensor shapes, so the action-flow graph
     key is identical and the manager takes the replay path,
  2. instruments ``inference.py``: captures, replays, graph-key misses and
     fallbacks are counted, so the path taken is known rather than assumed,
  3. alternates the observations, then compares every graph-on result with the
     graph-off result for the *same* observation and the same seed,
  4. also compares observations against each other, to show that the action
     block really depends on the input values.

Pass criteria:
  * every capture/replay counter movement is accounted for,
  * for each observation, graph-on vs graph-off max |diff| <= tolerance,
  * different observations produce clearly different action blocks.

Nothing here touches a robot, camera or serial port.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import traceback
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import molmoact2_nf4 as m  # noqa: E402

NORM_TAG = "so100_so101_molmoact2"
MAT = Path(r"D:\桌面\萌芽杯\policy_v1_refined\results\20261112_214119_107_Grasp_bf16")


def split_collage(path: Path):
    from PIL import Image

    img = Image.open(path).convert("RGB")
    w, h = img.size
    if w % 2:
        raise SystemExit(f"{path}: width {w} is not a two-up collage")
    return [img.crop((0, 0, w // 2, h)), img.crop((w // 2, 0, w, h))]


def synthetic_images(size=(640, 480)):
    from PIL import Image, ImageDraw

    out = []
    for _ in range(2):
        img = Image.new("RGB", size, (110, 110, 110))
        d = ImageDraw.Draw(img)
        d.rectangle([256, 264, 384, 360], fill=(230, 200, 40))
        out.append(img)
    return out


class GraphCounters:
    """Counts capture / replay / miss / fallback inside the remote inference module."""

    def __init__(self, inference_mod):
        self.mod = inference_mod
        self.n = {"capture": 0, "replay": 0, "graph_miss": 0, "fallback": 0, "eligible": 0}
        self._orig = {
            "capture": inference_mod._capture_cuda_graph,
            "run": inference_mod.ActionCudaGraphManager.run_action_flow,
            "can_use": inference_mod.ActionCudaGraphManager.can_use_action_flow,
        }
        counters = self

        def capture(*a, **k):
            counters.n["capture"] += 1
            return counters._orig["capture"](*a, **k)

        def run(self_, inputs, steps, run_loop):
            key = counters.mod._cuda_graph_key(inputs, steps)
            cache = self_.action_flow_graph
            if cache is None or cache.key != key:
                counters.n["graph_miss"] += 1
            else:
                counters.n["replay"] += 1
            return counters._orig["run"](self_, inputs, steps, run_loop)

        def can_use(self_, inputs):
            ok = counters._orig["can_use"](self_, inputs)
            counters.n["eligible" if ok else "fallback"] += 1
            return ok

        inference_mod._capture_cuda_graph = capture
        inference_mod.ActionCudaGraphManager.run_action_flow = run
        inference_mod.ActionCudaGraphManager.can_use_action_flow = can_use

    def snapshot(self) -> dict:
        return dict(self.n)

    def delta(self, before: dict) -> dict:
        return {k: self.n[k] - before.get(k, 0) for k in self.n}

    def restore(self) -> None:
        self.mod._capture_cuda_graph = self._orig["capture"]
        self.mod.ActionCudaGraphManager.run_action_flow = self._orig["run"]
        self.mod.ActionCudaGraphManager.can_use_action_flow = self._orig["can_use"]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", default=str(HERE / "model"))
    ap.add_argument("--num-steps", type=int, default=10)
    ap.add_argument("--seed", type=int, default=20261112)
    ap.add_argument("--task", default="put the yellow block into the basket")
    ap.add_argument("--tol", type=float, default=1e-3,
                    help="max |graph_on - graph_off| allowed per observation (degrees)")
    ap.add_argument("--min_separation", type=float, default=0.5,
                    help="observations must differ by at least this much (degrees)")
    ap.add_argument("--run-name", default="graph_consistency")
    ap.add_argument("--run-root", default=str(HERE / "tmp" / "runs"))
    args = ap.parse_args()

    stamp = time.strftime("%Y%m%d_%H%M%S")
    run_dir = Path(args.run_root) / f"{stamp}_{args.run_name}"
    run_dir.mkdir(parents=True, exist_ok=True)

    report: dict = {
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "run_dir": str(run_dir),
        "argv": sys.argv,
        "status": "started",
    }
    out_path = run_dir / "report.json"

    def flush():
        report["saved_to"] = str(out_path)
        out_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    try:
        print(json.dumps(m.configure_cuda_allocator(), indent=2))
        torch.set_grad_enabled(False)
        model, load_report = m.load_model(
            Path(args.model_dir), device="cuda", dtype=torch.bfloat16, verbose=False
        )
        processor = m.load_processor(Path(args.model_dir))
        report["load"] = {
            "load_seconds": None,
            "audit": load_report["audit"],
            "key_accounting": load_report["key_accounting"],
        }

        # ---------------- observations: same shapes, different content
        saved_last = split_collage(MAT / "camera_last_observation.jpg")
        saved_before = split_collage(MAT / "camera_before.jpg")
        synth = synthetic_images()
        state_last = np.asarray(
            [-11.5165, 26.8571, -27.6044, 88.3956, 0.2198, 2.1438], dtype=np.float32
        )
        state_other = np.asarray(
            [3.7802, -102.3736, 96.6154, 73.7143, 0.2198, 1.4523], dtype=np.float32
        )
        observations = [
            {"name": "saved_last_frame", "images": saved_last, "state": state_last},
            {"name": "saved_before_frame", "images": saved_before, "state": state_last},
            {"name": "saved_last_images_other_state", "images": saved_last, "state": state_other},
            {"name": "synthetic_images_and_state", "images": synth, "state": state_other},
        ]
        report["observations"] = [
            {"name": o["name"], "state_degrees": o["state"].tolist()} for o in observations
        ]

        inference_mod = m.remote_module("inference", Path(args.model_dir))
        counters = GraphCounters(inference_mod)
        actions_on: dict[str, np.ndarray] = {}
        actions_off: dict[str, np.ndarray] = {}
        per_call: list[dict] = []

        def run(obs, graph_on: bool) -> np.ndarray:
            gen = torch.Generator(device="cuda").manual_seed(args.seed)
            before = counters.snapshot()
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            out = model.predict_action(
                processor=processor,
                images=obs["images"],
                task=args.task,
                state=obs["state"],
                norm_tag=NORM_TAG,
                inference_action_mode="continuous",
                enable_depth_reasoning=False,
                num_steps=args.num_steps,
                generator=gen,
                enable_cuda_graph=graph_on,
            )
            torch.cuda.synchronize()
            wall = (time.perf_counter() - t0) * 1000
            a = out.actions.detach().float().cpu().numpy()
            delta = counters.delta(before)
            per_call.append(
                {
                    "observation": obs["name"],
                    "cuda_graph": graph_on,
                    "wall_ms": round(wall, 2),
                    "action_checksum": round(float(a.astype("float64").sum()), 6),
                    "first_row": [round(float(x), 4) for x in a.reshape(-1, a.shape[-1])[0]],
                    "graph_counters": delta,
                }
            )
            print(
                f"  {'graph_on ' if graph_on else 'graph_off'} {obs['name']:34s} "
                f"{wall:8.1f} ms  sum={a.astype('float64').sum():12.4f}  {delta}",
                flush=True,
            )
            return a

        # ---------------- phase 1: graph on, alternating observations so that
        # every replay after the first must copy new values into the graph.
        on_phase_start = counters.snapshot()
        for rep in range(2):
            for obs in observations:
                actions_on[obs["name"]] = run(obs, True)
        # repeat the first observation last: graph must still follow the input
        actions_on["saved_last_frame_repeat"] = run(observations[0], True)
        on_phase = counters.delta(on_phase_start)

        # ---------------- phase 2: graph off reference, same seeds
        off_phase_start = counters.snapshot()
        for obs in observations:
            actions_off[obs["name"]] = run(obs, False)
        off_phase = counters.delta(off_phase_start)

        # ---------------- comparisons
        comparisons = []
        for obs in observations:
            name = obs["name"]
            on = actions_on[name].reshape(-1, actions_on[name].shape[-1])
            off = actions_off[name].reshape(-1, actions_off[name].shape[-1])
            diff = float(np.max(np.abs(on - off)))
            comparisons.append(
                {
                    "observation": name,
                    "max_abs_diff_on_vs_off": diff,
                    "mean_abs_diff_on_vs_off": float(np.mean(np.abs(on - off))),
                    "allclose_tol": diff <= args.tol,
                    "first_row_on": [round(float(x), 4) for x in on[0]],
                    "first_row_off": [round(float(x), 4) for x in off[0]],
                }
            )
        # separation between observations (graph on)
        names = [o["name"] for o in observations]
        separations = []
        base = actions_on[names[0]].reshape(-1, 6)
        for name in names[1:]:
            other = actions_on[name].reshape(-1, 6)
            d = float(np.max(np.abs(base - other)))
            separations.append(
                {
                    "compared_with": names[0],
                    "observation": name,
                    "max_abs_diff": d,
                    "distinguishable": d >= args.min_separation,
                }
            )
        repeat_diff = float(
            np.max(
                np.abs(
                    actions_on["saved_last_frame_repeat"].reshape(-1, 6)
                    - actions_on["saved_last_frame"].reshape(-1, 6)
                )
            )
        )

        counts = counters.snapshot()
        verdict = {
            "all_on_vs_off_within_tolerance": all(c["allclose_tol"] for c in comparisons),
            "all_observations_distinguishable": all(s["distinguishable"] for s in separations),
            "replay_path_used": on_phase["replay"] > 0,
            "one_capture_for_one_key": on_phase["capture"] == 1
            and on_phase["graph_miss"] == 1,
            "no_fallback_while_graph_enabled": on_phase["fallback"] == 0,
            "graph_disabled_phase_never_replays": off_phase["replay"] == 0
            and off_phase["capture"] == 0,
            "repeat_of_first_observation_is_stable": repeat_diff <= args.tol,
        }
        verdict["passed"] = all(verdict.values())

        report.update(
            {
                "tolerance_degrees": args.tol,
                "min_separation_degrees": args.min_separation,
                "graph_counters_total": counts,
                "graph_counters_graph_on_phase": on_phase,
                "graph_counters_graph_off_phase": off_phase,
                "calls": per_call,
                "comparisons_on_vs_off": comparisons,
                "separations": separations,
                "repeat_first_observation_max_abs_diff": repeat_diff,
                "verdict": verdict,
                "note": (
                    "graph-off and graph-on runs share the same seed and the same "
                    "observation; a stale-input replay would show a large on-vs-off "
                    "difference for the later observations and near-zero separation "
                    "between observations"
                ),
                "status": "passed" if verdict["passed"] else "failed",
            }
        )
        np.savez_compressed(
            run_dir / "actions.npz",
            **{f"on_{k}": v.astype("float32") for k, v in actions_on.items()},
            **{f"off_{k}": v.astype("float32") for k, v in actions_off.items()},
        )
        counters.restore()
        print("\n=== verdict ===")
        print(json.dumps(verdict, indent=2))
        print(f"report: {out_path}")
        return 0 if verdict["passed"] else 1
    except Exception:
        report["status"] = "failed"
        report["error"] = traceback.format_exc()
        raise
    finally:
        flush()


if __name__ == "__main__":
    sys.exit(main())
