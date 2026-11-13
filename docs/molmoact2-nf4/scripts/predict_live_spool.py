#!/usr/bin/env python
# WARNING: this entry point does NOT apply the official joint-frame bridge.
# It feeds LeRobot arm coordinates straight into the checkpoint, which is the
# interface bug documented in 坐标兼容复核_20261113.md. Keep it only for A/B
# comparison; use coordinate_bridge_20261113/predict_live_bridged.py for any
# real read-only check.
"""Consume a live read-only capture spool and predict continuously.

Runs in the project venv (low-integrity path) because it only needs CUDA and
the model; the camera/serial work happens in ``capture_live_readonly.py``
under a medium-integrity interpreter.

For each new sample it records the full (1, 30, 6) chunk, the first target,
the difference against the measured position, the normalised state and whether
each dimension is clipped.  Nothing is sent to the robot.

Usage:
  python predict_live_spool.py --spool tmp/live_spool --duration 60
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import traceback
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
NORM_TAG = "so100_so101_molmoact2"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--spool", default=str(HERE / "tmp" / "live_spool"))
    ap.add_argument("--model-dir", default=str(HERE / "model"))
    ap.add_argument("--duration", type=float, default=90.0)
    ap.add_argument("--num-steps", type=int, default=10)
    ap.add_argument("--seed", type=int, default=20261112)
    ap.add_argument("--task", default="put the yellow block into the basket")
    ap.add_argument("--patch-embedding", action="store_true", default=True)
    ap.add_argument("--cuda-graph", choices=["on", "off"], default="on")
    ap.add_argument("--run-name", default="live_predict")
    ap.add_argument("--run-root", default=str(HERE / "tmp" / "runs"))
    args = ap.parse_args()

    spool = Path(args.spool)
    stamp = time.strftime("%Y%m%d_%H%M%S")
    run_dir = Path(args.run_root) / f"{stamp}_{args.run_name}"
    run_dir.mkdir(parents=True, exist_ok=True)
    report: dict = {
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "run_dir": str(run_dir),
        "argv": sys.argv,
        "mode": "model consumes synchronised live samples; no motor command",
        "status": "started",
    }
    report_path = run_dir / "report.json"

    def flush():
        report["saved_to"] = str(report_path)
        report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    try:
        import torch
        from PIL import Image

        import molmoact2_nf4 as m

        report["allocator"] = m.configure_cuda_allocator()
        torch.set_grad_enabled(False)
        t0 = time.perf_counter()
        model, load_report = m.load_model(
            Path(args.model_dir), device="cuda", dtype=torch.bfloat16, verbose=False
        )
        torch.cuda.synchronize()
        processor = m.load_processor(Path(args.model_dir))
        sys.path.insert(0, str(HERE / "patches"))
        import embedding_no_cat

        report["embedding_patch"] = embedding_no_cat.install(
            m.remote_module("modeling_molmoact2", Path(args.model_dir))
        )
        report["load"] = {
            "seconds": round(time.perf_counter() - t0, 2),
            "audit": load_report["audit"],
            "key_accounting": load_report["key_accounting"],
        }
        flush()

        stats_payload = json.loads((Path(args.model_dir) / "norm_stats.json").read_text("utf-8"))
        st = stats_payload["metadata_by_tag"][NORM_TAG]["state_stats"]
        q01 = np.asarray(st["q01"], dtype=np.float64)
        q99 = np.asarray(st["q99"], dtype=np.float64)
        graph_on = args.cuda_graph == "on"

        samples_path = run_dir / "samples.jsonl"
        processed: list[int] = []
        t_start = time.perf_counter()
        last_seq = -1
        idle_since = None
        with open(samples_path, "w", encoding="utf-8") as fh:
            while time.perf_counter() - t_start < args.duration:
                # Newest completed sample directory; no shared pointer file.
                candidates = sorted(
                    (p for p in spool.glob("sample_*") if p.is_dir()),
                    key=lambda p: p.name,
                )
                if not candidates:
                    time.sleep(0.05)
                    continue
                sample_dir = candidates[-1]
                seq = int(sample_dir.name.split("_")[-1])
                if seq <= last_seq:
                    status_path = spool / "capture_status.json"
                    done = False
                    if status_path.exists():
                        try:
                            s = json.loads(status_path.read_text(encoding="utf-8"))
                            done = s.get("status") in ("completed", "failed")
                        except Exception:  # noqa: BLE001
                            done = False
                    if done:
                        if idle_since is None:
                            idle_since = time.perf_counter()
                        elif time.perf_counter() - idle_since > 1.0:
                            break
                    else:
                        idle_since = None
                    time.sleep(0.03)
                    continue
                idle_since = None
                last_seq = seq
                meta = json.loads((sample_dir / "meta.json").read_text(encoding="utf-8"))
                top = Image.open(sample_dir / "top.jpg").convert("RGB")
                wrist = Image.open(sample_dir / "wrist.jpg").convert("RGB")
                state = np.asarray(meta["state"], dtype=np.float32)

                gen = torch.Generator(device="cuda").manual_seed(args.seed)
                torch.cuda.synchronize()
                p0 = time.perf_counter()
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
                    enable_cuda_graph=graph_on,
                )
                torch.cuda.synchronize()
                latency = time.perf_counter() - p0

                chunk = out.actions.detach().float().cpu().numpy()
                rows = chunk.reshape(-1, chunk.shape[-1])
                first = rows[0]
                z = np.clip(2.0 * (state.astype(np.float64) - q01) / (q99 - q01) - 1.0, -1.0, 1.0)
                record = {
                    "seq": seq,
                    "capture_elapsed_s": meta["elapsed_s"],
                    "capture_image_state_skew_ms": meta["image_state_skew_ms"],
                    "predict_started_s": round(p0 - t_start, 3),
                    "state": [round(float(x), 4) for x in state],
                    "state_normalized": [round(float(x), 4) for x in z],
                    "state_clipped": [bool(abs(v) >= 0.999) for v in z],
                    "first_action": [round(float(x), 4) for x in first],
                    "first_action_minus_state": [
                        round(float(a - s), 4) for a, s in zip(first, state)
                    ],
                    "chunk": [[round(float(v), 4) for v in row] for row in rows],
                    "all_finite": bool(np.isfinite(rows).all()),
                    "latency_ms": round(latency * 1000, 1),
                    "cuda_alloc_GiB": round(torch.cuda.memory_allocated() / (1 << 30), 3),
                }
                fh.write(json.dumps(record) + "\n")
                fh.flush()
                processed.append(seq)
                if len(processed) % 5 == 0:
                    print(
                        f"seq={seq} lat={record['latency_ms']:6.1f}ms "
                        f"first={record['first_action']} clipped={sum(record['state_clipped'])}/6",
                        flush=True,
                    )
        report.update(
            {
                "samples_processed": len(processed),
                "first_seq": processed[0] if processed else None,
                "last_seq": processed[-1] if processed else None,
                "samples_jsonl": str(samples_path),
                "status": "completed",
            }
        )
    except Exception:
        report["status"] = "failed"
        report["error"] = traceback.format_exc()
        flush()
        raise
    finally:
        report["cuda_mem_end"] = {
            "peak_allocated_GiB": round(torch.cuda.max_memory_allocated() / (1 << 30), 3)
            if "torch" in sys.modules and torch.cuda.is_available()
            else None,
            "peak_reserved_GiB": round(torch.cuda.max_memory_reserved() / (1 << 30), 3)
            if "torch" in sys.modules and torch.cuda.is_available()
            else None,
        }
        flush()
        print(f"saved: {report.get('saved_to')} status={report.get('status')} "
              f"samples={report.get('samples_processed')}")
    return 0 if report.get("status") == "completed" else 1


if __name__ == "__main__":
    sys.exit(main())
