#!/usr/bin/env python
"""Offline inference / VRAM / latency verification for the pinned NF4 checkpoint.

No robot, camera, serial port or network access is used here: observations come
from image files and a joint-state vector passed on the command line.

Stages (recorded even when a later stage fails):

  1. environment + allocator report (before the first CUDA allocation)
  2. processor construction and a dry tokenisation of the prompt
  3. graph build on meta + NF4 install + dense weight load (timed, with peaks)
     + quant-state round-trip check + independent NF4 decode check
  4. first prediction (cold start, timed) with lm_head / embedding call counts
  5. warm-up chunks (not measured)
  6. N measured chunks, CUDA-event and wall-clock timing
  7. JSON report

The report path may be unwritable under the session sandbox (child processes
run at low integrity); in that case the report is written next to it in
``tmp/`` and the fallback path is printed and recorded.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import statistics
import subprocess
import sys
import time
import traceback
from pathlib import Path

HERE = Path(__file__).resolve().parent
NORM_TAG = "so100_so101_molmoact2"


def now_iso() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S%z")


def smi(query: str):
    try:
        r = subprocess.run(
            ["nvidia-smi", f"--query-gpu={query}", "--format=csv,noheader,nounits"],
            capture_output=True,
            text=True,
            timeout=20,
        )
        return r.stdout.strip() if r.returncode == 0 else None
    except Exception:  # noqa: BLE001
        return None


def host_memory() -> dict:
    try:
        import psutil

        vm = psutil.virtual_memory()
        proc = psutil.Process()
        return {
            "total_GiB": round(vm.total / (1 << 30), 2),
            "available_GiB": round(vm.available / (1 << 30), 2),
            "process_rss_GiB": round(proc.memory_info().rss / (1 << 30), 2),
        }
    except Exception as e:  # noqa: BLE001
        return {"error": f"{type(e).__name__}: {e}"}


def cuda_mem() -> dict:
    """Snapshot of CUDA memory.

    Naming is deliberate: ``free_GiB``/``nvidia_smi_*`` describe the moment of
    the call, while ``peak_allocated_GiB``/``peak_reserved_GiB`` are the maxima
    since the last ``reset_peak_memory_stats`` for the enclosing stage.  A
    point-in-time free value must never be quoted as the peak-time free value.
    """
    import torch

    if not torch.cuda.is_available():
        return {}
    free, total = torch.cuda.mem_get_info()
    return {
        "at": now_iso(),
        "free_GiB": round(free / (1 << 30), 3),
        "total_GiB": round(total / (1 << 30), 3),
        "allocated_GiB": round(torch.cuda.memory_allocated() / (1 << 30), 3),
        "reserved_GiB": round(torch.cuda.memory_reserved() / (1 << 30), 3),
        "peak_allocated_GiB": round(torch.cuda.max_memory_allocated() / (1 << 30), 3),
        "peak_reserved_GiB": round(torch.cuda.max_memory_reserved() / (1 << 30), 3),
        "nvidia_smi_used_MiB": smi("memory.used"),
        "nvidia_smi_free_MiB": smi("memory.free"),
        "fields": {
            "free_GiB": "CUDA-usable free memory at this instant (not a peak)",
            "peak_allocated_GiB": "max allocated since the stage reset",
            "peak_reserved_GiB": "max reserved since the stage reset",
        },
    }


def write_json(path: Path, payload: dict) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        return path
    except PermissionError:
        fallback = HERE / "tmp" / path.name
        fallback.parent.mkdir(parents=True, exist_ok=True)
        fallback.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(f"[warn] {path} not writable for this process; wrote {fallback}")
        return fallback


def graph_state(model) -> dict:
    """Whether the CUDA graphs really exist (not just that capture was asked).

    A non-null graph object proves capture happened for the key that was
    current at that moment; it does not by itself prove later replays consume
    new inputs.  That is what ``graph_input_consistency.py`` checks.
    """
    out: dict = {}
    mgr = getattr(model.model, "action_cuda_graph_manager", None)
    if mgr is not None:
        g = getattr(mgr, "action_flow_graph", None)
        key_repr = ""
        if g is not None:
            key_repr = repr(getattr(g, "key", ()))
        out["action_flow"] = {
            "manager_enabled": bool(getattr(mgr, "enabled", False)),
            "graph_captured": g is not None,
            "key_entries": None if g is None else len(getattr(g, "key", ())),
            "key_digest": None
            if g is None
            else __import__("hashlib").sha256(key_repr.encode()).hexdigest()[:16],
            "static_output_shape": None if g is None else list(g.output.shape),
        }
    depth = getattr(model, "depth_decode_cuda_graph_manager", None)
    if depth is not None:
        out["depth_decode"] = {
            "manager_enabled": bool(getattr(depth, "enabled", False)),
            "graph_captured": getattr(depth, "graph", None) is not None,
        }
    return out


class CallCounter:

    def __init__(self, module, name: str):
        self.name = name
        self.n = 0
        self.shapes: list = []
        self._handle = module.register_forward_hook(self._hook)

    def _hook(self, module, inputs, output):
        self.n += 1
        shape = tuple(output.shape) if hasattr(output, "shape") else None
        if shape not in self.shapes:
            self.shapes.append(shape)

    def report(self) -> dict:
        return {"module": self.name, "calls": self.n, "output_shapes": self.shapes}

    def reset(self) -> None:
        self.n = 0
        self.shapes = []

    def remove(self) -> None:
        self._handle.remove()


def load_images(path_a: str, path_b: str):
    from PIL import Image

    imgs = []
    meta = []
    for p in (path_a, path_b):
        img = Image.open(p).convert("RGB")
        meta.append({"path": p, "size": list(img.size), "mode": img.mode})
        imgs.append(img)
    return imgs, meta


def synthetic_images(size=(640, 480)) -> tuple[list, list]:
    """Deterministic non-photographic observation for the API smoke test.

    The checkpoint does not ship example images (the only sample in the repo is
    a docstring URL), so this is a deterministic stand-in: a grey field with a
    yellow square in a fixed position, identical in both views.
    """
    from PIL import Image, ImageDraw

    imgs = []
    for i in range(2):
        img = Image.new("RGB", size, (110, 110, 110))
        d = ImageDraw.Draw(img)
        d.rectangle([int(size[0] * 0.4), int(size[1] * 0.55),
                     int(size[0] * 0.6), int(size[1] * 0.75)],
                    fill=(230, 200, 40))
        imgs.append(img)
    return imgs, [{"synthetic": True, "size": list(size)} for _ in range(2)]


def build_prompt(model_mod, model_dir: Path, state, task: str, n_images: int) -> str:
    stats_payload = json.loads((model_dir / "norm_stats.json").read_text("utf-8"))
    cfg_payload = json.loads((model_dir / "config.json").read_text("utf-8"))
    stats = model_mod._RobotStats(stats_payload)
    meta = stats.get_metadata(NORM_TAG)
    norm = stats.normalize_state(state, NORM_TAG)
    disc = model_mod._build_discrete_state_string(
        norm, int(cfg_payload["num_state_tokens"])
    )
    return model_mod._build_robot_text(
        task=model_mod._normalize_question_text(task),
        style="robot_action",
        discrete_state_string=disc,
        setup_type=str(meta.get("setup_type", "")),
        control_mode=str(meta.get("control_mode", "")),
        add_setup_tokens=bool(cfg_payload["add_setup_tokens"]),
        add_control_tokens=bool(cfg_payload["add_control_tokens"]),
        num_images=n_images,
    )


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", default=str(HERE / "model"))
    ap.add_argument("--image-top", default=None, help="overhead view (camera1)")
    ap.add_argument("--image-wrist", default=None, help="wrist view (camera2)")
    ap.add_argument("--synthetic", action="store_true",
                    help="use the deterministic synthetic observation instead of files")
    ap.add_argument("--state", nargs=6, type=float, required=True,
                    help="6 joints in DEGREES (5 arm + gripper 0-100)")
    ap.add_argument("--task", default="put the yellow block into the basket")
    ap.add_argument("--chunks", type=int, default=30)
    ap.add_argument("--warmup", type=int, default=3)
    ap.add_argument("--num-steps", type=int, default=10)
    ap.add_argument("--n-action-steps", type=int, default=None)
    ap.add_argument("--seed", type=int, default=20261112)
    ap.add_argument("--cuda-graph", choices=["on", "off"], default="on")
    ap.add_argument("--patch-embedding", action="store_true")
    ap.add_argument("--max-crops", type=int, default=None)
    ap.add_argument("--run-name", default="run", help="label for this run directory")
    ap.add_argument("--run-root", default=str(HERE / "tmp" / "runs"),
                    help="parent directory; a timestamped subdirectory is created")
    ap.add_argument("--out", default=None,
                    help="explicit report path; overrides the timestamped run directory")
    ap.add_argument("--dry-run", action="store_true", help="stages 1-2 only")
    args = ap.parse_args()

    # Every run gets its own timestamped directory so reports are never
    # overwritten and each experiment stays reproducible.
    stamp = time.strftime("%Y%m%d_%H%M%S")
    run_dir = Path(args.run_root) / f"{stamp}_{args.run_name}"
    run_dir.mkdir(parents=True, exist_ok=True)
    out_path = Path(args.out) if args.out else run_dir / "report.json"

    report: dict = {
        "created_at": now_iso(),
        "argv": sys.argv,
        "run_dir": str(run_dir),
        "mode": "offline_only_no_robot_no_camera_no_serial",
        "stages": {},
        "status": "started",
    }

    def flush():
        report["saved_to"] = str(write_json(out_path, report))

    # ---------------------------------------------------------------- stage 1
    t_start = time.perf_counter()
    import torch

    sys.path.insert(0, str(HERE))
    import molmoact2_nf4 as m

    # The allocator must be configured before anything initialises CUDA, so it
    # is applied before the first device query below.
    allocator_report = m.configure_cuda_allocator()

    stage1 = {
        "python": sys.version,
        "executable": sys.executable,
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "cuda_available": torch.cuda.is_available(),
        "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "capability": list(torch.cuda.get_device_capability(0))
        if torch.cuda.is_available()
        else None,
        "host_memory": host_memory(),
        "nvidia_smi": smi("name,memory.total,memory.used,memory.free,driver_version"),
        "allocator": allocator_report,
        "cuda_mem_before": cuda_mem(),
    }
    import transformers

    stage1["transformers"] = transformers.__version__
    import bitsandbytes as bnb

    stage1["bitsandbytes"] = bnb.__version__
    import accelerate

    stage1["accelerate"] = accelerate.__version__
    import safetensors

    stage1["safetensors"] = safetensors.__version__
    try:
        import torchvision

        stage1["torchvision_import"] = getattr(torchvision, "__version__", "ok")
        stage1["torchvision_transforms_import"] = bool(
            __import__("torchvision.transforms", fromlist=["transforms"])
        )
    except Exception as e:  # noqa: BLE001
        stage1["torchvision_error"] = f"{type(e).__name__}: {e}"
    report["stages"]["environment"] = stage1
    flush()

    model_dir = Path(args.model_dir)

    # ---------------------------------------------------------------- stage 2
    try:
        if args.synthetic:
            images, image_meta = synthetic_images()
        else:
            if not (args.image_top and args.image_wrist):
                raise SystemExit("--image-top/--image-wrist required unless --synthetic")
            images, image_meta = load_images(args.image_top, args.image_wrist)
        processor = m.load_processor(model_dir)
        if args.max_crops is not None:
            processor.image_processor.max_crops = int(args.max_crops)
        import numpy as np

        state = np.asarray([float(x) for x in args.state], dtype=np.float32)
        model_mod = m.remote_module("modeling_molmoact2", model_dir)
        prompt = build_prompt(model_mod, model_dir, state, args.task, len(images))
        proc = processor(text=prompt, images=images, return_tensors="pt")
        report["stages"]["processor"] = {
            "observation_source": "synthetic" if args.synthetic else "image_files",
            "images": image_meta,
            "max_crops": int(processor.image_processor.max_crops),
            "prompt_text": prompt,
            "prompt_chars": len(prompt),
            "input_ids_shape": list(proc["input_ids"].shape),
            "pixel_values_shape": list(proc["pixel_values"].shape),
            "pixel_values_dtype": str(proc["pixel_values"].dtype),
            "image_num_crops": proc["image_num_crops"].tolist()
            if "image_num_crops" in proc else None,
            "image_grids": proc["image_grids"].tolist() if "image_grids" in proc else None,
            "state_degrees": state.tolist(),
            "norm_tag": NORM_TAG,
        }
        flush()
    except SystemExit:
        raise
    except Exception:
        report["status"] = "failed_at_processor"
        report["stages"]["processor_error"] = traceback.format_exc()
        flush()
        raise

    if args.dry_run:
        report["status"] = "dry_run_only"
        flush()
        print(json.dumps(report["stages"]["processor"], indent=2))
        return 0

    if args.patch_embedding:
        sys.path.insert(0, str(HERE / "patches"))
        import embedding_no_cat

        report["stages"]["embedding_patch"] = embedding_no_cat.install(model_mod)
        flush()

    # ---------------------------------------------------------------- stage 3
    stage3 = {}
    try:
        torch.cuda.reset_peak_memory_stats()
        torch.set_grad_enabled(False)
        t0 = time.perf_counter()
        model, load_report = m.load_model(
            model_dir, device="cuda", dtype=torch.bfloat16, verbose=True
        )
        torch.cuda.synchronize()
        stage3["load_seconds"] = round(time.perf_counter() - t0, 3)
        stage3["load_report"] = load_report
        stage3["cuda_mem_after_load"] = cuda_mem()
        stage3["host_memory_after_load"] = host_memory()
        stage3["quant_state_roundtrip"] = m.quant_state_roundtrip_check(model, model_dir)
        stage3["verify_nf4"] = m.verify_nf4_vs_raw_bytes(model, model_dir)
        report["stages"]["load"] = stage3
        flush()
    except Exception:
        stage3["error"] = traceback.format_exc()
        stage3["cuda_mem_at_failure"] = cuda_mem()
        report["status"] = "failed_at_load"
        report["stages"]["load"] = stage3
        flush()
        raise

    # instrumentation: does the continuous path actually run lm_head / wte?
    counters = [
        CallCounter(model.lm_head, "lm_head"),
        CallCounter(model.model.transformer.wte, "wte(MolmoAct2Embedding)"),
        CallCounter(model.model.vision_backbone, "vision_backbone"),
    ]

    graph_on = args.cuda_graph == "on"
    collected: list[tuple[str, "object"]] = []

    def run_once(tag: str) -> dict:
        gen = torch.Generator(device="cuda").manual_seed(args.seed)
        entry: dict = {"tag": tag}
        torch.cuda.synchronize()
        ev0 = torch.cuda.Event(enable_timing=True)
        ev1 = torch.cuda.Event(enable_timing=True)
        t0 = time.perf_counter()
        ev0.record()
        out = model.predict_action(
            processor=processor,
            images=images,
            task=args.task,
            state=state,
            norm_tag=NORM_TAG,
            inference_action_mode="continuous",
            enable_depth_reasoning=False,
            num_steps=args.num_steps,
            n_action_steps=args.n_action_steps,
            generator=gen,
            enable_cuda_graph=graph_on,
        )
        ev1.record()
        ev1.synchronize()
        wall = time.perf_counter() - t0
        gpu_ms = ev0.elapsed_time(ev1)
        actions = out.actions if hasattr(out, "actions") else out
        a = actions.detach().float().cpu()
        rows = a.reshape(-1, a.shape[-1])
        collected.append((tag, a.clone()))
        entry.update(
            {
                "wall_ms": round(wall * 1000, 2),
                "cuda_event_ms": round(gpu_ms, 2),
                "actions_shape": list(a.shape),
                "actions_dtype": str(actions.dtype),
                "action_rows": int(rows.shape[0]),
                "action_dim": int(rows.shape[1]),
                "all_finite": bool(torch.isfinite(a).all()),
                "min": round(float(a.min()), 3),
                "max": round(float(a.max()), 3),
                "first_action_row": [round(float(v), 3) for v in rows[0]],
                "last_action_row": [round(float(v), 3) for v in rows[-1]],
                "action_checksum": round(float(a.double().sum().item()), 6),
            }
        )
        del out, actions, a
        return entry

    # ---------------------------------------------------------------- stage 4
    stage4 = {}
    try:
        torch.cuda.reset_peak_memory_stats()
        for c in counters:
            c.reset()
        first = run_once("first_prediction")
        first["module_calls"] = [c.report() for c in counters]
        stage4["first_prediction"] = first
        stage4["cuda_graph_state"] = graph_state(model)
        stage4["cuda_mem_after_first"] = cuda_mem()
        stage4["host_memory_after_first"] = host_memory()
        report["stages"]["first_prediction"] = stage4
        flush()
    except Exception:
        stage4["error"] = traceback.format_exc()
        stage4["cuda_mem_at_failure"] = cuda_mem()
        report["status"] = "failed_at_first_prediction"
        report["stages"]["first_prediction"] = stage4
        flush()
        raise

    # ---------------------------------------------------------------- stage 5/6
    stage6 = {
        "chunks": args.chunks,
        "warmup_chunks": args.warmup,
        "seed": args.seed,
        "num_steps": args.num_steps,
        "n_action_steps": args.n_action_steps,
        "cuda_graph": graph_on,
        "embedding_patch": bool(args.patch_embedding),
        "max_crops": int(processor.image_processor.max_crops),
        "input": {
            "observation_source": "synthetic" if args.synthetic else "image_files",
            "images": image_meta,
            "state_degrees": state.tolist(),
            "task": args.task,
            "fixed_observation": "same observation for every chunk (throughput test)",
        },
        "timings": [],
    }
    try:
        for i in range(args.warmup):
            stage6.setdefault("warmup", []).append(run_once(f"warmup_{i}"))
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        stage6["cuda_mem_before_measured"] = cuda_mem()
        for i in range(args.chunks):
            stage6["timings"].append(run_once(f"chunk_{i:02d}"))
        stage6["cuda_mem_after_measured"] = cuda_mem()
        stage6["host_memory_after_measured"] = host_memory()
        walls = [t["wall_ms"] for t in stage6["timings"]]
        gpus = [t["cuda_event_ms"] for t in stage6["timings"]]

        def pct(values, q):
            s = sorted(values)
            return round(s[max(0, min(len(s) - 1, math.ceil(q * len(s)) - 1))], 2)

        stage6["summary"] = {
            "wall_ms": {
                "median": round(statistics.median(walls), 2),
                "mean": round(statistics.fmean(walls), 2),
                "p95": pct(walls, 0.95),
                "max": round(max(walls), 2),
                "min": round(min(walls), 2),
            },
            "cuda_event_ms": {
                "median": round(statistics.median(gpus), 2),
                "mean": round(statistics.fmean(gpus), 2),
                "p95": pct(gpus, 0.95),
                "max": round(max(gpus), 2),
                "min": round(min(gpus), 2),
            },
            "all_finite": all(t["all_finite"] for t in stage6["timings"]),
            "shapes": sorted({str(t["actions_shape"]) for t in stage6["timings"]}),
            "oom": False,
        }
        report["stages"]["continuous"] = stage6
        report["status"] = "completed"
    except Exception:
        stage6["error"] = traceback.format_exc()
        stage6["cuda_mem_at_failure"] = cuda_mem()
        report["status"] = "failed_during_continuous"
        report["stages"]["continuous"] = stage6
        flush()
        raise
    finally:
        report["total_seconds"] = round(time.perf_counter() - t_start, 1)
        report["stages"].setdefault("artifacts", {})
        if collected:
            try:
                import numpy as np

                npz_path = run_dir / "actions.npz"
                np.savez_compressed(
                    npz_path,
                    **{tag: arr.numpy().astype("float32") for tag, arr in collected},
                )
                report["stages"]["artifacts"]["actions_npz"] = str(npz_path)
                report["stages"]["artifacts"]["actions_arrays"] = len(collected)
            except Exception as e:  # noqa: BLE001
                report["stages"]["artifacts"]["actions_npz_error"] = (
                    f"{type(e).__name__}: {e}"
                )
        report["stages"]["cuda_graph_state_final"] = graph_state(model)
        flush()

    for c in counters:
        c.remove()

    s = report["stages"]["continuous"]["summary"]
    print("\n=== result ===")
    print(f"status            : {report['status']}")
    print(f"load              : {report['stages']['load']['load_seconds']} s")
    fp = report["stages"]["first_prediction"]["first_prediction"]
    print(f"first prediction  : {fp['wall_ms']} ms  shape={fp['actions_shape']}  finite={fp['all_finite']}")
    for c in fp.get("module_calls", []):
        print(f"  calls {c['module']:28s} {c['calls']:4d}  shapes={c['output_shapes']}")
    print(f"chunks            : {args.chunks}")
    print(f"median wall       : {s['wall_ms']['median']} ms  p95={s['wall_ms']['p95']} ms  max={s['wall_ms']['max']} ms")
    print(f"median cuda-event : {s['cuda_event_ms']['median']} ms  p95={s['cuda_event_ms']['p95']} ms")
    print(f"peak allocated    : {report['stages']['continuous']['cuda_mem_after_measured']['peak_allocated_GiB']} GiB (stage maximum)")
    print(f"peak reserved     : {report['stages']['continuous']['cuda_mem_after_measured']['peak_reserved_GiB']} GiB (stage maximum)")
    print(f"free at end       : {report['stages']['continuous']['cuda_mem_after_measured']['free_GiB']} GiB (point in time)")
    print(f"run dir           : {run_dir}")
    print(f"report            : {report['saved_to']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
