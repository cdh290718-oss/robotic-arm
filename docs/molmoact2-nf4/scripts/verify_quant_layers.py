#!/usr/bin/env python
"""Stage A: small-scale NF4 load/decode verification, without building the model.

For a few small quantized layers taken from the real checkpoint file it:

  1. calibrates the bit layout on a synthetic tensor that bitsandbytes itself
     quantized (high-nibble-first + offset sign), so the independent decoder is
     proven before it is used on real weights,
  2. installs each layer's packed payload as a ``Params4bit`` with a
     ``QuantState`` built from the file (the same code path the full loader
     uses),
  3. decodes the first rows by hand and compares them with
     ``bnb.dequantize_4bit`` under a bf16-rounding tolerance,
  4. compares ``bnb.matmul_4bit`` with a dense matmul on the dequantized
     weights.

An empty layer selection is treated as a failure.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import molmoact2_nf4 as m  # noqa: E402

FAMILIES = {
    "text_attn": "model.transformer.blocks.",
    "vision_ff": "model.vision_backbone.image_vit.transformer.resblocks.",
    "action_expert": "model.action_expert.",
    "image_projector": "model.vision_backbone.image_projector.",
}


def calibrate() -> dict:
    """Determine the nibble order and offset sign from bnb's own round-trip."""
    import bitsandbytes as bnb

    torch.manual_seed(0)
    x = (torch.randn(8, 128, device="cuda") * 0.7).to(torch.bfloat16)
    packed, qs = bnb.functional.quantize_4bit(
        x, blocksize=64, compress_statistics=True, quant_type="nf4"
    )
    ref = bnb.functional.dequantize_4bit(packed, qs).float().cpu().numpy().reshape(-1)
    qmap = qs.code.detach().float().cpu().numpy().reshape(-1)
    codes = qs.absmax.detach().cpu().numpy().reshape(-1).astype(np.int64)
    n_absmax = qs.state2.absmax.detach().float().cpu().numpy().reshape(-1)
    n_code = qs.state2.code.detach().float().cpu().numpy().reshape(-1)
    n_bs = int(qs.state2.blocksize)
    offset = float(qs.offset.detach().item())
    packed_np = packed.cpu().numpy().reshape(-1)
    results = []
    for order in ("high_first", "low_first"):
        for sign in (+1.0, -1.0):
            scales = n_code[codes] * n_absmax[np.arange(codes.size) // n_bs] + sign * offset
            nib = np.empty(packed_np.size * 2, dtype=np.uint8)
            if order == "high_first":
                nib[0::2] = (packed_np >> 4) & 0x0F
                nib[1::2] = packed_np & 0x0F
            else:
                nib[0::2] = packed_np & 0x0F
                nib[1::2] = (packed_np >> 4) & 0x0F
            vals = (qmap[nib.reshape(-1, int(qs.blocksize))] * scales[:, None]).reshape(-1)
            results.append(
                {"order": order, "offset_sign": sign, "max_abs_err": float(np.max(np.abs(vals - ref)))}
            )
    best = min(results, key=lambda r: r["max_abs_err"])
    tol = max(
        2.0 * float(torch.finfo(torch.bfloat16).eps) * float(np.max(np.abs(ref))), 1e-6
    )
    if best["max_abs_err"] > tol:
        raise RuntimeError(f"calibration failed: {json.dumps(results)}")
    return {
        "candidates": results,
        "chosen": best,
        "tolerance": tol,
        "quant_state_api": {
            "blocksize": int(qs.blocksize),
            "nested_blocksize": n_bs,
            "has_state2": qs.state2 is not None,
            "offset": offset,
            "code_len": int(qmap.size),
        },
    }


def pick_layers(header: dict, per_family: int, max_numel: int) -> list[tuple[str, str]]:
    from bitsandbytes.functional import QuantState

    names = m.quantized_module_names(header)
    # element count comes from the packed payload, 2 values per byte
    size = {
        n: int(header[n + ".weight"]["shape"][0]) * 2
        for n in names
        if header[n + ".weight"]["shape"][0] > 0
    }
    picked = []
    for family, prefix in FAMILIES.items():
        cands = [
            (size[n], n)
            for n in names
            if n.startswith(prefix) and size[n] <= max_numel
        ]
        for numel, n in sorted(cands)[:per_family]:
            picked.append((family, n))
    if not picked:
        raise RuntimeError("no quantized layer matched the selection filters")
    return picked


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", default=str(HERE / "model"))
    ap.add_argument("--per-family", type=int, default=2)
    ap.add_argument("--max-numel", type=int, default=12_000_000)
    ap.add_argument("--out", default=str(HERE / "tmp" / "stage_a_quant_layers.json"))
    args = ap.parse_args()

    import bitsandbytes as bnb
    from bitsandbytes.functional import QuantState
    from safetensors import safe_open

    report: dict = {
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "model_dir": str(args.model_dir),
        "bitsandbytes": bnb.__version__,
        "torch": torch.__version__,
        "gpu": torch.cuda.get_device_name(0),
        "status": "started",
    }
    try:
        report["calibration"] = calibrate()
        shard = Path(args.model_dir) / "model.safetensors"
        header = m.read_safetensors_header(shard)
        picked = pick_layers(header, args.per_family, args.max_numel)
        report["selected_layers"] = [n for _, n in picked]
        eps = float(torch.finfo(torch.bfloat16).eps)
        checked = []
        with safe_open(str(shard), framework="pt", device="cuda") as f:
            for family, name in picked:
                packed = f.get_tensor(name + ".weight")
                qs_dict = {
                    k[len(name) + 1 :]: f.get_tensor(k)
                    for k in header
                    if k.startswith(name + ".weight.")
                }
                qs = QuantState.from_dict(qs_dict, device="cuda")
                weight = bnb.nn.Params4bit(
                    packed,
                    requires_grad=False,
                    quant_state=qs,
                    blocksize=int(qs.blocksize),
                    compress_statistics=True,
                    quant_type="nf4",
                    quant_storage=torch.uint8,
                )
                weight.bnb_quantized = True
                rows = min(8, int(qs.shape[0]))
                cols = int(qs.shape[1])
                manual = m._dequant_4bit_by_hand(packed, qs, rows, cols)
                ref = (
                    bnb.functional.dequantize_4bit(weight.data, qs)
                    .float()
                    .cpu()
                    .numpy()
                    .reshape(int(qs.shape[0]), cols)[:rows]
                )
                diff = float(np.max(np.abs(manual - ref)))
                tol = max(2.0 * eps * float(np.max(np.abs(ref))), 1e-6)
                x = torch.randn(4, cols, device="cuda", dtype=torch.bfloat16) * 0.5
                y4 = bnb.matmul_4bit(x, weight.t(), quant_state=qs)
                yref = torch.nn.functional.linear(
                    x, bnb.functional.dequantize_4bit(weight.data, qs).to(torch.bfloat16)
                )
                mdiff = float((y4.float() - yref.float()).abs().max())
                mtol = max(8.0 * eps * float(yref.float().abs().max().item()), 1e-3)
                entry = {
                    "family": family,
                    "module": name,
                    "shape": [int(qs.shape[0]), cols],
                    "numel": int(qs.shape[0]) * cols,
                    "rows_checked": rows,
                    "decode_max_abs_diff": diff,
                    "decode_tolerance": tol,
                    "decode_ok": bool(diff <= tol),
                    "matmul_max_abs_diff": mdiff,
                    "matmul_tolerance": mtol,
                    "matmul_ok": bool(mdiff <= mtol),
                    "blocksize": int(qs.blocksize),
                    "nested_blocksize": int(qs.state2.blocksize),
                    "offset": float(qs.offset.detach().item()),
                    "storage_dtype": str(weight.dtype),
                    "output_dtype": str(qs.dtype),
                    "packed_bytes": int(weight.numel()),
                    "dense_bf16_bytes": int(qs.shape[0]) * cols * 2,
                }
                checked.append(entry)
                print(
                    f"  {family:14s} {name[:60]:60s} decode_err={diff:.3e} "
                    f"(tol {tol:.3e}) matmul_err={mdiff:.3e} (tol {mtol:.3e})",
                    flush=True,
                )
                del packed, qs, weight, x, y4, yref, manual, ref
                torch.cuda.empty_cache()
        bad = [c for c in checked if not (c["decode_ok"] and c["matmul_ok"])]
        report["checked"] = checked
        report["n_checked"] = len(checked)
        report["families"] = sorted({c["family"] for c in checked})
        report["status"] = "failed" if bad else "passed"
        if bad:
            raise RuntimeError(
                f"{len(bad)} layer(s) failed verification: {[b['module'] for b in bad][:3]}"
            )
    except Exception:
        import traceback

        report["status"] = "failed"
        report["error"] = traceback.format_exc()
        raise
    finally:
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(f"report: {out}  status={report['status']}  layers={report.get('n_checked')}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
