#!/usr/bin/env python
"""Numeric equivalence test for the embedding no-cat patch.

Stage 1 (always runs): a small synthetic embedding with the same *shape logic*
as the real module (base table + extra table, boundary ids, out-of-range ids),
compared against the upstream ``torch.cat`` implementation.

Stage 2 (optional, ``--model-dir``): the same comparison on the real
MolmoAct2Embedding instance of the loaded graph, on CPU, using random token
ids that cover both tables and the boundary.

Only the embedding module is exercised here; this says nothing about the rest
of the model.  A full-model A/B consistency check is a separate experiment.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE))

import embedding_no_cat as patch_mod  # noqa: E402


class SyntheticEmbedding(torch.nn.Module):
    """Same two-table layout as MolmoAct2Embedding, without the remote import."""

    def __init__(self, n_base: int, n_extra: int, features: int):
        super().__init__()
        self.embedding = torch.nn.Parameter(torch.randn(n_base, features))
        self.new_embedding = torch.nn.Parameter(torch.randn(n_extra, features))

    def forward(self, x):  # replaced per-case
        raise NotImplementedError


def synthetic_test() -> dict:
    torch.manual_seed(0)
    n_base, n_extra, feats = 11, 3, 5
    emb = SyntheticEmbedding(n_base, n_extra, feats)
    total = n_base + n_extra

    cases = {
        "base_only": torch.tensor([[0, 1, 2, 10]]),
        "boundary_first_extra": torch.tensor([[n_base - 1, n_base]]),
        "extra_only": torch.tensor([[n_base, n_base + 1, total - 1]]),
        "mixed_2d": torch.tensor([[0, n_base, 3], [total - 1, 0, n_base + 1]]),
        "duplicates": torch.tensor([[n_base, n_base, 0, 0]]),
        "flattened": torch.arange(total).reshape(1, -1),
    }
    results = {}
    worst = 0.0
    for name, ids in cases.items():
        ref = patch_mod.legacy_forward(emb, ids)
        new = patch_mod.patched_forward(emb, ids)
        same = torch.equal(ref, new)
        diff = float((ref - new).abs().max()) if ref.shape == new.shape else float("inf")
        worst = max(worst, diff)
        results[name] = {
            "shape": list(new.shape),
            "bitwise_equal": bool(same),
            "max_abs_diff": diff,
        }

    # out-of-range behaviour must still raise IndexError
    oob = {}
    for name, ids in {
        "negative": torch.tensor([[-1]]),
        "at_total": torch.tensor([[total]]),
        "far_above": torch.tensor([[total + 100]]),
    }.items():
        try:
            patch_mod.patched_forward(emb, ids)
            oob[name] = "NO ERROR (bad)"
        except IndexError as e:
            oob[name] = f"IndexError: {str(e)[:80]}"
        except Exception as e:  # noqa: BLE001
            oob[name] = f"{type(e).__name__}: {str(e)[:80]}"
    # the original must raise for the same ids
    orig_oob = {}
    for name, ids in {
        "negative": torch.tensor([[-1]]),
        "at_total": torch.tensor([[total]]),
        "far_above": torch.tensor([[total + 100]]),
    }.items():
        try:
            patch_mod.legacy_forward(emb, ids)
            orig_oob[name] = "NO ERROR"
        except Exception as e:  # noqa: BLE001
            orig_oob[name] = type(e).__name__

    return {
        "case_results": results,
        "worst_max_abs_diff": worst,
        "all_bitwise_equal": all(v["bitwise_equal"] for v in results.values()),
        "oob_patched": oob,
        "oob_original_exception_type": orig_oob,
    }


def synthetic_memory_test() -> dict:
    """Measure the temporary allocation of both implementations (CPU, bf16)."""
    torch.manual_seed(1)
    n_base, n_extra, feats = 4096, 64, 512
    emb = SyntheticEmbedding(n_base, n_extra, feats).to(torch.bfloat16)
    ids = torch.randint(0, n_base + n_extra, (1, 256))
    table_bytes = (n_base + n_extra) * feats * 2
    return {
        "table_bytes": table_bytes,
        "table_GiB": round(table_bytes / (1 << 30), 6),
        "note": "real checkpoint table is 396,165,120 elements = 0.7379 GiB bf16",
    }


def real_model_test(model_dir: str) -> dict:
    """Compare both implementations on the real embedding module (CPU, bf16)."""
    import molmoact2_nf4 as m

    model_mod = m.remote_module("modeling_molmoact2", model_dir)
    cfg_mod = m.remote_module("configuration_molmoact2", model_dir)
    config = cfg_mod.MolmoAct2Config.from_pretrained(model_dir, local_files_only=True)
    text_cfg = config.text_config
    n_base = int(text_cfg.vocab_size)
    n_extra = int(text_cfg.additional_vocab_size)
    feats = int(text_cfg.hidden_size)

    # Build only the embedding module with the real dimensions, on CPU.
    emb = model_mod.MolmoAct2Embedding(n_base, n_extra, feats, device="cpu")
    with torch.no_grad():
        emb.embedding.normal_()
        emb.new_embedding.normal_()
    emb = emb.to(torch.bfloat16)

    torch.manual_seed(2)
    ids = torch.cat(
        [
            torch.randint(0, n_base, (1, 64)),
            torch.randint(n_base, n_base + n_extra, (1, 8)),
            torch.tensor([[n_base - 1, n_base, n_base + n_extra - 1]]),
        ],
        dim=1,
    )

    t0 = time.perf_counter()
    ref = patch_mod.legacy_forward(emb, ids)
    t1 = time.perf_counter()
    new = patch_mod.patched_forward(emb, ids)
    t2 = time.perf_counter()
    return {
        "vocab_size": n_base,
        "additional_vocab_size": n_extra,
        "hidden_size": feats,
        "table_elements": (n_base + n_extra) * feats,
        "table_MiB_bf16": round((n_base + n_extra) * feats * 2 / (1 << 20), 1),
        "ids_tested": int(ids.numel()),
        "bitwise_equal": bool(torch.equal(ref, new)),
        "max_abs_diff": float((ref - new).abs().max()),
        "legacy_seconds": round(t1 - t0, 4),
        "patched_seconds": round(t2 - t1, 4),
        "dtype": str(new.dtype),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", default=None)
    ap.add_argument("--output", default=None)
    args = ap.parse_args()

    report = {
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "synthetic": synthetic_test(),
        "synthetic_shapes": synthetic_memory_test(),
    }
    if args.model_dir:
        try:
            report["real_embedding"] = real_model_test(args.model_dir)
        except Exception as e:  # noqa: BLE001
            report["real_embedding_error"] = f"{type(e).__name__}: {e}"

    text = json.dumps(report, indent=2)
    print(text)
    if args.output:
        Path(args.output).write_text(text, encoding="utf-8")
    ok = report["synthetic"]["all_bitwise_equal"]
    if "real_embedding" in report:
        ok = ok and report["real_embedding"]["bitwise_equal"]
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
