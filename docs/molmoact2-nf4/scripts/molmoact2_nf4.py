#!/usr/bin/env python
"""Independent loader for the pinned OpenRAL MolmoAct2 SO-100/101 NF4 checkpoint.

Why this file exists
--------------------
The Hub repo ships pre-quantized bitsandbytes NF4 weights *and* the upstream
``modeling_molmoact2.py`` custom code, but no loader: OpenRAL's own
``install_prequantized_linears`` path is part of the closed OpenRAL Pro
package.  This module re-implements the loader from the public artifacts only:

  * the graph is built from the repository's own ``configuration_molmoact2.py``
    / ``modeling_molmoact2.py`` (no ``trust_remote_code`` download at runtime,
    no OpenRAL code),
  * NF4 layers are restored with public ``bitsandbytes`` primitives,
  * the checkpoint's own ``quantization_metadata.json`` rule is applied and
    *verified against the file* instead of being trusted.

Memory discipline (8 GiB card)
------------------------------
* the whole graph is built on the ``meta`` device, so no fp32/bf16 weight is
  ever materialised on the GPU (or in host RAM) before its final tensor is
  known,
* quantized weights are installed as packed ``uint8`` -> ``Params4bit``; the
  integer payload is *never* treated as a float tensor and never re-quantized,
* remaining weights are streamed one tensor at a time from the safetensors
  file straight to the GPU and assigned (no CPU staging copy of the model),
* every file key must be consumed and every model tensor must be filled; the
  only tolerated "missing" keys are the quantized weights this module installs
  itself, and that set is compared for exact equality.
"""

from __future__ import annotations

import collections
import importlib.util
import json
import os
import struct
import sys
import types
from pathlib import Path

# ---------------------------------------------------------------- CUDA allocator

def configure_cuda_allocator() -> dict:
    """Set the expandable-segments allocator *before* the first CUDA alloc.

    torch >= 2.9 reads ``PYTORCH_ALLOC_CONF``; older builds read
    ``PYTORCH_CUDA_ALLOC_CONF``.  We set both and report whether the process
    already had a CUDA context, because that decides whether the setting can
    still take effect.
    """
    report = {
        "env_before": {
            k: os.environ.get(k)
            for k in ("PYTORCH_ALLOC_CONF", "PYTORCH_CUDA_ALLOC_CONF")
        },
        "cuda_initialized_before": None,
        "applied": False,
    }
    import torch

    report["torch_version"] = torch.__version__
    report["cuda_available"] = torch.cuda.is_available()
    report["cuda_initialized_before"] = bool(
        torch.cuda.is_available() and torch.cuda.is_initialized()
    )
    value = "expandable_segments:True"
    os.environ["PYTORCH_ALLOC_CONF"] = value
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = value
    try:
        from torch.cuda.memory import _set_allocator_settings

        _set_allocator_settings("expandable_segments:True")
        report["applied"] = True
    except Exception as e:  # noqa: BLE001
        report["set_allocator_settings_error"] = f"{type(e).__name__}: {e}"
    report["env_after"] = {
        k: os.environ.get(k) for k in ("PYTORCH_ALLOC_CONF", "PYTORCH_CUDA_ALLOC_CONF")
    }
    # Whether the setting has any effect is a platform property: probe it with a
    # tiny allocation and keep whatever torch says.  Never claim it is active
    # just because the variable was set.
    import warnings

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        try:
            torch.zeros(1, device="cuda")
        except Exception as e:  # noqa: BLE001
            report["allocator_probe_error"] = f"{type(e).__name__}: {e}"
        report["allocator_probe_warnings"] = [
            str(w.message) for w in caught if "allocator" in str(w.message).lower()
            or "expandable" in str(w.message).lower()
        ]
    msgs = report["allocator_probe_warnings"]
    report["expandable_segments_effective"] = not any(
        "not supported on this platform" in m_ for m_ in msgs
    )
    if not report["expandable_segments_effective"]:
        report["note"] = (
            "torch reports expandable_segments is not supported on this platform; "
            "the setting is recorded as requested but NOT active"
        )
    return report


# ------------------------------------------------------------------ remote code

def import_remote_package(model_dir: str | Path, package: str = "molmoact2_remote"):
    """Import the repo's own modules as a package, without trust_remote_code.

    The files use relative imports (``from .configuration_molmoact2 import ...``),
    so they must be imported as a package rooted at the model directory.  No
    code is fetched at runtime and nothing is written to the HF module cache.

    ``importlib.import_module`` is used (not a hand-built spec) so that the
    normal ``sys.modules`` cache applies: executing a submodule twice would
    create two distinct class objects (e.g. two ``ActionCudaGraphManager``
    classes), and patching one of them would silently miss the one the model
    actually uses.
    """
    model_dir = Path(model_dir).resolve()
    if package in sys.modules:
        return sys.modules[package]
    pkg = types.ModuleType(package)
    pkg.__path__ = [str(model_dir)]
    pkg.__package__ = package
    sys.modules[package] = pkg
    for name in (
        "configuration_molmoact2",
        "modeling_molmoact2",
        "image_processing_molmoact2",
        "video_processing_molmoact2",
        "processing_molmoact2",
        "inference",
    ):
        if (model_dir / f"{name}.py").exists():
            importlib.import_module(f"{package}.{name}")
    return sys.modules[package]


def remote_module(name: str, model_dir: str | Path, package: str = "molmoact2_remote"):
    import_remote_package(model_dir, package)
    return sys.modules[f"{package}.{name}"]


# ----------------------------------------------------------------- file reading

def read_safetensors_header(path: str | Path) -> dict:
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        header = json.loads(f.read(n))
    header.pop("__metadata__", None)
    return header


QUANT_SUFFIX = ".weight.quant_state.bitsandbytes__nf4"
QUANT_AUX_SUFFIXES = (
    ".weight.absmax",
    ".weight.quant_map",
    ".weight.nested_absmax",
    ".weight.nested_quant_map",
)


def quantized_module_names(keys) -> list[str]:
    return sorted(k[: -len(QUANT_SUFFIX)] for k in keys if k.endswith(QUANT_SUFFIX))


def is_quant_aux_key(key: str) -> bool:
    return any(key.endswith(s) for s in QUANT_AUX_SUFFIXES) or key.endswith(QUANT_SUFFIX)


# ----------------------------------------------------------------------- model

def load_config(model_dir: str | Path):
    import torch

    cfg_mod = remote_module("configuration_molmoact2", model_dir)
    config = cfg_mod.MolmoAct2Config.from_pretrained(
        str(model_dir), local_files_only=True
    )
    # config.json carries "dtype": "float32" from the conversion; the actual
    # non-quantized tensors in the file are bf16.  Never let the config default
    # create fp32 weights.
    config.dtype = torch.bfloat16
    config._attn_implementation = "sdpa"
    # We build the graph directly instead of through ``from_pretrained``, so the
    # source directory must be recorded here: ``_get_robot_stats`` resolves
    # norm_stats.json from ``config._name_or_path``.
    config._name_or_path = str(Path(model_dir).resolve())
    return config


def build_meta_model(config, model_dir: str | Path):
    """Instantiate the full graph on the meta device (no storage allocated)."""
    import torch

    model_mod = remote_module("modeling_molmoact2", model_dir)
    original = model_mod.MolmoAct2PreTrainedModel._init_weights

    def _skip(self, module):  # noqa: ANN001
        return None

    model_mod.MolmoAct2PreTrainedModel._init_weights = _skip
    try:
        with torch.device("meta"):
            model = model_mod.MolmoAct2ForConditionalGeneration(config)
    finally:
        model_mod.MolmoAct2PreTrainedModel._init_weights = original
    return model


def _get_parent(model, dotted: str):
    parts = dotted.split(".")
    obj = model
    for p in parts[:-1]:
        obj = getattr(obj, p) if not p.isdigit() else obj[int(p)]
    return obj, parts[-1]


def _make_linear4bit(bnb, in_f, out_f, has_bias, dtype):
    """Create an empty Linear4bit without materialising a full-precision weight."""
    import torch

    kw = dict(
        bias=has_bias,
        compute_dtype=dtype,
        compress_statistics=True,
        quant_type="nf4",
        quant_storage=torch.uint8,
    )
    try:
        return bnb.nn.Linear4bit(in_f, out_f, device="meta", **kw), "meta"
    except Exception:  # noqa: BLE001 - some bnb versions reject meta device
        return bnb.nn.Linear4bit(in_f, out_f, device="cpu", **kw), "cpu"


def install_prequantized(
    model,
    model_dir: str | Path,
    device: str = "cuda",
    dtype=None,
    min_numel: int = 4_000_000,
    verbose: bool = True,
) -> dict:
    """Replace eligible Linear modules with bnb Linear4bit and load the file.

    Returns a machine-readable report; raises on any inconsistency instead of
    silently ignoring keys.
    """
    import torch
    from safetensors import safe_open

    if dtype is None:
        dtype = torch.bfloat16
    model_dir = Path(model_dir)
    shard = model_dir / "model.safetensors"
    header = read_safetensors_header(shard)
    keys = list(header)

    import bitsandbytes as bnb
    from bitsandbytes.functional import QuantState

    qmodules = quantized_module_names(keys)
    qset = set(qmodules)
    aux_keys = [k for k in keys if is_quant_aux_key(k) and not k.endswith(QUANT_SUFFIX)]
    aux_keys += [k for k in keys if k.endswith(QUANT_SUFFIX)]

    report = {
        "safetensors": str(shard),
        "quantization_metadata": json.loads(
            (model_dir / "quantization_metadata.json").read_text(encoding="utf-8")
        )
        if (model_dir / "quantization_metadata.json").exists()
        else None,
        "file_keys": len(keys),
        "quantized_modules": len(qmodules),
        "quant_aux_keys": len(aux_keys),
        "min_numel_rule": min_numel,
        "installed": [],
        "kept_bf16_large_linears": [],
        "linear4bit_init_device": None,
        "errors": [],
        "bitsandbytes_version": getattr(bnb, "__version__", "unknown"),
        "compute_dtype": str(dtype),
    }

    # ---- 1. audit the rule against the *graph* before touching anything
    graph_keys_before = set(model.state_dict().keys())
    linear_sizes = {}
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Linear):
            linear_sizes[name] = tuple(module.weight.shape)

    not_linear = [n for n in qmodules if n not in linear_sizes]
    if not_linear:
        raise RuntimeError(
            "checkpoint quantizes modules that are not nn.Linear in this graph: "
            f"{not_linear[:5]}"
        )

    eligible = {n: s for n, s in linear_sizes.items() if s[0] * s[1] >= min_numel}
    missing_from_file = sorted(set(eligible) - qset)
    if missing_from_file:
        report["errors"].append(
            {"kind": "eligible_but_not_quantized", "modules": missing_from_file}
        )
        raise RuntimeError(
            "graph has Linear modules with >= %d weights that the checkpoint did "
            "not quantize: %s" % (min_numel, missing_from_file[:5])
        )
    for name in sorted(set(linear_sizes) - qset):
        size = linear_sizes[name]
        if size[0] * size[1] >= min_numel:
            report["kept_bf16_large_linears"].append(name)

    with safe_open(str(shard), framework="pt", device="cpu") as f:
        # ---- 2. install NF4 modules one at a time
        for name in qmodules:
            parent, leaf = _get_parent(model, name)
            module = getattr(parent, leaf) if not leaf.isdigit() else parent[int(leaf)]
            packed = f.get_tensor(f"{name}.weight")
            qs_dict = {
                k[len(name) + 1 :]: f.get_tensor(k)
                for k in keys
                if k.startswith(name + ".weight.")
            }
            n_elem = packed.numel() * 2
            out_f, in_f = module.weight.shape
            if n_elem != out_f * in_f:
                raise RuntimeError(
                    f"{name}: packed uint8 payload implies {n_elem} weights but the "
                    f"graph expects {out_f}x{in_f}={out_f * in_f}"
                )
            quant_state = QuantState.from_dict(qs_dict, device=device)
            if tuple(quant_state.shape) != (out_f, in_f):
                raise RuntimeError(
                    f"{name}: quant_state.shape={tuple(quant_state.shape)} does not "
                    f"match graph shape {(out_f, in_f)}"
                )
            new_module, init_dev = _make_linear4bit(
                bnb, in_f, out_f, module.bias is not None, dtype
            )
            report["linear4bit_init_device"] = report["linear4bit_init_device"] or init_dev
            weight = bnb.nn.Params4bit(
                packed.to(device),
                requires_grad=False,
                quant_state=quant_state,
                blocksize=int(quant_state.blocksize),
                compress_statistics=True,
                quant_type="nf4",
                quant_storage=torch.uint8,
            )
            weight.bnb_quantized = True
            new_module.weight = weight
            new_module.compute_dtype = dtype
            if module.bias is not None:
                new_module.bias = torch.nn.Parameter(
                    torch.empty_like(module.bias, device=device), requires_grad=False
                )
            setattr(parent, leaf, new_module)
            report["installed"].append(name)
            if verbose and len(report["installed"]) % 60 == 0:
                print(f"  installed NF4 linears: {len(report['installed'])}", flush=True)
            del packed, qs_dict

        # ---- 3. collect everything else (includes biases of replaced modules)
        installed_weight_keys = {f"{n}.weight" for n in qmodules}
        skip = set(installed_weight_keys) | set(aux_keys)
        sd = {}
        for key in keys:
            if key in skip:
                continue
            sd[key] = f.get_tensor(key).to(device)

    # Coverage is checked against the graph as it was *before* the NF4 modules
    # were installed: bnb's Params4bit adds the quant-state tensors to the
    # module's own state_dict, so a post-install key set would contain the very
    # keys we already consumed.
    unexpected_in_file = sorted(set(sd) - graph_keys_before)
    if unexpected_in_file:
        raise RuntimeError(
            f"checkpoint has tensors the graph does not declare: {unexpected_in_file[:5]}"
        )
    absent = sorted(graph_keys_before - set(sd))
    if set(absent) != installed_weight_keys:
        raise RuntimeError(
            "tensor coverage mismatch. Graph tensors with no source: "
            f"{sorted(set(absent) - installed_weight_keys)[:8]}; "
            f"pre-installed NF4 weights not declared by graph: "
            f"{sorted(installed_weight_keys - set(absent))[:8]}"
        )

    # ---- 4. assign tensor by tensor (no load_state_dict, no ignore flags)
    assigned = 0
    for key, tensor in sd.items():
        parent, leaf = _get_parent(model, key)
        current = getattr(parent, leaf)
        if tuple(current.shape) != tuple(tensor.shape):
            raise RuntimeError(
                f"{key}: source shape {tuple(tensor.shape)} != graph shape "
                f"{tuple(current.shape)}"
            )
        if isinstance(current, torch.nn.Parameter):
            setattr(parent, leaf, torch.nn.Parameter(tensor, requires_grad=False))
        else:
            parent._buffers[leaf] = tensor
        assigned += 1
    del sd
    import gc

    gc.collect()

    report["consumed_file_keys"] = len(keys) - len(skip)
    report["assigned_tensors"] = assigned
    report["installed_weight_keys"] = len(installed_weight_keys)
    report["unexpected_keys"] = []
    report["missing_keys_explained_count"] = 0
    meta_params = sorted(n for n, p in model.named_parameters() if p.is_meta)
    meta_buffers_persistent = []
    meta_buffers_runtime = []
    for n, b in model.named_buffers():
        if not b.is_meta:
            continue
        # Non-persistent buffers (e.g. the lazy RoPE cos/sin cache) are not
        # checkpoint tensors: the module declares them persistent=False and
        # rebuilds them from inv_freq on first use.  Only persistent buffers
        # must be filled from the file.
        (meta_buffers_persistent if n in graph_keys_before else meta_buffers_runtime).append(n)
    report["meta_tensors_left"] = meta_params + meta_buffers_persistent
    report["runtime_derived_buffers"] = meta_buffers_runtime
    if meta_params or meta_buffers_persistent:
        raise RuntimeError(
            f"persistent tensors still on meta after load: "
            f"{(meta_params + meta_buffers_persistent)[:8]}"
        )
    # every file tensor must be accounted for exactly once
    accounted = assigned + len(installed_weight_keys) + len(aux_keys)
    if accounted != len(keys):
        raise RuntimeError(
            f"key accounting mismatch: {accounted} != {len(keys)} file keys"
        )
    report["key_accounting"] = {
        "file_keys": len(keys),
        "assigned_dense_tensors": assigned,
        "packed_nf4_weights": len(installed_weight_keys),
        "quant_state_tensors_consumed_by_quantstate": len(aux_keys),
        "sum": accounted,
    }
    return report


def audit_model(model) -> dict:
    """Dtype/device census + proof that the NF4 path is really wired."""
    import torch
    import bitsandbytes as bnb

    dtypes = collections.Counter()
    bytes_by_dtype = collections.Counter()
    devices = collections.Counter()
    n_linear4bit = 0
    n_plain_linear = 0
    quant_state_devices = collections.Counter()
    quant_types = collections.Counter()
    for name, p in model.named_parameters():
        dtypes[str(p.dtype)] += p.numel()
        bytes_by_dtype[str(p.dtype)] += p.numel() * p.element_size()
        devices[str(p.device)] += 1
    for name, b in model.named_buffers():
        dtypes[str(b.dtype)] += b.numel()
        bytes_by_dtype[str(b.dtype)] += b.numel() * b.element_size()
        devices[str(b.device)] += 1
    for name, m in model.named_modules():
        if isinstance(m, bnb.nn.Linear4bit):
            n_linear4bit += 1
            if getattr(m.weight, "quant_state", None) is None:
                raise RuntimeError(f"{name}: Linear4bit without quant_state")
            quant_types[str(m.weight.quant_state.quant_type)] += 1
            quant_state_devices[str(m.weight.quant_state.absmax.device)] += 1
        elif isinstance(m, torch.nn.Linear):
            n_plain_linear += 1
    return {
        "param_dtype_elements": dict(dtypes),
        "param_dtype_bytes": {k: int(v) for k, v in bytes_by_dtype.items()},
        "param_dtype_GiB": {k: round(v / (1 << 30), 4) for k, v in bytes_by_dtype.items()},
        "devices": dict(devices),
        "linear4bit_modules": n_linear4bit,
        "plain_linear_modules": n_plain_linear,
        "quant_types": dict(quant_types),
        "quant_state_devices": dict(quant_state_devices),
        "cuda_allocated_GiB": round(torch.cuda.memory_allocated() / (1 << 30), 4)
        if torch.cuda.is_available()
        else None,
        "cuda_reserved_GiB": round(torch.cuda.memory_reserved() / (1 << 30), 4)
        if torch.cuda.is_available()
        else None,
    }


def _dequant_4bit_by_hand(packed: "object", qs, rows: int, cols: int):
    """Independent NF4 decode of the first ``rows`` rows of one packed layer.

    Layout (verified against bitsandbytes' own round-trip in
    ``verify_quant_layers.py``):

      * packed byte ``b`` holds element ``2k`` in its **high** nibble and
        element ``2k+1`` in its low nibble,
      * every ``qs.blocksize`` elements share one scale,
      * the scale is double-quantized: ``code[absmax_code] * nested_absmax[
        block_index // nested_blocksize] + offset``,
      * value = ``qs.code[nibble] * scale``.

    Only ``qs``'s public attributes are used (``code``, ``state2``,
    ``blocksize``, ``offset``), i.e. the current bitsandbytes API.
    """
    import numpy as np

    qmap = qs.code.detach().float().cpu().numpy().reshape(-1)
    codes = qs.absmax.detach().cpu().numpy().reshape(-1).astype(np.int64)
    offset = float(qs.offset.detach().item()) if qs.offset is not None else 0.0
    if qs.state2 is not None:
        nested_absmax = qs.state2.absmax.detach().float().cpu().numpy().reshape(-1)
        nested_code = qs.state2.code.detach().float().cpu().numpy().reshape(-1)
        nested_bs = int(qs.state2.blocksize)
        scales = (
            nested_code[codes] * nested_absmax[np.arange(codes.size) // nested_bs]
            + offset
        )
    else:
        scales = qs.absmax.detach().float().cpu().numpy().reshape(-1) + offset

    blocksize = int(qs.blocksize)
    packed_np = packed.detach().cpu().numpy().reshape(-1)
    n = rows * cols
    flat = packed_np[: n // 2]
    nibbles = np.empty(flat.size * 2, dtype=np.uint8)
    nibbles[0::2] = (flat >> 4) & 0x0F  # high nibble first
    nibbles[1::2] = flat & 0x0F
    blocks = nibbles.reshape(-1, blocksize)
    if blocks.shape[0] > scales.size:
        raise RuntimeError(
            f"decode needs {blocks.shape[0]} scales but the quant state has "
            f"{scales.size}"
        )
    return (qmap[blocks] * scales[: blocks.shape[0], None]).reshape(rows, cols)


def _pick_verification_layers(model, header, max_numel: int, per_family: int = 2):
    """Choose small quantized layers from several module families."""
    families = {
        "text_attn": "model.transformer.blocks.",
        "vision_ff": "model.vision_backbone.image_vit.transformer.resblocks.",
        "action_expert": "model.action_expert.",
        "image_projector": "model.vision_backbone.image_projector.",
        "lm_head": "lm_head",
    }
    names = [
        n for n in quantized_module_names(header) if n in dict(model.named_modules())
    ]
    picked = []
    for family, prefix in families.items():
        cands = []
        for n in names:
            if not n.startswith(prefix):
                continue
            module = dict(model.named_modules())[n]
            qs = getattr(module.weight, "quant_state", None)
            if qs is None:
                continue
            numel = int(qs.shape[0]) * int(qs.shape[1])
            if numel <= max_numel:
                cands.append((numel, n))
        for numel, n in sorted(cands)[:per_family]:
            picked.append((family, n, numel))
    return picked


def verify_nf4_vs_raw_bytes(
    model, model_dir: str | Path, max_numel: int = 12_000_000, per_family: int = 2
) -> dict:
    """Decode packed NF4 weights by hand and compare with bnb's dequantization.

    Proves (a) the payload really is 4-bit NF4 data, (b) the QuantState was
    attached to the right tensors, (c) the 4-bit matmul path consumes the same
    values.  An empty selection is an error, not a pass.
    """
    import numpy as np
    import torch
    import torch.nn.functional as F
    import bitsandbytes as bnb
    from safetensors import safe_open

    shard = Path(model_dir) / "model.safetensors"
    header = read_safetensors_header(shard)
    picked = _pick_verification_layers(model, header, max_numel, per_family)
    if not picked:
        raise RuntimeError(
            "no quantized layer was selected for verification; refusing to report "
            "an empty check as success"
        )

    eps = float(torch.finfo(torch.bfloat16).eps)
    out = []
    with safe_open(str(shard), framework="pt", device="cpu") as f:
        for family, name, numel in picked:
            module = dict(model.named_modules())[name]
            w = module.weight
            qs = w.quant_state
            packed = f.get_tensor(name + ".weight")
            rows = min(8, int(qs.shape[0]))
            cols = int(qs.shape[1])
            manual = _dequant_4bit_by_hand(packed, qs, rows, cols)
            ref = (
                bnb.functional.dequantize_4bit(w.data, qs)
                .float()
                .cpu()
                .numpy()
                .reshape(int(qs.shape[0]), cols)[:rows]
            )
            max_abs = float(np.max(np.abs(manual - ref)))
            tol = max(2.0 * eps * float(np.max(np.abs(ref))), 1e-6)
            # 4-bit matmul vs dense matmul on the same dequantized weights
            x = torch.randn(4, cols, device=w.device, dtype=torch.bfloat16) * 0.5
            y4 = bnb.matmul_4bit(x, w.t(), quant_state=qs)
            ref_weight = (
                bnb.functional.dequantize_4bit(w.data, qs).to(torch.bfloat16)
            )
            yref = F.linear(x, ref_weight)
            matmul_err = float((y4.float() - yref.float()).abs().max())
            matmul_tol = max(
                8.0 * eps * float(yref.float().abs().max().item()), 1e-3
            )
            del ref_weight, y4, yref, x
            entry = {
                "family": family,
                "module": name,
                "shape": [int(qs.shape[0]), cols],
                "numel": numel,
                "rows_checked": rows,
                "max_abs_diff": max_abs,
                "tolerance": tol,
                "max_abs_diff_over_tolerance": max_abs / tol,
                "decode_matches": bool(max_abs <= tol),
                "blocksize": int(qs.blocksize),
                "nested_blocksize": int(qs.state2.blocksize) if qs.state2 else None,
                "offset": float(qs.offset.detach().item()) if qs.offset is not None else None,
                "quant_type": str(qs.quant_type),
                "storage_dtype": str(w.dtype),
                "output_dtype": str(qs.dtype),
                "packed_bytes": int(w.numel()),
                "bf16_bytes_if_dense": int(numel * 2),
                "matmul_vs_dense_max_abs_diff": matmul_err,
                "matmul_tolerance": matmul_tol,
                "matmul_matches": bool(matmul_err <= matmul_tol),
            }
            out.append(entry)
            del manual, ref
    failed = [
        e["module"]
        for e in out
        if not (e["decode_matches"] and e["matmul_matches"])
    ]
    if failed:
        raise RuntimeError(f"NF4 verification failed for {failed[:3]}")
    return {
        "checked_layers": len(out),
        "families": sorted({e["family"] for e in out}),
        "method": (
            "independent high-nibble-first NF4 decode vs bnb.dequantize_4bit, plus "
            "bnb.matmul_4bit vs dense matmul on the dequantized weights"
        ),
        "tolerance_rule": "2*eps_bf16*max|ref| for decode, 8*eps_bf16*max|y| for matmul",
        "checked": out,
    }


def quant_state_roundtrip_check(model, model_dir: str | Path) -> dict:
    """Verify each Linear4bit's own state_dict aux tensors equal the file's.

    bnb registers the quantization state inside the module's state_dict, so a
    naive ``load_state_dict`` comparison sees those keys as 'missing'.  This
    check compares them against the file instead of ignoring them.
    """
    import torch
    from safetensors import safe_open

    shard = Path(model_dir) / "model.safetensors"
    header = read_safetensors_header(shard)
    names = [n for n in quantized_module_names(header) if n in dict(model.named_modules())]
    mods = dict(model.named_modules())
    expected_keys = {
        "weight",
        "weight.absmax",
        "weight.quant_map",
        "weight.nested_absmax",
        "weight.nested_quant_map",
        "weight.quant_state.bitsandbytes__nf4",
    }
    mismatched_modules = []
    key_shape_errors = []
    checked = 0
    with safe_open(str(shard), framework="pt", device="cpu") as f:
        for name in names:
            mod = mods[name]
            sd = mod.state_dict(prefix="")
            keys = set(sd)
            want = expected_keys | ({"bias"} if mod.bias is not None else set())
            if keys != want:
                key_shape_errors.append({"module": name, "keys": sorted(keys)})
                continue
            for leaf in ("absmax", "quant_map", "nested_absmax", "nested_quant_map"):
                local = sd[f"weight.{leaf}"]
                src = f.get_tensor(f"{name}.weight.{leaf}").to(local.device)
                if local.shape != src.shape or not torch.equal(local, src):
                    mismatched_modules.append(f"{name}.{leaf}")
            checked += 1
    if key_shape_errors or mismatched_modules:
        raise RuntimeError(
            "quant-state round-trip mismatch: "
            f"{key_shape_errors[:2]} {mismatched_modules[:5]}"
        )
    return {
        "modules_checked": checked,
        "keys_per_module": sorted(expected_keys),
        "note": (
            "Linear4bit exposes 6 state_dict entries per module (packed weight + "
            "5 quant-state tensors); all are compared with the file, none ignored"
        ),
    }


def load_processor(model_dir: str | Path):
    """Build MolmoAct2Processor from the repo's own code + tokenizer.json.

    The processor settings live in ``processor_config.json`` (there is no
    ``preprocessor_config.json``), so the sub-processors are constructed from
    that file's nested dictionaries.  Everything that influences the prompt
    (``image_use_col_tokens``, ``use_single_crop_col_tokens``,
    ``use_single_crop_start_token``, ``video_use_col_tokens``,
    ``use_frame_special_tokens``) is taken from the checkpoint, not from
    library defaults.
    """
    import json

    from transformers import AutoTokenizer

    model_dir = Path(model_dir)
    proc_mod = remote_module("processing_molmoact2", model_dir)
    cfg = json.loads((model_dir / "processor_config.json").read_text(encoding="utf-8"))
    ip_kwargs = {
        k: v
        for k, v in cfg["image_processor"].items()
        if k not in ("auto_map", "image_processor_type")
    }
    vp_kwargs = {
        k: v
        for k, v in cfg["video_processor"].items()
        if k not in ("auto_map", "video_processor_type")
    }
    image_processor = proc_mod.MolmoAct2ImageProcessor(**ip_kwargs)
    video_processor = proc_mod.MolmoAct2VideoProcessor(**vp_kwargs)
    tokenizer = AutoTokenizer.from_pretrained(str(model_dir), local_files_only=True)
    processor = proc_mod.MolmoAct2Processor(
        image_processor=image_processor,
        video_processor=video_processor,
        tokenizer=tokenizer,
        image_use_col_tokens=cfg.get("image_use_col_tokens", True),
        use_single_crop_col_tokens=cfg.get("use_single_crop_col_tokens", None),
        use_single_crop_start_token=cfg.get("use_single_crop_start_token", True),
        video_use_col_tokens=cfg.get("video_use_col_tokens", False),
        use_frame_special_tokens=cfg.get("use_frame_special_tokens", True),
    )
    processor.image_processor.max_crops = int(
        cfg["image_processor"].get("max_crops", 8)
    )
    return processor


def load_model(
    model_dir: str | Path,
    device: str = "cuda",
    dtype=None,
    verbose: bool = True,
):
    import torch

    if dtype is None:
        dtype = torch.bfloat16
    config = load_config(model_dir)
    model = build_meta_model(config, model_dir)
    report = install_prequantized(
        model, model_dir, device=device, dtype=dtype, verbose=verbose
    )
    model.eval()
    model.requires_grad_(False)
    report["audit"] = audit_model(model)
    return model, report
