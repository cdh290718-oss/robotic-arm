"""Memory patch: remove the per-call full-vocabulary ``torch.cat`` in the embedding.

Upstream ``MolmoAct2Embedding.forward`` is

    F.embedding(x, torch.cat([self.embedding, self.new_embedding], dim=0))

With ``vocab_size=154624``, ``additional_vocab_size=128`` and
``hidden_size=2560`` the concatenated table is 396,165,120 elements, i.e.
**0.7379 GiB in bf16** (1.4758 GiB in fp32).  That copy is allocated on every
call, even though only a few thousand distinct token ids are actually looked
up.  This patch looks the two tables up separately and merges only the rows
that were requested, so the temporary is ``num_tokens * hidden * 2`` bytes
instead of the whole vocabulary.

Behaviour preserved
-------------------
* token id meaning is untouched (base table keeps ids ``[0, vocab_size)``,
  the additional table keeps ``[vocab_size, vocab_size + additional)``),
* the boundary id ``vocab_size`` is served by the additional table, exactly as
  the concatenation did,
* out-of-range ids (negative, or ``>= vocab_size + additional``) still raise
  ``IndexError`` from ``F.embedding``; they are not clamped or silently mapped,
* the result is bit-for-bit identical: every row is a verbatim copy of the row
  that the concatenated table would have contained.

The original module is not modified on disk; ``install()`` monkeypatches the
class in the already-imported remote module, and ``legacy_forward`` keeps a
handle on the original implementation for A/B numeric comparison.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

_PATCH_STATE = {"installed": False, "class": None, "original": None}


def patched_forward(self, x: torch.Tensor) -> torch.Tensor:
    base = self.embedding
    extra = self.new_embedding
    n_base = int(base.shape[0])
    n_extra = int(extra.shape[0])
    total = n_base + n_extra

    # Preserve F.embedding's failure behaviour for ids outside the joint table.
    if x.numel():
        lo = int(x.min())
        hi = int(x.max())
        if lo < 0 or hi >= total:
            raise IndexError(
                "index out of range in self (MolmoAct2Embedding): token ids must "
                f"be in [0, {total}) for vocab_size={n_base} + "
                f"additional_vocab_size={n_extra}, got [{lo}, {hi}]"
            )

    is_extra = x >= n_base
    if not bool(is_extra.any()):
        return F.embedding(x, base)

    base_ids = torch.where(is_extra, torch.zeros_like(x), x)
    out = F.embedding(base_ids, base)
    extra_vecs = F.embedding((x - n_base).clamp(min=0), extra)
    return torch.where(is_extra.unsqueeze(-1), extra_vecs, out)


def install(model_module) -> dict:
    """Patch ``MolmoAct2Embedding.forward`` in the given remote module."""
    cls = model_module.MolmoAct2Embedding
    if getattr(cls, "_no_cat_patch", False):
        return {"installed": True, "already": True}
    _PATCH_STATE["class"] = cls
    _PATCH_STATE["original"] = cls.forward
    cls.forward = patched_forward
    cls._no_cat_patch = True
    _PATCH_STATE["installed"] = True
    return {"installed": True, "already": False}


def uninstall() -> dict:
    cls = _PATCH_STATE["class"]
    if cls is None or _PATCH_STATE["original"] is None:
        return {"installed": False}
    cls.forward = _PATCH_STATE["original"]
    cls._no_cat_patch = False
    _PATCH_STATE["installed"] = False
    return {"installed": False}


def legacy_forward(self, x: torch.Tensor) -> torch.Tensor:
    """The original implementation (kept for A/B comparison)."""
    return F.embedding(x, torch.cat([self.embedding, self.new_embedding], dim=0))
