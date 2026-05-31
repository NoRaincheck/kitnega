from __future__ import annotations

import math
from typing import Dict, List, Tuple

import torch
import torch.nn as nn


def to_torch(model, dtype=torch.float32, device="cpu"):
    """Convert a ``KneserNeyModel`` to a ``KneserNeyTorch`` module."""
    return KneserNeyTorch.from_model(model, dtype=dtype, device=device)


def to_onnx(
    model,
    filepath: str,
    dtype=torch.float32,
    verbose: bool = False,
    opset_version: int = 17,
):
    """Export a ``KneserNeyModel`` to ONNX.

    The exported model accepts two ``LongTensor`` inputs:
    ``context_ids`` (batch, order-1) left-padded with 0, and
    ``target_id`` (batch,) and returns ``log_prob`` (batch,).
    """
    torch_module = to_torch(model, dtype=dtype).eval()
    order = model.order

    dummy_context = torch.zeros((1, order - 1), dtype=torch.long)
    dummy_target = torch.zeros((1,), dtype=torch.long)

    torch.onnx.export(
        torch_module,
        (dummy_context, dummy_target),
        filepath,
        dynamo=False,
        input_names=["context_ids", "target_id"],
        output_names=["log_prob"],
        dynamic_axes={
            "context_ids": {0: "batch"},
            "target_id": {0: "batch"},
            "log_prob": {0: "batch"},
        },
        opset_version=opset_version,
        verbose=verbose,
    )
    return torch_module


def verify_export(model, torch_module, sentences, atol=1e-6):
    """Compare stdlib model and torch module scores on sample sentences."""
    diffs = []
    word2id = torch_module.word2id
    unk_id = word2id.get("<UNK>", 0)

    for sentence in sentences:
        for i, word in enumerate(sentence):
            ctx = sentence[max(0, i - model.order + 1) : i]

            stdlib_p = model.score(word, tuple(ctx))
            if stdlib_p == 0:
                continue

            pad_len = model.order - 1 - len(ctx)
            ctx_ids = [word2id.get(w, unk_id) for w in (["<PAD>"] * pad_len + list(ctx))]
            target_id_val = word2id.get(word, unk_id)

            ctx_tensor = torch.tensor([ctx_ids], dtype=torch.long)
            tgt_tensor = torch.tensor([target_id_val], dtype=torch.long)

            with torch.no_grad():
                torch_logp = torch_module(ctx_tensor, tgt_tensor).item()
            torch_p = math.exp(torch_logp) if torch_logp > -1e30 else 0.0
            diffs.append(abs(stdlib_p - torch_p))

    if not diffs:
        return {"max_diff": 0.0, "mean_diff": 0.0, "matched": True}

    return {
        "max_diff": max(diffs),
        "mean_diff": sum(diffs) / len(diffs),
        "matched": max(diffs) < atol,
    }


def _flat_index(ids: torch.Tensor, vocab_size: int) -> torch.Tensor:
    """Convert word IDs to a flat linear index using base-*V* encoding.

    ``ids`` has shape ``(..., N)``. Returns shape ``(...,)``.
    For an empty trailing dimension (N=0), returns a zero tensor.
    """
    if ids.shape[-1] == 0:
        return torch.zeros(ids.shape[:-1], dtype=torch.long, device=ids.device)
    idx = torch.zeros(ids.shape[:-1], dtype=torch.long, device=ids.device)
    for i in range(ids.shape[-1]):
        idx = idx * vocab_size + ids[..., i]
    return idx


class KneserNeyTorch(nn.Module):
    """PyTorch implementation of interpolated Kneser-Ney smoothing.

    Stores dense count tensors. Inputs are word IDs (not strings).
    The ``word2id`` mapping on the module provides the tokenizer bridge.
    """

    pad_token = "<PAD>"

    def __init__(
        self,
        order: int,
        discount: float,
        vocab_size: int,
        word2id: Dict[str, int],
        raw_counts: List[torch.Tensor],
        raw_n_plus: List[torch.Tensor],
        continuation_counts: List[torch.Tensor],
        continuation_totals: List[torch.Tensor],
        dtype=torch.float32,
    ):
        super().__init__()
        self.order = order
        self.discount = discount
        self.vocab_size = vocab_size
        self.word2id = word2id
        self._dtype = dtype

        self._raw_counts = nn.ParameterList([nn.Parameter(t.to(dtype), requires_grad=False) for t in raw_counts])
        self._raw_n_plus = nn.ParameterList([nn.Parameter(t.to(dtype), requires_grad=False) for t in raw_n_plus])
        self._continuation_counts = nn.ParameterList(
            [nn.Parameter(t.to(dtype), requires_grad=False) for t in continuation_counts]
        )
        self._continuation_totals = nn.ParameterList(
            [nn.Parameter(t.to(dtype), requires_grad=False) for t in continuation_totals]
        )

    @classmethod
    def from_model(cls, model, dtype=torch.float32, device="cpu") -> KneserNeyTorch:
        """Build from a trained ``KneserNeyModel``."""
        V = len(model.vocab_list)
        order = model.order
        word2id = {w: i for i, w in enumerate(model.vocab_list)}

        raw_counts_list: List[torch.Tensor] = []
        raw_n_plus_list: List[torch.Tensor] = []
        cont_counts_list: List[torch.Tensor] = []
        cont_totals_list: List[torch.Tensor] = []

        # Build raw count tensors
        for k in range(1, order + 1):
            shape = [V] * k
            tensor = torch.zeros(shape, device=device)

            if k == 1:
                for word, count in model._unigrams.items():
                    wid = word2id.get(word)
                    if wid is not None:
                        tensor[wid] = count
            else:
                ngrams = model._by_order.get(k, {})
                for ctx, dist in ngrams.items():
                    ctx_ids = [word2id.get(w) for w in ctx]
                    if any(i is None for i in ctx_ids):
                        continue
                    for word, count in dist.items():
                        wid = word2id.get(word)
                        if wid is not None:
                            tensor[tuple(ctx_ids) + (wid,)] = count

            if k > 1:
                n_plus = (tensor > 0).sum(dim=-1).to(dtype)
            else:
                n_plus = (tensor > 0).to(dtype)

            raw_counts_list.append(tensor)
            raw_n_plus_list.append(n_plus)

        # Build continuation tensors
        for k in range(1, order + 1):
            if k == order:
                c_counts = torch.zeros([V] * k, device=device)
                c_totals = torch.zeros([V] * (k - 1), device=device) if k > 1 else torch.tensor(0.0, device=device)
            else:
                next_counts = raw_counts_list[k]  # k+1 raw counts
                # Sum over first dimension: count[i, ctx..., word] > 0
                c_counts = (next_counts > 0).sum(dim=0).to(dtype)
                # Total over first dim and word: sum over word first, then sum over first dim
                n_distinct = (next_counts > 0).sum(dim=-1)
                c_totals = n_distinct.sum(dim=0).to(dtype)

            cont_counts_list.append(c_counts)
            cont_totals_list.append(c_totals)

        return cls(
            order=order,
            discount=model.discount,
            vocab_size=V,
            word2id=word2id,
            raw_counts=raw_counts_list,
            raw_n_plus=raw_n_plus_list,
            continuation_counts=cont_counts_list,
            continuation_totals=cont_totals_list,
            dtype=dtype,
        ).to(device)

    def forward(self, context_ids: torch.Tensor, target_id: torch.Tensor) -> torch.Tensor:
        """Log P(target | context).

        Parameters
        ----------
        context_ids : LongTensor, shape (batch, order-1)
            Left-padded context word IDs (0 = padding).
        target_id : LongTensor, shape (batch,)

        Returns
        -------
        Tensor, shape (batch,)
        """
        log_probs = []
        for i in range(context_ids.shape[0]):
            ctx = context_ids[i]
            tgt = target_id[i]

            p = self._unigram_score(tgt)

            for k in range(2, self.order + 1):
                ctx_k = ctx[-(k - 1) :]
                alpha, gamma = self._alpha_gamma(tgt, ctx_k, k)
                p = alpha + gamma * p

            log_probs.append(torch.log(torch.clamp(p, min=1e-30)))

        return torch.stack(log_probs)

    def _unigram_score(self, word_id: torch.Tensor) -> torch.Tensor:
        cont_counts = self._continuation_counts[0]
        cont_total = self._continuation_totals[0]

        score = cont_counts[word_id] / torch.clamp(cont_total, min=1e-30)
        zero = torch.tensor(0.0, dtype=self._dtype, device=word_id.device)
        return torch.where(cont_total > 0, score, zero)

    def _alpha_gamma(self, word_id: torch.Tensor, context: torch.Tensor, k: int) -> Tuple[torch.Tensor, torch.Tensor]:
        is_highest = k == self.order

        if is_highest:
            count = self._gather_count(self._raw_counts[k - 1], context, word_id)
            total = self._context_total(self._raw_counts[k - 1], context)
        else:
            count = self._gather_count(self._continuation_counts[k - 1], context, word_id)
            total = self._gather_total(self._continuation_totals[k - 1], context)

        has_data = total > 0

        alpha_raw = torch.clamp(count - self.discount, min=0.0) / torch.clamp(total, min=1e-30)
        n_plus = self._gather_n_plus(self._raw_n_plus[k - 1], context)
        gamma_raw = (self.discount * n_plus) / torch.clamp(total, min=1e-30)

        zero = torch.tensor(0.0, device=word_id.device, dtype=self._dtype)
        one = torch.tensor(1.0, device=word_id.device, dtype=self._dtype)

        alpha = torch.where(has_data, alpha_raw, zero)
        gamma = torch.where(has_data, gamma_raw, one)

        return alpha, gamma

    # ------------------------------------------------------------------
    # Tensor indexing helpers (ONNX-safe: no .tolist())
    # ------------------------------------------------------------------

    def _gather_count(self, tensor: torch.Tensor, context: torch.Tensor, word_id: torch.Tensor) -> torch.Tensor:
        if tensor.ndim == 1:
            return tensor[word_id]
        offset = _flat_index(context, self.vocab_size)
        flat = offset * self.vocab_size + word_id
        return tensor.flatten()[flat]

    def _context_total(self, tensor: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        if tensor.ndim == 1:
            return tensor.sum()
        offset = _flat_index(context, self.vocab_size)
        start = offset * self.vocab_size
        end = (offset + 1) * self.vocab_size
        return tensor.flatten()[start:end].sum()

    def _gather_total(self, tensor: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        return self._scalar_for_context(tensor, context)

    def _gather_n_plus(self, tensor: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        return self._scalar_for_context(tensor, context)

    def _scalar_for_context(self, tensor: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        if tensor.ndim == 0:
            return tensor
        if tensor.ndim == 1 and context.shape[0] == 0:
            flat = _flat_index(context, self.vocab_size)
            return tensor.flatten()[flat]
        flat = _flat_index(context, self.vocab_size)
        return tensor.flatten()[flat]

    def extra_repr(self) -> str:
        return f"order={self.order}, discount={self.discount}, vocab_size={self.vocab_size}"
