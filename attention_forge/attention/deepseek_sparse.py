"""DeepSeek-style compressed sparse attention reference helpers.

This is a CPU-friendly educational reference for the core CSA/DSA mechanics:

1. gated block compression of token states into compressed KV entries
2. lightning-indexer-style scoring of compressed entries
3. top-k compressed entry selection under a causal visibility mask
4. attention over local token KV plus selected compressed KV entries

It intentionally does not include production details such as RoPE, quantization,
Hadamard rotation, distributed context parallelism, or custom sparse kernels.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn.functional as F


@dataclass(frozen=True)
class SparseSelection:
    """Selected compressed blocks and local token positions for each query."""

    compressed_block_indices: torch.Tensor
    local_token_indices: torch.Tensor


def gated_block_compress(
    values: torch.Tensor,
    gate_logits: torch.Tensor,
    block_size: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compress token values into one entry per block with gated pooling.

    Args:
        values: ``[batch, seq_len, dim]`` token values to compress.
        gate_logits: ``[batch, seq_len, dim]`` per-dimension compression logits.
        block_size: number of tokens per compressed entry.

    Returns:
        ``compressed`` with shape ``[batch, num_blocks, dim]`` and ``weights`` with
        shape ``[batch, num_blocks, block_size, dim]``.
    """

    _validate_3d("values", values)
    _validate_3d("gate_logits", gate_logits)
    if values.shape != gate_logits.shape:
        raise ValueError("values and gate_logits must have the same shape.")
    _validate_positive("block_size", block_size)

    batch, seq_len, dim = values.shape
    num_blocks = math.ceil(seq_len / block_size)
    padded_len = num_blocks * block_size
    pad_len = padded_len - seq_len

    if pad_len:
        values = F.pad(values, (0, 0, 0, pad_len), value=0)
        gate_logits = F.pad(gate_logits, (0, 0, 0, pad_len), value=float("-inf"))

    values_by_block = values.view(batch, num_blocks, block_size, dim)
    gates_by_block = gate_logits.view(batch, num_blocks, block_size, dim)
    weights = torch.softmax(gates_by_block, dim=2)
    compressed = (values_by_block * weights).sum(dim=2)
    return compressed, weights


def compressed_block_visibility(
    query_len: int,
    num_blocks: int,
    block_size: int,
    *,
    device: torch.device | None = None,
) -> torch.Tensor:
    """Return causal visibility mask for compressed blocks.

    ``mask[q, b]`` is true when compressed block ``b`` is visible to query token
    ``q`` in a prefill-style sequence. A block becomes visible once its final
    token is not in the future.
    """

    _validate_positive("query_len", query_len)
    _validate_positive("num_blocks", num_blocks)
    _validate_positive("block_size", block_size)

    query_positions = torch.arange(query_len, device=device)
    block_ids = torch.arange(num_blocks, device=device)
    visible_blocks = (query_positions + 1) // block_size
    return block_ids.unsqueeze(0) < visible_blocks.unsqueeze(1)


def lightning_index_scores(
    index_queries: torch.Tensor,
    compressed_index_keys: torch.Tensor,
    head_weights: torch.Tensor,
) -> torch.Tensor:
    """Score compressed entries with a small multi-head indexer.

    This mirrors the core scoring structure in the DeepSeek inference
    code: per-index-head query/key similarity, ReLU, then weighted sum over
    index heads.

    Args:
        index_queries: ``[batch, query_len, index_heads, index_dim]``.
        compressed_index_keys: ``[batch, num_blocks, index_dim]``.
        head_weights: ``[batch, query_len, index_heads]``.

    Returns:
        Index scores with shape ``[batch, query_len, num_blocks]``.
    """

    if index_queries.ndim != 4:
        raise ValueError("index_queries must have shape [batch, query_len, heads, dim].")
    _validate_3d("compressed_index_keys", compressed_index_keys)
    _validate_3d("head_weights", head_weights)

    batch, query_len, index_heads, index_dim = index_queries.shape
    if compressed_index_keys.shape[0] != batch or compressed_index_keys.shape[2] != index_dim:
        raise ValueError("compressed_index_keys must match query batch and index dim.")
    if head_weights.shape != (batch, query_len, index_heads):
        raise ValueError("head_weights must have shape [batch, query_len, index_heads].")

    raw_scores = torch.einsum("bqhd,bnd->bqhn", index_queries, compressed_index_keys)
    return (raw_scores.relu() * head_weights.unsqueeze(-1)).sum(dim=2)


def select_topk_compressed_blocks(
    index_scores: torch.Tensor,
    visible_mask: torch.Tensor,
    top_k: int,
) -> torch.Tensor:
    """Select top-k visible compressed blocks for each query.

    Invalid entries are returned as ``-1`` when fewer than ``top_k`` compressed
    blocks are visible.
    """

    _validate_3d("index_scores", index_scores)
    if visible_mask.ndim != 2:
        raise ValueError("visible_mask must have shape [query_len, num_blocks].")
    _validate_positive("top_k", top_k)

    batch, query_len, num_blocks = index_scores.shape
    if visible_mask.shape != (query_len, num_blocks):
        raise ValueError("visible_mask must match [query_len, num_blocks].")

    masked_scores = index_scores.masked_fill(~visible_mask.unsqueeze(0), float("-inf"))
    k = min(top_k, num_blocks)
    top_scores, top_indices = masked_scores.topk(k, dim=-1)
    top_indices = torch.where(torch.isfinite(top_scores), top_indices, -1)

    if k == top_k:
        return top_indices

    pad = top_indices.new_full((batch, query_len, top_k - k), -1)
    return torch.cat([top_indices, pad], dim=-1)


def local_token_indices(
    query_len: int,
    local_window: int,
    *,
    device: torch.device | None = None,
) -> torch.Tensor:
    """Return padded causal local-window token indices for each query."""

    _validate_positive("query_len", query_len)
    _validate_positive("local_window", local_window)

    query_positions = torch.arange(query_len, device=device).unsqueeze(1)
    offsets = torch.arange(local_window, device=device).unsqueeze(0)
    indices = query_positions - local_window + 1 + offsets
    return torch.where(indices >= 0, indices, -1)


def build_sparse_selection(
    index_scores: torch.Tensor,
    block_size: int,
    top_k: int,
    local_window: int,
) -> SparseSelection:
    """Build local-token and compressed-block selections."""

    _, query_len, num_blocks = index_scores.shape
    visible = compressed_block_visibility(
        query_len,
        num_blocks,
        block_size,
        device=index_scores.device,
    )
    return SparseSelection(
        compressed_block_indices=select_topk_compressed_blocks(index_scores, visible, top_k),
        local_token_indices=local_token_indices(
            query_len,
            local_window,
            device=index_scores.device,
        ),
    )


def sparse_attention_with_compressed_entries(
    query: torch.Tensor,
    token_kv: torch.Tensor,
    compressed_kv: torch.Tensor,
    selection: SparseSelection,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Attend over local token KV and selected compressed KV entries.

    Args:
        query: ``[batch, query_len, query_heads, dim]``.
        token_kv: ``[batch, seq_len, dim]`` local uncompressed KV entries.
        compressed_kv: ``[batch, num_blocks, dim]`` compressed entries.
        selection: local token and compressed block indices.

    Returns:
        ``output`` with shape ``[batch, query_len, query_heads, dim]`` and
        padded attention weights with shape
        ``[batch, query_len, query_heads, local_window + top_k]``.
    """

    if query.ndim != 4:
        raise ValueError("query must have shape [batch, query_len, query_heads, dim].")
    _validate_3d("token_kv", token_kv)
    _validate_3d("compressed_kv", compressed_kv)

    batch, query_len, query_heads, dim = query.shape
    if token_kv.shape[0] != batch or token_kv.shape[2] != dim:
        raise ValueError("token_kv must match query batch and dim.")
    if compressed_kv.shape[0] != batch or compressed_kv.shape[2] != dim:
        raise ValueError("compressed_kv must match query batch and dim.")

    local_indices = selection.local_token_indices
    block_indices = selection.compressed_block_indices
    if local_indices.shape[0] != query_len:
        raise ValueError("local indices must have one row per query token.")
    if block_indices.shape[:2] != (batch, query_len):
        raise ValueError("compressed block indices must match [batch, query_len].")

    output = query.new_empty(batch, query_len, query_heads, dim)
    max_entries = local_indices.shape[1] + block_indices.shape[2]
    all_weights = query.new_zeros(batch, query_len, query_heads, max_entries)

    for b in range(batch):
        for q_pos in range(query_len):
            entries = []
            for token_idx in local_indices[q_pos].tolist():
                if 0 <= token_idx < token_kv.shape[1]:
                    entries.append(token_kv[b, token_idx])
            for block_idx in block_indices[b, q_pos].tolist():
                if 0 <= block_idx < compressed_kv.shape[1]:
                    entries.append(compressed_kv[b, block_idx])

            if not entries:
                raise ValueError("selection produced no attention entries.")

            kv_entries = torch.stack(entries, dim=0)
            scores = query[b, q_pos] @ kv_entries.T / math.sqrt(dim)
            weights = torch.softmax(scores, dim=-1)
            output[b, q_pos] = weights @ kv_entries
            all_weights[b, q_pos, :, : kv_entries.shape[0]] = weights

    return output, all_weights


def _validate_3d(name: str, tensor: torch.Tensor) -> None:
    if tensor.ndim != 3:
        raise ValueError(f"{name} must be a 3D tensor.")


def _validate_positive(name: str, value: int) -> None:
    if value <= 0:
        raise ValueError(f"{name} must be positive, got {value}.")
