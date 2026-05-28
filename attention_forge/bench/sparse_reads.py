"""Attention read-cost estimates for dense and sparse decode.

These helpers count how many cache entries a query token attends to. They do
not model kernel overhead, memory layout, indexer cost, or quality impact.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class DSAReadConfig:
    """Simplified DeepSeek-style sparse read configuration.

    `compression_block_size` models CSA-style compression from many tokens to
    one compressed KV entry. `top_k` is the number of compressed entries selected
    by the sparse indexer. `local_window` models the extra recent uncompressed
    tokens used to preserve local detail.
    """

    sequence_length: int
    compression_block_size: int
    top_k: int
    local_window: int = 0


def dense_decode_reads(sequence_length: int) -> int:
    """Return dense attention reads for one decode query."""

    _validate_positive("sequence_length", sequence_length)
    return sequence_length


def compressed_entries(sequence_length: int, block_size: int) -> int:
    """Return the number of compressed entries after block compression."""

    _validate_positive("sequence_length", sequence_length)
    _validate_positive("block_size", block_size)
    return (sequence_length + block_size - 1) // block_size


def dsa_decode_reads(config: DSAReadConfig) -> int:
    """Estimate cache entries read by one simplified DSA decode query.

    The result counts selected compressed entries plus local uncompressed token
    entries. If the context is shorter than top-k or local window, reads are
    clipped to the visible sequence.
    """

    _validate_dsa_config(config)
    num_compressed = compressed_entries(
        config.sequence_length,
        config.compression_block_size,
    )
    selected_compressed = min(config.top_k, num_compressed)
    selected_local = min(config.local_window, config.sequence_length)
    return selected_compressed + selected_local


def read_reduction_factor(dense_reads: int, sparse_reads: int) -> float:
    """Return how many times fewer entries sparse attention reads."""

    _validate_positive("dense_reads", dense_reads)
    _validate_positive("sparse_reads", sparse_reads)
    return dense_reads / sparse_reads


def _validate_dsa_config(config: DSAReadConfig) -> None:
    _validate_positive("sequence_length", config.sequence_length)
    _validate_positive("compression_block_size", config.compression_block_size)
    _validate_positive("top_k", config.top_k)
    if config.local_window < 0:
        raise ValueError(f"local_window must be non-negative, got {config.local_window}.")


def _validate_positive(name: str, value: int) -> None:
    if value <= 0:
        raise ValueError(f"{name} must be positive, got {value}.")
