"""KV-cache memory accounting helpers.

The formulas in this module are intentionally explicit. They are the first
building block for comparing MHA, MQA, GQA, MLA, and sparse attention policies.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class AttentionVariant(str, Enum):
    """Supported attention-cache layouts for memory accounting."""

    MHA = "mha"
    MQA = "mqa"
    GQA = "gqa"
    MLA = "mla"
    DSA = "dsa"


@dataclass(frozen=True)
class KVCacheConfig:
    """Configuration needed to estimate KV-cache memory."""

    batch_size: int
    sequence_length: int
    num_layers: int
    num_query_heads: int
    head_dim: int
    bytes_per_element: int
    variant: AttentionVariant = AttentionVariant.MHA
    num_kv_heads: int | None = None
    latent_dim: int | None = None


def kv_heads_for_variant(config: KVCacheConfig) -> int:
    """Return the number of KV heads stored for head-based cache variants."""

    if config.variant == AttentionVariant.MHA:
        return config.num_query_heads
    if config.variant == AttentionVariant.DSA:
        return config.num_kv_heads or config.num_query_heads
    if config.variant == AttentionVariant.MQA:
        return 1
    if config.variant == AttentionVariant.GQA:
        if config.num_kv_heads is None:
            raise ValueError("GQA requires num_kv_heads.")
        if config.num_query_heads % config.num_kv_heads != 0:
            raise ValueError("num_query_heads must be divisible by num_kv_heads for GQA.")
        return config.num_kv_heads
    raise ValueError(f"{config.variant.value} does not use head-based KV memory accounting.")


def kv_cache_bytes(config: KVCacheConfig) -> int:
    """Estimate total KV-cache bytes across all layers."""

    _validate_common_config(config)

    if config.variant == AttentionVariant.MLA:
        if config.latent_dim is None:
            raise ValueError("MLA requires latent_dim.")
        _validate_positive("latent_dim", config.latent_dim)
        return (
            config.batch_size
            * config.sequence_length
            * config.num_layers
            * config.latent_dim
            * config.bytes_per_element
        )

    return (
        config.batch_size
        * config.sequence_length
        * config.num_layers
        * kv_heads_for_variant(config)
        * config.head_dim
        * 2
        * config.bytes_per_element
    )


def bytes_to_gib(num_bytes: int) -> float:
    """Convert bytes to GiB, using powers of 2."""

    return num_bytes / 1024**3


def bytes_to_gb(num_bytes: int) -> float:
    """Convert bytes to decimal GB, using powers of 10."""

    return num_bytes / 1000**3


def _validate_common_config(config: KVCacheConfig) -> None:
    _validate_positive("batch_size", config.batch_size)
    _validate_positive("sequence_length", config.sequence_length)
    _validate_positive("num_layers", config.num_layers)
    _validate_positive("num_query_heads", config.num_query_heads)
    _validate_positive("head_dim", config.head_dim)
    _validate_positive("bytes_per_element", config.bytes_per_element)


def _validate_positive(name: str, value: int) -> None:
    if value <= 0:
        raise ValueError(f"{name} must be positive, got {value}.")
