"""Static plots for KV memory and sparse read-cost comparisons."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt

from attention_forge.bench.kv_memory import (
    AttentionVariant,
    KVCacheConfig,
    bytes_to_gib,
    kv_cache_bytes,
)
from attention_forge.bench.sparse_reads import (
    DSAReadConfig,
    dense_decode_reads,
    dsa_decode_reads,
)


@dataclass(frozen=True)
class MemoryCurve:
    name: str
    variant: AttentionVariant
    num_kv_heads: int | None = None
    latent_dim: int | None = None


DEFAULT_MEMORY_CURVES = (
    MemoryCurve("MHA", AttentionVariant.MHA),
    MemoryCurve("MQA", AttentionVariant.MQA),
    MemoryCurve("GQA-8", AttentionVariant.GQA, num_kv_heads=8),
    MemoryCurve("MLA-512", AttentionVariant.MLA, latent_dim=512),
    MemoryCurve("DSA dense-storage", AttentionVariant.DSA),
)


def memory_curve_points(
    sequence_lengths: list[int],
    curves: tuple[MemoryCurve, ...] = DEFAULT_MEMORY_CURVES,
) -> dict[str, list[float]]:
    """Return KV-cache memory in GiB for each curve."""

    points: dict[str, list[float]] = {}
    for curve in curves:
        points[curve.name] = [
            bytes_to_gib(
                kv_cache_bytes(
                    KVCacheConfig(
                        batch_size=1,
                        sequence_length=seq_len,
                        num_layers=32,
                        num_query_heads=32,
                        num_kv_heads=curve.num_kv_heads,
                        head_dim=128,
                        bytes_per_element=2,
                        variant=curve.variant,
                        latent_dim=curve.latent_dim,
                    )
                )
            )
            for seq_len in sequence_lengths
        ]
    return points


def sparse_read_points(sequence_lengths: list[int]) -> dict[str, list[int]]:
    """Return dense and simplified DSA/CSA read counts for each sequence length."""

    dense = [dense_decode_reads(seq_len) for seq_len in sequence_lengths]
    sparse = [
        dsa_decode_reads(
            DSAReadConfig(
                sequence_length=seq_len,
                compression_block_size=64,
                top_k=128,
                local_window=256,
            )
        )
        for seq_len in sequence_lengths
    ]
    return {"Dense": dense, "DSA/CSA reads": sparse}


def render_memory_plot(output_path: Path) -> Path:
    """Render the first KV-cache memory comparison plot."""

    sequence_lengths = [4096, 8192, 32768, 131072]
    points = memory_curve_points(sequence_lengths)

    fig, ax = plt.subplots(figsize=(8, 5))
    for name, values in points.items():
        ax.plot(sequence_lengths, values, marker="o", label=name)

    ax.set_xscale("log", base=2)
    ax.set_yscale("log", base=2)
    ax.set_xlabel("Sequence length")
    ax.set_ylabel("KV cache memory (GiB)")
    ax.set_title("KV cache memory grows linearly with context length")
    ax.grid(True, which="both", alpha=0.25)
    ax.legend()
    fig.tight_layout()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180)
    plt.close(fig)
    return output_path


def render_sparse_reads_plot(output_path: Path) -> Path:
    """Render dense-vs-sparse decode read-cost plot."""

    sequence_lengths = [4096, 8192, 32768, 131072, 1_000_000]
    points = sparse_read_points(sequence_lengths)

    fig, ax = plt.subplots(figsize=(8, 5))
    for name, values in points.items():
        ax.plot(sequence_lengths, values, marker="o", label=name)

    ax.set_xscale("log", base=2)
    ax.set_yscale("log", base=10)
    ax.set_xlabel("Sequence length")
    ax.set_ylabel("Entries read per decode query")
    ax.set_title("Sparse attention reduces decode reads")
    ax.grid(True, which="both", alpha=0.25)
    ax.legend()
    fig.tight_layout()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180)
    plt.close(fig)
    return output_path
