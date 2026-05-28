"""Print first-pass KV memory and sparse read-cost tables.

This script is intentionally simple. It gives us a runnable artifact for
Milestone 1 before we build real decode kernels.
"""

from __future__ import annotations

import argparse

from attention_forge.bench.kv_memory import (
    AttentionVariant,
    KVCacheConfig,
    bytes_to_gib,
    kv_cache_bytes,
    kv_heads_for_variant,
)
from attention_forge.bench.sparse_reads import (
    DSAReadConfig,
    dense_decode_reads,
    dsa_decode_reads,
    read_reduction_factor,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=["memory", "reads", "all"], default="all")
    args = parser.parse_args()

    if args.mode in {"memory", "all"}:
        print_memory_table()
    if args.mode in {"reads", "all"}:
        if args.mode == "all":
            print()
        print_read_table()


def print_memory_table() -> None:
    print("KV cache memory, batch=1, layers=32, q_heads=32, head_dim=128, fp16")
    print()
    print(f"{'variant':<10} {'seq_len':>10} {'kv_heads/latent':>16} {'GiB':>10}")
    print("-" * 52)

    sequence_lengths = [4096, 8192, 32768, 131072]
    configs = [
        ("MHA", AttentionVariant.MHA, None, None),
        ("MQA", AttentionVariant.MQA, None, None),
        ("GQA-8", AttentionVariant.GQA, 8, None),
        ("MLA-512", AttentionVariant.MLA, None, 512),
        ("DSA", AttentionVariant.DSA, None, None),
    ]

    for seq_len in sequence_lengths:
        for name, variant, num_kv_heads, latent_dim in configs:
            config = KVCacheConfig(
                batch_size=1,
                sequence_length=seq_len,
                num_layers=32,
                num_query_heads=32,
                num_kv_heads=num_kv_heads,
                head_dim=128,
                bytes_per_element=2,
                variant=variant,
                latent_dim=latent_dim,
            )
            width = latent_dim if latent_dim is not None else kv_heads_for_variant(config)
            gib = bytes_to_gib(kv_cache_bytes(config))
            print(f"{name:<10} {seq_len:>10} {width:>16} {gib:>10.3f}")


def print_read_table() -> None:
    print("Dense vs simplified DSA/CSA decode reads per query")
    print()
    print(f"{'seq_len':>10} {'block':>8} {'top_k':>8} {'local':>8} {'reads':>10} {'reduction':>12}")
    print("-" * 68)

    rows = [
        DSAReadConfig(sequence_length=4096, compression_block_size=64, top_k=32, local_window=256),
        DSAReadConfig(sequence_length=32768, compression_block_size=64, top_k=64, local_window=256),
        DSAReadConfig(
            sequence_length=1_000_000,
            compression_block_size=64,
            top_k=128,
            local_window=256,
        ),
    ]

    for config in rows:
        dense = dense_decode_reads(config.sequence_length)
        sparse = dsa_decode_reads(config)
        reduction = read_reduction_factor(dense, sparse)
        print(
            f"{config.sequence_length:>10} "
            f"{config.compression_block_size:>8} "
            f"{config.top_k:>8} "
            f"{config.local_window:>8} "
            f"{sparse:>10} "
            f"{reduction:>11.1f}x"
        )


if __name__ == "__main__":
    main()
