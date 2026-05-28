import pytest

from attention_forge.bench.kv_memory import (
    AttentionVariant,
    KVCacheConfig,
    bytes_to_gb,
    bytes_to_gib,
    kv_cache_bytes,
    kv_heads_for_variant,
)


def test_mha_kv_cache_memory_matches_manual_formula() -> None:
    config = KVCacheConfig(
        batch_size=1,
        sequence_length=4096,
        num_layers=32,
        num_query_heads=32,
        head_dim=128,
        bytes_per_element=2,
    )

    assert kv_cache_bytes(config) == 2_147_483_648
    assert bytes_to_gib(kv_cache_bytes(config)) == 2.0
    assert round(bytes_to_gb(kv_cache_bytes(config)), 2) == 2.15


def test_doubling_sequence_length_doubles_cache_memory() -> None:
    base = KVCacheConfig(
        batch_size=1,
        sequence_length=4096,
        num_layers=32,
        num_query_heads=32,
        head_dim=128,
        bytes_per_element=2,
    )
    longer = KVCacheConfig(
        batch_size=1,
        sequence_length=8192,
        num_layers=32,
        num_query_heads=32,
        head_dim=128,
        bytes_per_element=2,
    )

    assert kv_cache_bytes(longer) == 2 * kv_cache_bytes(base)


def test_mqa_uses_one_kv_head() -> None:
    config = KVCacheConfig(
        batch_size=1,
        sequence_length=1024,
        num_layers=24,
        num_query_heads=32,
        head_dim=128,
        bytes_per_element=2,
        variant=AttentionVariant.MQA,
    )

    assert kv_heads_for_variant(config) == 1


def test_gqa_uses_configured_kv_heads() -> None:
    config = KVCacheConfig(
        batch_size=1,
        sequence_length=1024,
        num_layers=24,
        num_query_heads=32,
        num_kv_heads=8,
        head_dim=128,
        bytes_per_element=2,
        variant=AttentionVariant.GQA,
    )

    assert kv_heads_for_variant(config) == 8


def test_gqa_requires_query_heads_divisible_by_kv_heads() -> None:
    config = KVCacheConfig(
        batch_size=1,
        sequence_length=1024,
        num_layers=24,
        num_query_heads=30,
        num_kv_heads=8,
        head_dim=128,
        bytes_per_element=2,
        variant=AttentionVariant.GQA,
    )

    with pytest.raises(ValueError, match="divisible"):
        kv_heads_for_variant(config)


def test_mla_uses_latent_cache_width() -> None:
    config = KVCacheConfig(
        batch_size=1,
        sequence_length=4096,
        num_layers=32,
        num_query_heads=32,
        head_dim=128,
        bytes_per_element=2,
        variant=AttentionVariant.MLA,
        latent_dim=512,
    )

    assert kv_cache_bytes(config) == 134_217_728
    assert bytes_to_gib(kv_cache_bytes(config)) == 0.125


def test_dsa_defaults_to_dense_kv_storage() -> None:
    dense = KVCacheConfig(
        batch_size=1,
        sequence_length=2048,
        num_layers=32,
        num_query_heads=32,
        head_dim=128,
        bytes_per_element=2,
        variant=AttentionVariant.MHA,
    )
    sparse = KVCacheConfig(
        batch_size=1,
        sequence_length=2048,
        num_layers=32,
        num_query_heads=32,
        head_dim=128,
        bytes_per_element=2,
        variant=AttentionVariant.DSA,
    )

    assert kv_heads_for_variant(sparse) == 32
    assert kv_cache_bytes(sparse) == kv_cache_bytes(dense)


def test_dsa_can_account_for_grouped_kv_storage() -> None:
    config = KVCacheConfig(
        batch_size=1,
        sequence_length=2048,
        num_layers=32,
        num_query_heads=32,
        num_kv_heads=8,
        head_dim=128,
        bytes_per_element=2,
        variant=AttentionVariant.DSA,
    )

    assert kv_heads_for_variant(config) == 8
