import pytest

from attention_forge.bench.sparse_reads import (
    DSAReadConfig,
    compressed_entries,
    dense_decode_reads,
    dsa_decode_reads,
    read_reduction_factor,
)


def test_dense_decode_reads_all_previous_entries() -> None:
    assert dense_decode_reads(4096) == 4096


def test_compressed_entries_rounds_up() -> None:
    assert compressed_entries(sequence_length=4096, block_size=64) == 64
    assert compressed_entries(sequence_length=4097, block_size=64) == 65


def test_dsa_reads_top_k_compressed_entries_plus_local_window() -> None:
    config = DSAReadConfig(
        sequence_length=1_000_000,
        compression_block_size=64,
        top_k=128,
        local_window=256,
    )

    assert dsa_decode_reads(config) == 384


def test_dsa_read_reduction_for_long_context() -> None:
    dense = dense_decode_reads(1_000_000)
    sparse = dsa_decode_reads(
        DSAReadConfig(
            sequence_length=1_000_000,
            compression_block_size=64,
            top_k=128,
            local_window=256,
        )
    )

    assert round(read_reduction_factor(dense, sparse), 1) == 2604.2


def test_dsa_reads_are_clipped_for_short_context() -> None:
    config = DSAReadConfig(
        sequence_length=128,
        compression_block_size=64,
        top_k=8,
        local_window=256,
    )

    assert compressed_entries(128, 64) == 2
    assert dsa_decode_reads(config) == 130


def test_dsa_rejects_invalid_top_k() -> None:
    with pytest.raises(ValueError, match="top_k"):
        dsa_decode_reads(
            DSAReadConfig(
                sequence_length=1024,
                compression_block_size=64,
                top_k=0,
            )
        )
