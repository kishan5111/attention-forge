import torch

from attention_forge.attention.deepseek_sparse import (
    SparseSelection,
    build_sparse_selection,
    compressed_block_visibility,
    gated_block_compress,
    lightning_index_scores,
    local_token_indices,
    select_topk_compressed_blocks,
    sparse_attention_with_compressed_entries,
)


def test_gated_block_compress_averages_when_gates_are_equal() -> None:
    values = torch.arange(1, 13, dtype=torch.float32).view(1, 6, 2)
    gate_logits = torch.zeros_like(values)

    compressed, weights = gated_block_compress(values, gate_logits, block_size=3)

    expected = torch.tensor([[[3.0, 4.0], [9.0, 10.0]]])
    assert compressed.shape == (1, 2, 2)
    assert weights.shape == (1, 2, 3, 2)
    torch.testing.assert_close(compressed, expected)


def test_gated_block_compress_handles_padding() -> None:
    values = torch.tensor([[[1.0], [2.0], [3.0], [4.0], [5.0]]])
    gate_logits = torch.zeros_like(values)

    compressed, _ = gated_block_compress(values, gate_logits, block_size=4)

    expected = torch.tensor([[[2.5], [5.0]]])
    torch.testing.assert_close(compressed, expected)


def test_compressed_block_visibility_waits_until_block_is_complete() -> None:
    mask = compressed_block_visibility(query_len=8, num_blocks=2, block_size=4)

    expected = torch.tensor(
        [
            [False, False],
            [False, False],
            [False, False],
            [True, False],
            [True, False],
            [True, False],
            [True, False],
            [True, True],
        ]
    )
    torch.testing.assert_close(mask, expected)


def test_lightning_index_scores_uses_relu_and_head_weights() -> None:
    index_queries = torch.tensor([[[[1.0, 0.0], [-1.0, 0.0]]]])
    compressed_keys = torch.tensor([[[2.0, 0.0], [-3.0, 0.0]]])
    head_weights = torch.tensor([[[1.0, 10.0]]])

    scores = lightning_index_scores(index_queries, compressed_keys, head_weights)

    # Head 0 scores block 0 positively. Head 1 scores block 1 positively.
    expected = torch.tensor([[[2.0, 30.0]]])
    torch.testing.assert_close(scores, expected)


def test_select_topk_compressed_blocks_masks_invisible_blocks() -> None:
    index_scores = torch.tensor([[[10.0, 0.0], [1.0, 9.0]]])
    visible_mask = torch.tensor([[False, False], [True, False]])

    selected = select_topk_compressed_blocks(index_scores, visible_mask, top_k=2)

    expected = torch.tensor([[[-1, -1], [0, -1]]])
    torch.testing.assert_close(selected, expected)


def test_local_token_indices_are_left_padded_with_negative_one() -> None:
    indices = local_token_indices(query_len=4, local_window=3)

    expected = torch.tensor([[-1, -1, 0], [-1, 0, 1], [0, 1, 2], [1, 2, 3]])
    torch.testing.assert_close(indices, expected)


def test_build_sparse_selection_combines_local_and_topk_blocks() -> None:
    scores = torch.arange(8, dtype=torch.float32).view(1, 4, 2)

    selection = build_sparse_selection(
        scores,
        block_size=2,
        top_k=1,
        local_window=2,
    )

    assert selection.local_token_indices.shape == (4, 2)
    assert selection.compressed_block_indices.shape == (1, 4, 1)
    torch.testing.assert_close(
        selection.compressed_block_indices,
        torch.tensor([[[-1], [0], [0], [1]]]),
    )


def test_sparse_attention_with_compressed_entries_returns_expected_shape() -> None:
    query = torch.randn(1, 4, 2, 3)
    token_kv = torch.randn(1, 4, 3)
    compressed_kv = torch.randn(1, 2, 3)
    selection = SparseSelection(
        compressed_block_indices=torch.tensor([[[-1], [0], [0], [1]]]),
        local_token_indices=torch.tensor([[-1, 0], [0, 1], [1, 2], [2, 3]]),
    )

    output, weights = sparse_attention_with_compressed_entries(
        query,
        token_kv,
        compressed_kv,
        selection,
    )

    assert output.shape == query.shape
    assert weights.shape == (1, 4, 2, 3)
    torch.testing.assert_close(weights.sum(dim=-1), torch.ones(1, 4, 2))
