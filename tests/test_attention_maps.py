import numpy as np

from attention_forge.viz.attention_maps import (
    block_selected_mask,
    dense_causal_mask,
    sink_local_mask,
    sliding_window_mask,
)


def test_dense_causal_mask_blocks_future_positions() -> None:
    mask = dense_causal_mask(4)

    np.testing.assert_array_equal(
        mask,
        np.array(
            [
                [1, 0, 0, 0],
                [1, 1, 0, 0],
                [1, 1, 1, 0],
                [1, 1, 1, 1],
            ],
            dtype=np.uint8,
        ),
    )


def test_sliding_window_mask_keeps_recent_tokens_only() -> None:
    mask = sliding_window_mask(sequence_length=5, window_size=2)

    np.testing.assert_array_equal(mask[4], np.array([0, 0, 0, 1, 1], dtype=np.uint8))


def test_sink_local_mask_keeps_sink_tokens_visible() -> None:
    mask = sink_local_mask(sequence_length=8, num_sink_tokens=2, window_size=2)

    np.testing.assert_array_equal(mask[7], np.array([1, 1, 0, 0, 0, 0, 1, 1], dtype=np.uint8))


def test_block_selected_mask_adds_selected_past_blocks() -> None:
    mask = block_selected_mask(
        sequence_length=16,
        block_size=4,
        selected_blocks=[1],
        local_window=2,
    )

    np.testing.assert_array_equal(
        mask[12],
        np.array([0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 1, 1, 0, 0, 0], dtype=np.uint8),
    )
