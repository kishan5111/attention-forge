"""Attention mask generators and visualizations.

Rows are query positions. Columns are key/value positions. A value of 1 means
the query is allowed to attend to that key/value position.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap


def dense_causal_mask(sequence_length: int) -> np.ndarray:
    """Return a dense causal attention mask."""

    _validate_positive("sequence_length", sequence_length)
    return np.tril(np.ones((sequence_length, sequence_length), dtype=np.uint8))


def sliding_window_mask(sequence_length: int, window_size: int) -> np.ndarray:
    """Return a causal local sliding-window attention mask."""

    _validate_positive("sequence_length", sequence_length)
    _validate_positive("window_size", window_size)

    row = np.arange(sequence_length)[:, None]
    col = np.arange(sequence_length)[None, :]
    return ((col <= row) & (col >= row - window_size + 1)).astype(np.uint8)


def sink_local_mask(sequence_length: int, num_sink_tokens: int, window_size: int) -> np.ndarray:
    """Return a causal mask with global sink tokens plus local window attention."""

    _validate_positive("sequence_length", sequence_length)
    _validate_positive("window_size", window_size)
    if num_sink_tokens < 0:
        raise ValueError(f"num_sink_tokens must be non-negative, got {num_sink_tokens}.")

    mask = sliding_window_mask(sequence_length, window_size)
    if num_sink_tokens == 0:
        return mask

    for sink_col in range(min(num_sink_tokens, sequence_length)):
        mask[sink_col:, sink_col] = 1
    return mask


def block_selected_mask(
    sequence_length: int,
    block_size: int,
    selected_blocks: list[int],
    local_window: int,
    num_sink_tokens: int = 0,
) -> np.ndarray:
    """Return a simplified DSA/CSA-style selected-block causal mask.

    This educational mask uses the same selected block ids for every query,
    plus a local window and optional sink tokens. A real indexer would select
    blocks per query.
    """

    _validate_positive("sequence_length", sequence_length)
    _validate_positive("block_size", block_size)
    _validate_positive("local_window", local_window)

    mask = sink_local_mask(sequence_length, num_sink_tokens, local_window)
    row = np.arange(sequence_length)[:, None]
    col = np.arange(sequence_length)[None, :]

    for block_id in selected_blocks:
        if block_id < 0:
            raise ValueError(f"selected block ids must be non-negative, got {block_id}.")
        start = block_id * block_size
        end = min(start + block_size, sequence_length)
        if start >= sequence_length:
            continue
        in_block = (col >= start) & (col < end)
        mask[(col <= row) & in_block] = 1

    return mask


def render_attention_patterns(output_path: Path) -> Path:
    """Render the first educational attention pattern panel."""

    sequence_length = 128
    masks = [
        ("Dense causal", dense_causal_mask(sequence_length)),
        ("Sliding window", sliding_window_mask(sequence_length, window_size=24)),
        ("Sink + local", sink_local_mask(sequence_length, num_sink_tokens=4, window_size=24)),
        (
            "Selected blocks + local",
            block_selected_mask(
                sequence_length,
                block_size=16,
                selected_blocks=[1, 4, 6],
                local_window=24,
                num_sink_tokens=4,
            ),
        ),
    ]

    fig, axes = plt.subplots(1, len(masks), figsize=(13, 4), constrained_layout=True)
    cmap = ListedColormap(["#f2f2f2", "#1f77b4"])

    for ax, (title, mask) in zip(axes, masks, strict=True):
        ax.imshow(mask, origin="upper", interpolation="nearest", cmap=cmap, vmin=0, vmax=1)
        ax.set_title(title)
        ax.set_xlabel("KV position")
        ax.set_ylabel("Query position")
        ax.set_xticks([])
        ax.set_yticks([])

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180)
    plt.close(fig)
    return output_path


def _validate_positive(name: str, value: int) -> None:
    if value <= 0:
        raise ValueError(f"{name} must be positive, got {value}.")
