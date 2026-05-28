"""Reference attention implementations."""

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

__all__ = [
    "SparseSelection",
    "build_sparse_selection",
    "compressed_block_visibility",
    "gated_block_compress",
    "lightning_index_scores",
    "local_token_indices",
    "select_topk_compressed_blocks",
    "sparse_attention_with_compressed_entries",
]
