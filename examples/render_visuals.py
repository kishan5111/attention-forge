"""Render static visual assets for the project README and docs."""

from __future__ import annotations

import argparse
from pathlib import Path

from attention_forge.viz.attention_maps import render_attention_patterns
from attention_forge.viz.memory_plots import render_memory_plot, render_sparse_reads_plot


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, default=Path("assets/images"))
    args = parser.parse_args()

    memory_path = render_memory_plot(args.out_dir / "kv_memory_comparison.png")
    reads_path = render_sparse_reads_plot(args.out_dir / "sparse_decode_reads.png")
    patterns_path = render_attention_patterns(args.out_dir / "attention_patterns.png")

    print(f"Wrote {memory_path}")
    print(f"Wrote {reads_path}")
    print(f"Wrote {patterns_path}")


if __name__ == "__main__":
    main()
