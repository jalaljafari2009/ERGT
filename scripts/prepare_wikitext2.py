"""Prepare WikiText-2 token blocks for ERGT experiments."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from experiments.data_utils import prepare_wikitext2  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare WikiText-2 for ERGT experiments.")
    parser.add_argument(
        "--output-dir",
        default="data/processed/wikitext2_gpt2_ctx256",
        help="Directory where token blocks and metadata will be saved.",
    )
    parser.add_argument(
        "--tokenizer",
        default="gpt2",
        help="Tokenizer name or local tokenizer path.",
    )
    parser.add_argument(
        "--context-length",
        type=int,
        default=256,
        help="Input sequence length. Saved blocks contain context_length + 1 tokens.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    metadata = prepare_wikitext2(
        output_dir=Path(args.output_dir),
        tokenizer_name=args.tokenizer,
        context_length=args.context_length,
    )
    print(
        "Prepared WikiText-2: "
        f"{metadata.train_sequences} train sequences, "
        f"{metadata.validation_sequences} validation sequences, "
        f"context_length={metadata.context_length}, tokenizer={metadata.tokenizer}"
    )


if __name__ == "__main__":
    main()
