"""Dataset and tokenization helpers shared by baseline and ERGT experiments."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import Dataset


@dataclass(frozen=True)
class PreparedDatasetMetadata:
    dataset_name: str
    tokenizer: str
    context_length: int
    vocab_size: int
    train_tokens: int
    validation_tokens: int
    train_sequences: int
    validation_sequences: int
    eos_token_id: int | None
    preprocessing_version: str = "phase0_v1"


class TokenBlockDataset(Dataset[tuple[torch.Tensor, torch.Tensor]]):
    """Next-token dataset from contiguous token blocks."""

    def __init__(self, token_blocks: torch.Tensor) -> None:
        if token_blocks.dim() != 2:
            raise ValueError("token_blocks must have shape [num_sequences, context_length + 1]")
        if token_blocks.size(1) < 2:
            raise ValueError("token_blocks must contain at least two tokens per sequence")
        self.token_blocks = token_blocks.to(dtype=torch.long)

    def __len__(self) -> int:
        return self.token_blocks.size(0)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        block = self.token_blocks[index]
        return block[:-1], block[1:]


def save_json(path: str | Path, payload: dict[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def load_json(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def load_tokenizer(tokenizer_name: str):
    """Load a tokenizer by name.

    Kept local to avoid making `transformers` a hard import for tests that only
    exercise tensor packing.
    """
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def load_wikitext2_texts() -> dict[str, list[str]]:
    """Load WikiText-2 raw text splits."""
    from datasets import load_dataset

    try:
        dataset = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1")
    except Exception:
        dataset = load_dataset("wikitext", "wikitext-2-raw-v1")
    return {
        "train": _filter_nonempty_texts(dataset["train"]["text"]),
        "validation": _filter_nonempty_texts(dataset["validation"]["text"]),
        "test": _filter_nonempty_texts(dataset["test"]["text"]),
    }


def _filter_nonempty_texts(texts: list[str]) -> list[str]:
    return [text for text in texts if text and text.strip()]


def tokenize_texts(texts: list[str], tokenizer_name: str) -> tuple[list[int], int | None]:
    tokenizer = load_tokenizer(tokenizer_name)
    eos_token_id = tokenizer.eos_token_id

    token_ids: list[int] = []
    for text in texts:
        encoded = tokenizer.encode(text, add_special_tokens=False)
        token_ids.extend(encoded)
        if eos_token_id is not None:
            token_ids.append(eos_token_id)

    return token_ids, eos_token_id


def pack_token_ids(token_ids: list[int], context_length: int) -> torch.Tensor:
    """Pack token IDs into fixed blocks of `context_length + 1`.

    Blocks include one extra token so each item can return input and next-token
    target sequences of length `context_length`.
    """
    if context_length <= 0:
        raise ValueError("context_length must be positive")

    block_size = context_length + 1
    usable_tokens = (len(token_ids) // block_size) * block_size
    if usable_tokens == 0:
        raise ValueError("not enough tokens to create one sequence block")

    tokens = torch.tensor(token_ids[:usable_tokens], dtype=torch.long)
    return tokens.view(-1, block_size).contiguous()


def save_prepared_blocks(
    output_dir: str | Path,
    train_blocks: torch.Tensor,
    validation_blocks: torch.Tensor,
    metadata: PreparedDatasetMetadata,
) -> None:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    torch.save(train_blocks, output_dir / "train_blocks.pt")
    torch.save(validation_blocks, output_dir / "validation_blocks.pt")
    save_json(output_dir / "metadata.json", asdict(metadata))


def load_prepared_datasets(
    data_dir: str | Path,
) -> tuple[TokenBlockDataset, TokenBlockDataset, dict[str, Any]]:
    data_dir = Path(data_dir)
    train_blocks = torch.load(data_dir / "train_blocks.pt", map_location="cpu")
    validation_blocks = torch.load(data_dir / "validation_blocks.pt", map_location="cpu")
    metadata = load_json(data_dir / "metadata.json")
    return TokenBlockDataset(train_blocks), TokenBlockDataset(validation_blocks), metadata


def prepare_wikitext2(
    output_dir: str | Path,
    tokenizer_name: str,
    context_length: int,
) -> PreparedDatasetMetadata:
    splits = load_wikitext2_texts()
    tokenizer = load_tokenizer(tokenizer_name)
    train_token_ids, eos_token_id = tokenize_texts(splits["train"], tokenizer_name)
    validation_token_ids, _ = tokenize_texts(splits["validation"], tokenizer_name)

    train_blocks = pack_token_ids(train_token_ids, context_length)
    validation_blocks = pack_token_ids(validation_token_ids, context_length)

    metadata = PreparedDatasetMetadata(
        dataset_name="wikitext-2",
        tokenizer=tokenizer_name,
        context_length=context_length,
        vocab_size=len(tokenizer),
        train_tokens=len(train_token_ids),
        validation_tokens=len(validation_token_ids),
        train_sequences=train_blocks.size(0),
        validation_sequences=validation_blocks.size(0),
        eos_token_id=eos_token_id,
    )
    save_prepared_blocks(output_dir, train_blocks, validation_blocks, metadata)
    return metadata
