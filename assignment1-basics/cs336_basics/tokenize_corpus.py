"""Stream a corpus into a raw uint16 token array for memory-mapped training."""

import argparse
import sys
from collections.abc import Iterator
from itertools import islice
from pathlib import Path
from time import perf_counter

import numpy as np
from tqdm import tqdm

from cs336_basics.tokenizer import Tokenizer

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
SPECIAL_TOKEN = "<|endoftext|>"
BATCH_SIZE = 1_000_000
EDGE_TOKENS = 1_024
CORPORA = {
    "tinystories": {
        "train": "TinyStoriesV2-GPT4-train.txt",
        "valid": "TinyStoriesV2-GPT4-valid.txt",
    },
    "owt": {
        "train": "owt_train.txt",
        "valid": "owt_valid.txt",
    },
}


def load_tokenizer(dataset: str) -> Tokenizer:
    tokenizer_dir = DATA_DIR / "tokenizers"
    return Tokenizer.from_files(
        str(tokenizer_dir / f"{dataset}_train_vocab.json"),
        str(tokenizer_dir / f"{dataset}_train_merges.json"),
        [SPECIAL_TOKEN],
    )


def verify_edges(source_path: Path, token_path: Path, tokenizer: Tokenizer, token_count: int) -> None:
    """Check that the saved stream preserves the source bytes at both ends."""
    saved = np.memmap(token_path, dtype=np.uint16, mode="r")
    if len(saved) != token_count:
        raise ValueError(f"Unexpected token count in {token_path}")

    edge_length = min(EDGE_TOKENS, token_count)
    prefix = b"".join(tokenizer.vocab[int(token_id)] for token_id in saved[:edge_length])
    suffix = b"".join(tokenizer.vocab[int(token_id)] for token_id in saved[-edge_length:])
    with source_path.open("rb") as source:
        if source.read(len(prefix)) != prefix:
            raise ValueError("Token stream does not match the start of the source corpus")
        source.seek(-len(suffix), 2)
        if source.read(len(suffix)) != suffix:
            raise ValueError("Token stream does not match the end of the source corpus")
    del saved


def tokenize_corpus(dataset: str, split: str, output_dir: Path, batch_size: int) -> Path:
    source_path = DATA_DIR / CORPORA[dataset][split]
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{dataset}_{split}_uint16.bin"
    partial_path = output_dir / f"{dataset}_{split}_uint16.bin.partial"
    if output_path.exists() or partial_path.exists():
        raise FileExistsError(f"Output or partial output already exists: {output_path}")

    tokenizer = load_tokenizer(dataset)
    if max(tokenizer.vocab) > np.iinfo(np.uint16).max:
        raise ValueError("Token IDs exceed the uint16 range")

    token_count = 0
    start = perf_counter()
    with (
        source_path.open("rb") as source,
        partial_path.open("xb") as output,
        tqdm(
            total=source_path.stat().st_size,
            desc=f"{dataset}-{split}",
            unit="B",
            unit_scale=True,
            unit_divisor=1024,
            disable=not sys.stderr.isatty(),
        ) as progress,
    ):
        def decoded_lines() -> Iterator[str]:
            for raw_line in source:
                progress.update(len(raw_line))
                yield raw_line.decode("utf-8")

        ids = tokenizer.encode_iterable(decoded_lines())
        while True:
            batch = np.fromiter(islice(ids, batch_size), dtype=np.uint16)
            if batch.size == 0:
                break
            batch.tofile(output)
            token_count += batch.size
            progress.set_postfix_str(f"{token_count:,} tokens", refresh=False)

    if token_count == 0:
        raise ValueError(f"No tokens encoded from {source_path}")
    verify_edges(source_path, partial_path, tokenizer, token_count)
    partial_path.rename(output_path)

    elapsed = perf_counter() - start
    print(
        f"Saved {token_count:,} uint16 IDs ({output_path.stat().st_size:,} bytes) "
        f"from {source_path.name} in {elapsed:.1f} seconds to {output_path}",
        flush=True,
    )
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", choices=CORPORA)
    parser.add_argument("split", choices=("train", "valid"))
    parser.add_argument("--output-dir", type=Path, default=DATA_DIR / "tokenized")
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    args = parser.parse_args()
    if args.batch_size <= 0:
        parser.error("--batch-size must be positive")
    tokenize_corpus(args.dataset, args.split, args.output_dir, args.batch_size)


if __name__ == "__main__":
    main()
