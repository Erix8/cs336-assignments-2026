"""Run, profile, and save BPE training experiments."""

import argparse
import cProfile
import json
import os
import pstats
import sys
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter

from tqdm import tqdm

from cs336_basics.bpe_training import BPETrainer, Pair, count_pretokens

PROJECT_ROOT = Path(__file__).resolve().parents[1]
TOKENIZER_OUTPUT_DIR = PROJECT_ROOT / "data" / "tokenizers"


def _output_paths(filename: str) -> tuple[Path, Path]:
    return (
        TOKENIZER_OUTPUT_DIR / f"{filename}_train_vocab.json",
        TOKENIZER_OUTPUT_DIR / f"{filename}_train_merges.json",
    )


def _check_output_paths(filename: str) -> None:
    for path in _output_paths(filename):
        if path.exists():
            raise FileExistsError(f"Tokenizer output already exists: {path}")


@dataclass(frozen=True, slots=True)
class BPETrainingStats:
    vocab_size: int
    num_merges: int
    num_unique_pretokens: int
    pretokenization_seconds: float
    initialization_seconds: float
    merging_seconds: float
    total_seconds: float


def print_training_stats(stats: BPETrainingStats) -> None:
    print(f"vocab size: {stats.vocab_size:,}")
    print(f"merges: {stats.num_merges:,}")
    print(f"unique pre-tokens: {stats.num_unique_pretokens:,}")
    print(f"pre-tokenization: {stats.pretokenization_seconds:.2f}s")
    print(f"initialization: {stats.initialization_seconds:.2f}s")
    print(f"merging: {stats.merging_seconds:.2f}s")
    print(f"training total: {stats.total_seconds:.2f}s")


def run_bpe_experiment(
    name: str,
    input_path: str | os.PathLike[str],
    vocab_size: int,
    special_tokens: list[str],
    split_special_token: str,
    profile_merges: bool = False,
    output_name: str | None = None,
) -> BPETrainingStats:
    """Train one corpus, report timings, and save a new tokenizer."""

    if split_special_token not in special_tokens:
        raise ValueError("split_special_token must be one of special_tokens")
    output_name = output_name or name
    _check_output_paths(output_name)
    total_start = perf_counter()

    print(f"Pre-tokenizing {name}...", flush=True)
    pretokenization_start = perf_counter()
    freq_table = count_pretokens(input_path, split_special_token, special_tokens, show_progress=True)
    pretokenization_seconds = perf_counter() - pretokenization_start
    num_unique_pretokens = len(freq_table)

    initialization_start = perf_counter()
    trainer = BPETrainer(freq_table, special_tokens)
    initialization_seconds = perf_counter() - initialization_start
    del freq_table

    print(f"Training {name} BPE merges...", flush=True)
    merging_start = perf_counter()
    if profile_merges:
        profiler = cProfile.Profile()
        vocab, merges = profiler.runcall(trainer.train, vocab_size)
        pstats.Stats(profiler).sort_stats("cumulative").print_stats(20)
    else:
        merges_remaining = max(0, vocab_size - len(trainer.vocab))
        with tqdm(
            total=merges_remaining,
            desc=f"{name} BPE merges",
            unit="merge",
            disable=not sys.stderr.isatty(),
        ) as progress:
            vocab, merges = trainer.train(vocab_size, progress.update)
    merging_seconds = perf_counter() - merging_start

    total_seconds = perf_counter() - total_start

    stats = BPETrainingStats(
        vocab_size=len(vocab),
        num_merges=len(merges),
        num_unique_pretokens=num_unique_pretokens,
        pretokenization_seconds=pretokenization_seconds,
        initialization_seconds=initialization_seconds,
        merging_seconds=merging_seconds,
        total_seconds=total_seconds,
    )

    print_training_stats(stats)
    save_vocab_merges(output_name, vocab, merges)
    print_longest_tokens(vocab, special_tokens)

    return stats


def save_vocab_merges(
    filename: str,
    vocab: dict[int, bytes],
    merges: list[Pair],
) -> None:
    """Save the hex JSON format consumed by Tokenizer.from_files."""
    TOKENIZER_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    _check_output_paths(filename)
    output_vocab, output_merges = _output_paths(filename)

    # Exclusive creation protects existing artifacts even after the early check.
    with open(output_vocab, "x", encoding="utf-8") as f:
        vocab_dict = {str(token_id): token.hex() for token_id, token in vocab.items()}
        json.dump(vocab_dict, f, indent=2)

    with open(output_merges, "x", encoding="utf-8") as f:
        merges_list = [[left.hex(), right.hex()] for left, right in merges]
        json.dump(merges_list, f, indent=2)


def print_longest_tokens(vocab: dict[int, bytes], special_tokens: list[str]) -> None:
    special_token_bytes = {token.encode("utf-8") for token in special_tokens}
    ordinary_tokens = [(token_id, token) for token_id, token in vocab.items() if token not in special_token_bytes]
    longest_length = max(len(token) for _, token in ordinary_tokens)
    longest_tokens = [(token_id, token) for token_id, token in ordinary_tokens if len(token) == longest_length]

    print(f"longest token length: {longest_length} bytes")

    for token_id, token in longest_tokens[:10]:
        readable = token.decode("utf-8", errors="backslashreplace")
        print(f"longest token: id={token_id}, bytes={token!r}, text={readable!r}")


def train_bpe_tinystories(
    vocab_size: int = 10_000,
    profile_merges: bool = False,
) -> BPETrainingStats:
    return run_bpe_experiment(
        name="tinystories",
        input_path=PROJECT_ROOT / "data" / "TinyStoriesV2-GPT4-train.txt",
        vocab_size=vocab_size,
        special_tokens=["<|endoftext|>"],
        split_special_token="<|endoftext|>",
        profile_merges=profile_merges,
        output_name=f"tinystories_{vocab_size}" if vocab_size != 10_000 else None,
    )


def train_bpe_owt(
    vocab_size: int = 32_000,
    profile_merges: bool = False,
) -> BPETrainingStats:
    return run_bpe_experiment(
        name="owt",
        input_path=PROJECT_ROOT / "data" / "owt_train.txt",
        vocab_size=vocab_size,
        special_tokens=["<|endoftext|>"],
        split_special_token="<|endoftext|>",
        profile_merges=profile_merges,
        output_name=f"owt_{vocab_size}" if vocab_size != 32_000 else None,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", choices=("tinystories", "owt"))
    parser.add_argument("--vocab-size", type=int, help="Override the dataset's default vocabulary size")
    parser.add_argument("--profile-merges", action="store_true", help="Profile BPE merging instead of showing progress")

    args = parser.parse_args()

    if args.vocab_size is not None and args.vocab_size < 257:
        parser.error("--vocab-size must allow one special token and all 256 byte tokens")

    if args.dataset == "tinystories":
        train_bpe_tinystories(
            vocab_size=args.vocab_size if args.vocab_size is not None else 10_000,
            profile_merges=args.profile_merges,
        )
    else:
        train_bpe_owt(
            vocab_size=args.vocab_size if args.vocab_size is not None else 32_000,
            profile_merges=args.profile_merges,
        )


if __name__ == "__main__":
    main()
