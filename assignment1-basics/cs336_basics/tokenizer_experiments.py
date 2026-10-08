"""Measure tokenizer compression on reproducible document samples."""

import json
import random
from collections.abc import Iterator
from pathlib import Path
from time import perf_counter

from cs336_basics.tokenizer import Tokenizer

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
SAMPLE_DIR = DATA_DIR / "tokenizer_samples"
SPECIAL_TOKEN = "<|endoftext|>"
SAMPLE_SIZE = 10
SAMPLE_SEED = 336
READ_SIZE = 1024 * 1024
BENCHMARK_SECONDS = 3.0
PILE_SIZE_BYTES = 825_000_000_000


def iter_documents(path: Path) -> Iterator[str]:
    """Read documents without loading an entire training corpus into memory."""
    pending = ""
    with path.open(encoding="utf-8", newline="") as corpus:
        while chunk := corpus.read(READ_SIZE):
            parts = (pending + chunk).split(SPECIAL_TOKEN)
            yield from (document for document in parts[:-1] if document)
            pending = parts[-1]
    if pending:
        yield pending


def sample_documents(path: Path) -> tuple[list[str], int]:
    """Give each document an equal chance of entering the fixed-size sample."""
    rng = random.Random(SAMPLE_SEED)
    sample: list[str] = []
    document_count = 0
    for document_count, document in enumerate(iter_documents(path), start=1):
        if len(sample) < SAMPLE_SIZE:
            sample.append(document)
        else:
            index = rng.randrange(document_count)
            if index < SAMPLE_SIZE:
                sample[index] = document
    if len(sample) != SAMPLE_SIZE:
        raise ValueError(f"Expected at least {SAMPLE_SIZE} documents in {path}")
    return sample, document_count


def load_or_sample(name: str, corpus_path: Path) -> list[str]:
    cache_path = SAMPLE_DIR / f"{name}_seed{SAMPLE_SEED}_n{SAMPLE_SIZE}.json"
    if cache_path.exists():
        with cache_path.open(encoding="utf-8") as sample_file:
            saved = json.load(sample_file)
        documents = saved["documents"]
        if saved["seed"] != SAMPLE_SEED or len(documents) != SAMPLE_SIZE:
            raise ValueError(f"Invalid sample cache: {cache_path}")
        print(f"Loaded {name} sample from {cache_path}")
        return documents

    documents, document_count = sample_documents(corpus_path)
    SAMPLE_DIR.mkdir(parents=True, exist_ok=True)
    with cache_path.open("w", encoding="utf-8", newline="") as sample_file:
        json.dump(
            {
                "source": corpus_path.name,
                "source_size_bytes": corpus_path.stat().st_size,
                "seed": SAMPLE_SEED,
                "document_count": document_count,
                "documents": documents,
            },
            sample_file,
            ensure_ascii=False,
            indent=2,
        )
    print(f"Saved {name} sample to {cache_path}")
    return documents


def compression_stats(documents: list[str], tokenizer: Tokenizer) -> tuple[int, int, float]:
    total_bytes = sum(len(document.encode("utf-8")) for document in documents)
    total_tokens = sum(len(tokenizer.encode(document)) for document in documents)
    if total_tokens == 0:
        raise ValueError("Cannot compute bytes per token from an empty sample")
    return total_bytes, total_tokens, total_bytes / total_tokens


def benchmark_throughput(documents: list[str], tokenizer: Tokenizer) -> tuple[float, float, int]:
    sample_bytes = sum(len(document.encode("utf-8")) for document in documents)
    if sample_bytes == 0:
        raise ValueError("Cannot benchmark an empty sample")

    for document in documents:
        tokenizer.encode(document)

    start = perf_counter()
    rounds = 0
    while True:
        for document in documents:
            tokenizer.encode(document)
        rounds += 1
        elapsed = perf_counter() - start
        if elapsed >= BENCHMARK_SECONDS:
            break

    return sample_bytes * rounds / elapsed, elapsed, rounds


def main() -> None:
    tinystories_documents = load_or_sample("tinystories", DATA_DIR / "TinyStoriesV2-GPT4-train.txt")
    owt_documents = load_or_sample("owt", DATA_DIR / "owt_train.txt")

    tinystories_tokenizer = Tokenizer.from_files(
        str(DATA_DIR / "tokenizers" / "tinystories_train_vocab.json"),
        str(DATA_DIR / "tokenizers" / "tinystories_train_merges.json"),
        [SPECIAL_TOKEN],
    )
    owt_tokenizer = Tokenizer.from_files(
        str(DATA_DIR / "tokenizers" / "owt_train_vocab.json"),
        str(DATA_DIR / "tokenizers" / "owt_train_merges.json"),
        [SPECIAL_TOKEN],
    )

    comparisons = (
        ("TinyStories with TinyStories tokenizer", tinystories_documents, tinystories_tokenizer),
        ("OWT with OWT tokenizer", owt_documents, owt_tokenizer),
        ("OWT with TinyStories tokenizer", owt_documents, tinystories_tokenizer),
    )
    for label, documents, tokenizer in comparisons:
        total_bytes, total_tokens, ratio = compression_stats(documents, tokenizer)
        print(f"{label}: {ratio:.3f} bytes/token ({total_bytes:,} bytes / {total_tokens:,} tokens)")

    throughput, elapsed, rounds = benchmark_throughput(owt_documents, owt_tokenizer)
    pile_hours = PILE_SIZE_BYTES / throughput / 3600
    print(f"OWT tokenizer throughput: {throughput:,.0f} bytes/second ({rounds} passes in {elapsed:.2f} seconds)")
    print(f"Estimated Pile time (825 GB): {pile_hours:,.1f} hours ({pile_hours / 24:,.1f} days)")


if __name__ == "__main__":
    main()
