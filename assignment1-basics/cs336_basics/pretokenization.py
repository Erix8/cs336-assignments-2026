"""Convert a text corpus into pre-token frequencies for BPE training."""

import cProfile
import os
import pstats
from collections import Counter
from itertools import pairwise
from multiprocessing import Pool
from typing import BinaryIO

import regex as re

type Word = tuple[bytes, ...]
type PretokenizationTask = tuple[str | os.PathLike[str], int, int, list[str]]
MAX_WORKERS = 6
TARGET_CHUNK_SIZE_BYTES = 128 * 1024**2
PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""


def find_chunk_boundaries(file: BinaryIO, desired_num_chunks: int, split_special_token: bytes) -> list[int]:
    """Find byte offsets that allow corpus chunks to be counted independently.

    Interior boundaries are moved forward to the next occurrence of
    ``split_special_token`` so that no pre-token spans two chunks. Duplicate
    boundaries are removed, so the result can contain fewer chunks than
    requested.

    Args:
        file: Binary corpus positioned anywhere in the file.
        desired_num_chunks: Target number of approximately equal chunks.
        split_special_token: Byte sequence that is safe to split before.

    Returns:
        Sorted byte offsets containing both the start and end of the file.
    """
    assert isinstance(split_special_token, bytes), "Must represent special token as a bytestring"

    file.seek(0, os.SEEK_END)
    file_size = file.tell()
    file.seek(0)

    chunk_size = file_size // desired_num_chunks

    chunk_boundaries = [i * chunk_size for i in range(desired_num_chunks + 1)]
    chunk_boundaries[-1] = file_size

    mini_chunk_size = 4096

    for bi in range(1, len(chunk_boundaries) - 1):
        initial_position = chunk_boundaries[bi]
        file.seek(initial_position)
        while True:
            mini_chunk = file.read(mini_chunk_size)

            if mini_chunk == b"":
                chunk_boundaries[bi] = file_size
                break

            found_at = mini_chunk.find(split_special_token)
            if found_at != -1:
                chunk_boundaries[bi] = initial_position + found_at
                break
            initial_position += mini_chunk_size

    return sorted(set(chunk_boundaries))


def split_on_special_tokens(chunk: str, special_tokens: list[str]) -> list[str]:
    """Split text around special tokens without including them in the result."""
    if not special_tokens:
        return [chunk]
    # Prefer the longest match when one special token is a prefix of another.
    escaped = [re.escape(token) for token in sorted(special_tokens, key=len, reverse=True)]
    return re.split("|".join(escaped), chunk)


def pretokenize_chunk(
    input_path: str | os.PathLike[str], start: int, end: int, special_tokens: list[str]
) -> Counter[Word]:
    """Count byte-level pre-tokens within one half-open byte range.

    Each pre-token is represented as a tuple of single-byte ``bytes`` objects,
    which is the initial symbol sequence consumed by BPE training.
    """
    freq_table = Counter()
    with open(input_path, "rb") as f:
        f.seek(start)
        chunk = f.read(end - start).decode("utf-8", errors="ignore")
    for segment in split_on_special_tokens(chunk, special_tokens):
        for match in re.finditer(PAT, segment):
            encoded = match.group(0).encode("utf-8")
            key = tuple(bytes([byte]) for byte in encoded)
            freq_table[key] += 1
    return freq_table


def _pretokenize_chunk_from_args(args: PretokenizationTask) -> Counter[Word]:
    return pretokenize_chunk(*args)


def pretokenize(
    input_path: str | os.PathLike[str], split_special_token: str, special_tokens: list[str]
) -> dict[Word, int]:
    """Count pre-token frequencies across a corpus in parallel.

    Args:
        input_path: Path to a UTF-8 encoded text corpus.
        split_special_token: Token used to choose independent chunk boundaries.
        special_tokens: Tokens excluded from ordinary pre-token counts.

    Returns:
        freq_table: A mapping from byte-level pre-token sequences to corpus frequencies.
    """
    workers = min(os.cpu_count() or 1, MAX_WORKERS)
    file_size = os.path.getsize(input_path)
    if file_size == 0:
        return {}

    desired_num_chunks = max(
        workers,
        (file_size + TARGET_CHUNK_SIZE_BYTES - 1) // TARGET_CHUNK_SIZE_BYTES,
    )
    with open(input_path, "rb") as f:
        boundaries = find_chunk_boundaries(f, desired_num_chunks, split_special_token.encode("utf-8"))

    tasks = [(input_path, start, end, special_tokens) for start, end in pairwise(boundaries)]
    if len(tasks) == 1:
        return pretokenize_chunk(*tasks[0])

    freq_table = Counter()
    with Pool(processes=min(workers, len(tasks))) as pool:
        local_freq_tables = pool.imap_unordered(
            _pretokenize_chunk_from_args,
            tasks,
            chunksize=1,
        )
        for local_freq_table in local_freq_tables:
            freq_table.update(local_freq_table)
    return freq_table


if __name__ == "__main__":
    profiler = cProfile.Profile()
    profiler.enable()
    pretokenize("tests/fixtures/corpus.en", "<|endoftext|>", ["<|endoftext|>"])
    profiler.disable()
    stats = pstats.Stats(profiler)
    stats.sort_stats("cumulative")
    stats.print_stats(5)
