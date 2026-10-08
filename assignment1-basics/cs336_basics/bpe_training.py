"""Pre-tokenize a corpus and train a byte-level BPE vocabulary."""

import os
import sys
from collections import Counter, defaultdict
from collections.abc import Callable
from itertools import pairwise
from multiprocessing import Pool
from typing import BinaryIO

import regex as re
from tqdm import tqdm

type Pair = tuple[bytes, bytes]
type Word = tuple[bytes, ...]
type PretokenizationTask = tuple[str | os.PathLike[str], int, int, list[str]]
MAX_WORKERS = 6
TARGET_CHUNK_SIZE_BYTES = 128 * 1024**2
PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""


def find_chunk_boundaries(file: BinaryIO, desired_num_chunks: int, split_special_token: bytes) -> list[int]:
    """Align chunk boundaries with a special token so pre-tokens cannot cross them.

    Duplicate boundaries are removed, so fewer chunks may be returned.
    """
    if not isinstance(split_special_token, bytes):
        raise TypeError("split_special_token must be bytes")

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
    if not special_tokens:
        return [chunk]
    # Prefer the longest match when one special token is a prefix of another.
    escaped = [re.escape(token) for token in sorted(special_tokens, key=len, reverse=True)]
    return re.split("|".join(escaped), chunk)


def count_pretokens_in_chunk(
    input_path: str | os.PathLike[str], start: int, end: int, special_tokens: list[str]
) -> Counter[Word]:
    """Count pre-tokens in a byte range, representing each as single-byte symbols.

    This representation lets BPE start with all 256 possible byte tokens.
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


def _count_pretokens_in_chunk_from_args(args: PretokenizationTask) -> Counter[Word]:
    # Keep the worker at module scope so multiprocessing can pickle it.
    return count_pretokens_in_chunk(*args)


def count_pretokens(
    input_path: str | os.PathLike[str],
    split_special_token: str,
    special_tokens: list[str],
    show_progress: bool = False,
) -> dict[Word, int]:
    """Count corpus pre-tokens in parallel without splitting across documents.

    Special tokens are excluded from the ordinary pre-token counts.
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
    with tqdm(
        total=len(tasks),
        desc="Pre-tokenizing",
        unit="chunk",
        disable=not show_progress or not sys.stderr.isatty(),
    ) as progress:
        if len(tasks) == 1:
            freq_table = count_pretokens_in_chunk(*tasks[0])
            progress.update(1)
            return freq_table

        freq_table = Counter()
        with Pool(processes=min(workers, len(tasks))) as pool:
            local_freq_tables = pool.imap_unordered(
                _count_pretokens_in_chunk_from_args,
                tasks,
                chunksize=1,
            )
            for local_freq_table in local_freq_tables:
                freq_table.update(local_freq_table)
                progress.update(1)
        return freq_table


def _init_vocab(special_tokens: list[str]) -> dict[int, bytes]:
    vocab = {}
    for i, token in enumerate(special_tokens):
        vocab[i] = token.encode("utf-8")

    for i in range(256):
        vocab[len(special_tokens) + i] = bytes([i])

    return vocab


def _build_word_table(freq_table: dict[Word, int]) -> tuple[list[list[bytes]], list[int]]:
    words, freqs = [], []
    for word, count in freq_table.items():
        words.append(list(word))
        freqs.append(count)
    return words, freqs


def _build_pair_stats(
    words: list[list[bytes]], freqs: list[int]
) -> tuple[dict[Pair, int], defaultdict[Pair, set[int]]]:
    """Build pair frequencies and an index of the words containing each pair."""
    pair_counts = Counter()
    pair_to_words = defaultdict(set)
    for idx, (word, freq) in enumerate(zip(words, freqs, strict=True)):
        for pair in pairwise(word):
            pair_counts[pair] += freq
            pair_to_words[pair].add(idx)

    return pair_counts, pair_to_words


def _merge_pair_in_word(word: list[bytes], pair: Pair) -> list[bytes]:
    new_word = []
    i = 0
    while i < len(word):
        if i + 1 < len(word) and (word[i], word[i + 1]) == pair:
            new_word.append(word[i] + word[i + 1])
            i += 2
        else:
            new_word.append(word[i])
            i += 1
    return new_word


class BPETrainer:
    """Track pair counts and affected words for incremental BPE merges."""

    def __init__(self, freq_table: dict[Word, int], special_tokens: list[str]) -> None:
        self.words, self.freqs = _build_word_table(freq_table)
        self.pair_counts, self.pair_to_words = _build_pair_stats(self.words, self.freqs)
        self.vocab = _init_vocab(special_tokens)
        self.merges: list[Pair] = []

    def _apply_merge(self, pair: Pair) -> None:
        """Apply one selected pair and keep all derived statistics consistent."""
        # The index is mutated below, so iterate over a stable snapshot.
        affected_ids = list(self.pair_to_words[pair])
        for word_id in affected_ids:
            old_word, freq = self.words[word_id], self.freqs[word_id]

            for old_pair in pairwise(old_word):
                self.pair_counts[old_pair] -= freq
                if self.pair_counts[old_pair] == 0:
                    # A zero-count pair must not be selected for a later merge.
                    del self.pair_counts[old_pair]
                self.pair_to_words[old_pair].discard(word_id)

            new_word = _merge_pair_in_word(old_word, pair)
            self.words[word_id] = new_word
            # Re-index every pair because a merge changes neighboring pairs too.
            for new_pair in pairwise(new_word):
                self.pair_counts[new_pair] = self.pair_counts.get(new_pair, 0) + freq
                self.pair_to_words[new_pair].add(word_id)

        left, right = pair
        self.vocab[len(self.vocab)] = left + right
        self.merges.append(pair)

    def train(
        self, vocab_size: int, on_merge: Callable[[int], object] | None = None
    ) -> tuple[dict[int, bytes], list[Pair]]:
        """Continue training until the vocabulary reaches ``vocab_size``.

        Returned containers are copies so a later call that continues training
        does not mutate results already received by the caller.
        """
        while len(self.vocab) < vocab_size and self.pair_counts:
            # Pair order breaks frequency ties deterministically.
            best_pair = max(self.pair_counts, key=lambda pair: (self.pair_counts[pair], pair))
            self._apply_merge(best_pair)
            if on_merge is not None:
                on_merge(1)

        return self.vocab.copy(), self.merges.copy()


def train_bpe(
    input_path: str | os.PathLike,
    vocab_size: int,
    special_tokens: list[str],
    split_special_token: str = "<|endoftext|>",
) -> tuple[dict[int, bytes], list[Pair]]:
    """Train byte-level BPE and return the vocabulary and ranked merges.

    The requested size includes special tokens and the 256 initial byte tokens.
    """
    if split_special_token not in special_tokens:
        raise ValueError("split_special_token must be one of special_tokens")

    freq_table = count_pretokens(input_path, split_special_token, special_tokens)
    trainer = BPETrainer(freq_table, special_tokens)
    del freq_table

    return trainer.train(vocab_size)
