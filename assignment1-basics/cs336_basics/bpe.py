"""Train a byte pair encoding vocabulary from pre-token frequencies."""

import cProfile
import json
import os
import pstats
from collections import Counter, defaultdict
from dataclasses import dataclass
from itertools import pairwise
from pathlib import Path
from time import perf_counter

from cs336_basics.pretokenization import pretokenize

type Pair = tuple[bytes, bytes]
type Word = tuple[bytes, ...]
PROJECT_ROOT = Path(__file__).resolve().parents[1]
TOKENIZER_OUTPUT_DIR = PROJECT_ROOT / "data" / "tokenizers"


def _init_vocab(special_tokens: list[str]) -> dict[int, bytes]:
    """Create a vocabulary containing special tokens and all byte values."""
    vocab = {}
    for i, token in enumerate(special_tokens):
        vocab[i] = token.encode("utf-8")
    for i in range(256):
        vocab[len(special_tokens) + i] = bytes([i])
    return vocab


def _build_word_table(freq_table: dict[Word, int]) -> tuple[list[list[bytes]], list[int]]:
    """Convert the frequency mapping into parallel mutable training arrays."""
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
    """Replace non-overlapping occurrences of a pair in one word."""
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
    """Maintain the mutable state required for incremental BPE training."""

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
            # Remove the old word's contribution before replacing its symbols.
            for old_pair in pairwise(old_word):
                self.pair_counts[old_pair] -= freq
                if self.pair_counts[old_pair] == 0:
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

    def train(self, vocab_size: int) -> tuple[dict[int, bytes], list[Pair]]:
        """Continue training until the vocabulary reaches ``vocab_size``.

        Returned containers are copies so a later call that continues training
        does not mutate results already received by the caller.
        """
        while len(self.vocab) < vocab_size and self.pair_counts:
            best_pair = max(self.pair_counts, key=lambda pair: (self.pair_counts[pair], pair))
            self._apply_merge(best_pair)
        return self.vocab.copy(), self.merges.copy()


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


def train_bpe(
    input_path: str | os.PathLike,
    vocab_size: int,
    special_tokens: list[str],
    split_special_token: str = "<|endoftext|>",
) -> tuple[dict[int, bytes], list[Pair]]:
    """Train a BPE vocabulary and ordered merge list from a text corpus.

    Args:
        input_path: Path to the UTF-8 encoded training corpus.
        vocab_size: Desired size including special tokens and 256 byte tokens.
        special_tokens: Tokens inserted into the vocabulary without training.
        split_special_token: Token used to create independent corpus chunks.

    Returns:
        The token-ID vocabulary and merges in training order.
    """
    if split_special_token not in special_tokens:
        raise ValueError("split_special_token must be one of special_tokens")

    freq_table = pretokenize(input_path, split_special_token, special_tokens)
    trainer = BPETrainer(freq_table, special_tokens)
    del freq_table

    return trainer.train(vocab_size)


def run_bpe_experiment(
    name: str,
    input_path: str | os.PathLike[str],
    vocab_size: int,
    special_tokens: list[str],
    split_special_token: str,
    profile_merges: bool = False,
) -> BPETrainingStats:

    if split_special_token not in special_tokens:
        raise ValueError("split_special_token must be one of special_tokens")
    total_start = perf_counter()

    pretokenization_start = perf_counter()
    freq_table = pretokenize(input_path, split_special_token, special_tokens)
    pretokenization_seconds = perf_counter() - pretokenization_start
    num_unique_pretokens = len(freq_table)

    initialization_start = perf_counter()
    trainer = BPETrainer(freq_table, special_tokens)
    initialization_seconds = perf_counter() - initialization_start
    del freq_table

    merging_start = perf_counter()
    if profile_merges:
        profiler = cProfile.Profile()
        vocab, merges = profiler.runcall(trainer.train, vocab_size)
        pstats.Stats(profiler).sort_stats("cumulative").print_stats(20)
    else:
        vocab, merges = trainer.train(vocab_size)
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
    dump_vocab_merges(name, vocab, merges, special_tokens)

    return stats


def dump_vocab_merges(
    filename: str,
    vocab: dict[int, bytes],
    merges: list[Pair],
    special_tokens: list[str],
) -> None:
    TOKENIZER_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output_vocab = TOKENIZER_OUTPUT_DIR / f"{filename}_train_vocab.json"
    ouput_merges = TOKENIZER_OUTPUT_DIR / f"{filename}_train_merges.json"

    special_token_bytes = {token.encode("utf-8") for token in special_tokens}
    ordinary_tokens = [(token_id, token) for token_id, token in vocab.items() if token not in special_token_bytes]
    longest_length = max(len(token) for _, token in ordinary_tokens)
    longest_tokens = [(token_id, token) for token_id, token in ordinary_tokens if len(token) == longest_length]

    with open(output_vocab, "w", encoding="utf-8") as f:
        vocab_dict = {str(token_id): token.hex() for token_id, token in vocab.items()}
        json.dump(vocab_dict, f, indent=2)

    with open(ouput_merges, "w", encoding="utf-8") as f:
        merges_list = [[left.hex(), right.hex()] for left, right in merges]
        json.dump(merges_list, f, indent=2)
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
    )


def train_bpe_expts_owt(
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
    )

def main():

    ...

if __name__ == "__main__":
    # train_bpe_tinystories(vocab_size=10_000, profile_merges=False)
    train_bpe_expts_owt(vocab_size=32_000, profile_merges=False)
