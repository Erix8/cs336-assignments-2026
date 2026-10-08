import json
from collections.abc import Iterable, Iterator
from itertools import pairwise

import regex as re

type Pair = tuple[bytes, bytes]
type Word = tuple[bytes, ...]
PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""


class Tokenizer:
    """Encode and decode text with a byte-level BPE vocabulary."""

    def __init__(
        self,
        vocab: dict[int, bytes],
        merges: list[tuple[bytes, bytes]],
        special_tokens: list[str] | None = None,
    ) -> None:
        self.vocab: dict[int, bytes] = vocab.copy()
        self.special_token_to_id: dict[str, int] = {}
        # Reuse IDs when a trained vocabulary already contains the special tokens.
        for token in special_tokens or []:
            next_id = max(self.vocab, default=-1) + 1
            token_bytes = token.encode("utf-8")
            token_id = next(
                (id_ for id_, value in self.vocab.items() if value == token_bytes),
                None,
            )
            if token_id is None:
                token_id = next_id
                self.vocab[token_id] = token_bytes
            self.special_token_to_id[token] = token_id

        self.merge_ranks: dict[Pair, int] = {pair: rank for rank, pair in enumerate(merges)}
        self.bytes_to_id: dict[bytes, int] = {token_bytes: token_id for token_id, token_bytes in self.vocab.items()}

    @classmethod
    def from_files(
        cls,
        vocab_file: str,
        merges_file: str,
        special_tokens: list[str] | None = None,
    ) -> "Tokenizer":
        """Load the hex-encoded vocabulary and ordered merges saved by training."""
        with open(vocab_file, encoding="utf-8") as f:
            raw_vocab = json.load(f)
        vocab = {int(token_id): bytes.fromhex(value) for token_id, value in raw_vocab.items()}
        del raw_vocab

        with open(merges_file, encoding="utf-8") as f:
            raw_merges = json.load(f)
        merges = [(bytes.fromhex(left), bytes.fromhex(right)) for left, right in raw_merges]
        del raw_merges

        return cls(vocab, merges, special_tokens)

    def _find_best_merge_pair(self, pretoken: Word) -> Pair | None:
        best_rank = len(self.merge_ranks)
        best_pair = None
        for bytes_pair in pairwise(pretoken):
            rank = self.merge_ranks.get(bytes_pair)
            if rank is not None and rank < best_rank:
                best_rank = rank
                best_pair = bytes_pair
        return best_pair

    def _merge_selected_pair(self, old_token: Word, selected_pair: Pair) -> Word:
        new_token = []
        i = 0
        while i < len(old_token):
            if old_token[i : i + 2] == selected_pair:
                new_token.append(old_token[i] + old_token[i + 1])
                i += 2
            else:
                new_token.append(old_token[i])
                i += 1
        return tuple(new_token)

    def encode(self, text: str) -> list[int]:
        chunks, token_ids = [text], []
        if self.special_token_to_id:
            # Longer alternatives must win when special tokens share a prefix.
            pattern = "|".join(re.escape(token) for token in sorted(self.special_token_to_id, key=len, reverse=True))
            chunks = re.split(f"({pattern})", text)

        for chunk in chunks:
            if chunk in self.special_token_to_id:
                token_ids.append(self.special_token_to_id[chunk])
                continue
            for match in re.finditer(PAT, chunk):
                encoded = match.group(0).encode("utf-8")
                pretoken: Word = tuple(bytes([byte]) for byte in encoded)
                while len(pretoken) >= 2:
                    selected_pair = self._find_best_merge_pair(pretoken)
                    if selected_pair is None:
                        break
                    pretoken = self._merge_selected_pair(pretoken, selected_pair)

                for token in pretoken:
                    token_ids.append(self.bytes_to_id[token])

        return token_ids

    def _find_max_special_token_len(self) -> int:
        max_len = 0
        for special_token, _ in self.special_token_to_id.items():
            max_len = max(max_len, len(special_token))
        return max_len

    def encode_iterable(self, iterable: Iterable[str]) -> Iterator[int]:
        """Encode chunks while retaining text that may continue in the next chunk."""
        pending = ""

        if not self.special_token_to_id:
            for chunk in iterable:
                pending += chunk
                # The final pre-token may grow when the next chunk arrives.
                split_at = 0
                for match in re.finditer(PAT, pending):
                    split_at = match.start()
                if split_at > 0:
                    yield from self.encode(pending[:split_at])
                    pending = pending[split_at:]

            yield from self.encode(pending)
            return

        pattern = "|".join(re.escape(token) for token in sorted(self.special_token_to_id, key=len, reverse=True))
        special_pattern = re.compile(pattern)
        max_special_len = self._find_max_special_token_len()

        for chunk in iterable:
            pending += chunk
            while True:
                match = special_pattern.search(pending)
                # Wait until enough characters are available to rule out a longer special token.
                if match is None or len(pending) - match.start() < max_special_len:
                    break
                yield from self.encode(pending[: match.start()])
                yield self.special_token_to_id[match.group()]
                pending = pending[match.end() :]

            # A special token could begin within the final max_special_len - 1 characters.
            safe_limit = len(pending) - max_special_len + 1
            split_at = 0
            for match in re.finditer(PAT, pending):
                if match.start() >= safe_limit:
                    break
                split_at = match.start()
            if split_at > 0:
                yield from self.encode(pending[:split_at])
                pending = pending[split_at:]

        yield from self.encode(pending)

    def decode(self, ids: list[int]) -> str:
        tokens: list[bytes] = []
        for token_id in ids:
            tokens.append(self.vocab[token_id])

        # A UTF-8 character may span several tokens, so decode only after joining their bytes.
        return b"".join(tokens).decode("utf-8", errors="replace")
