# copy
import os
import json
import tqdm
import torch
import numpy as np
import regex as re
from typing import BinaryIO, Iterable, Iterator, Tuple
from multiprocessing import Pool, get_context
from collections import defaultdict

GPT2_PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
_COMPILED_PAT = re.compile(GPT2_PAT)

def run_train_bpe(
    input_path: str,
    vocab_size: int,
    special_tokens: list[str],
    num_processes: int = 1
) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    # 1. Vocabulary Initialization
    vocab = {i: bytes([i]) for i in range(256)}
    for tok in special_tokens:
        vocab[len(vocab)] = tok.encode("utf-8")
    # 2. Pre-tokenization
    with open(input_path, "rb") as f:
        bounds = find_chunk_boundaries(f, num_processes, "<|endoftext|>".encode("utf-8"))
    task_args = [
        (input_path, start, end, special_tokens)
        for start, end in zip(bounds[:-1], bounds[1:])
    ]
    with get_context("forkserver").Pool(processes=num_processes) as pool:
        chunk_results = pool.map(process_chunk, task_args) # list[list[list[int]]]
    # 3. Compute BPE merges
    ids: list[list[int]] = [
        token_ids for chunk_ids in chunk_results for token_ids in chunk_ids
    ] # chunk_ids is list[token_ids], token_ids is list[int]
    merges: list[tuple[int, int]] = []
    # Get all pairs from the pre-tokenized bytes
    pair_to_indices, counts = _get_pair_counts(ids)
    num_merges = vocab_size - len(vocab)
    for i in tqdm.tqdm(range(num_merges)):
        if not counts:
            break
        # Find the most frequent pair
        def rank(pair: tuple[int, int]) -> tuple[int, tuple[bytes, bytes]]:
            return counts[pair], (vocab[pair[0]], vocab[pair[1]])
        max_pair = max(counts, key=rank)
        new_token = vocab[max_pair[0]] + vocab[max_pair[1]]
        new_id = len(vocab)
        vocab[new_id] = new_token
        merges.append(max_pair)

        # Merge the most frequent pair in all affected tokens
        # Use affected_indices to only update tokens that contain the max_pair
        affected_indices = pair_to_indices[max_pair].copy()
        for j in affected_indices:
            token_ids = ids[j]
            if len(token_ids) < 2:
                continue
            # 1. Decrement counts for pairs in the old token
            # just ignore all the pair count in the old token right now
            for pair in zip(token_ids, token_ids[1:]):
                counts[pair] -= 1
                pair_to_indices[pair].discard(j)
                if counts[pair] == 0:
                    del counts[pair]
                    del pair_to_indices[pair]

            # 2. Merge the pair to create the new token
            new_token_ids = _merge_pair(token_ids, max_pair, new_id)

            # 3. Increment counts for pairs in the new token
            # add all the pair count in the new token
            for pair in zip(new_token_ids, new_token_ids[1:]):
                counts[pair] += 1
                pair_to_indices[pair].add(j)

            ids[j] = new_token_ids

    merges = [(vocab[a], vocab[b]) for a, b in merges]
    torch.save(merges, "merges.pt")
    torch.save(vocab, "vocab.pt")
    return vocab, merges

def _get_pair_counts(
        ids: list[list[int]]
    ) -> tuple[
        defaultdict[tuple[int, int], set],
        defaultdict[tuple[int, int], int]
    ]:
    """Counts initial byte pair frequencies."""
    pair_to_indices = defaultdict(set)
    counts = defaultdict(int)
    for i, token_ids in enumerate(ids):
        for pair in zip(token_ids, token_ids[1:]):
            pair_to_indices[pair].add(i)
            counts[pair] += 1
    return pair_to_indices, counts

def _merge_pair(
    token_ids: list[int], pair: tuple[int, int], new_id: int
) -> list[int]:
    """Merges a pair of bytes into a new token within a list of bytes."""
    new_token_ids = []
    i = 0
    while i < len(token_ids):
        if i < len(token_ids) - 1 and (token_ids[i], token_ids[i+1]) == pair:
            new_token_ids.append(new_id)
            i += 2
        else:
            new_token_ids.append(token_ids[i])
            i += 1
    return new_token_ids

def find_chunk_boundaries(
    file: BinaryIO,
    desired_num_chunks: int,
    split_special_token: bytes
) -> list[int]:
    assert isinstance(split_special_token, bytes), \
        "Must represent special token as a bytestring"
    file.seek(0, os.SEEK_END)
    file_size = file.tell()
    file.seek(0)
    chunk_size = max(1, file_size // desired_num_chunks)
    bounds = [i * chunk_size for i in range(desired_num_chunks + 1)]
    bounds[-1] = file_size
    mini = 4096  # 4k scan step (bigger save syscall)
    for bi in range(1, len(bounds) - 1):
        pos = bounds[bi]
        file.seek(pos)
        while True:
            buf = file.read(mini)
            if not buf:
                bounds[bi] = file_size
                break
            found = buf.find(split_special_token)
            if found != -1:
                bounds[bi] = pos + found
                break
            pos += len(buf)
    return sorted(set(bounds))


def process_chunk(args: tuple[str, int, int, list[str]]) -> list[list[int]]:
    input_path, start, end, special_tokens = args
    with open(input_path, "rb") as file:
        file.seek(start)
        chunk = file.read(end - start).decode("utf-8", errors="ignore")
    # 1. Remove special tokens by splitting the chunk at those tokens
    pattern = "|".join(re.escape(tok) for tok in special_tokens)
    documents = re.split(pattern, chunk)
    # 2. Pre-tokenize and count byte pair frequencies
    chunk_ids: list[list[int]] = []
    for doc in documents:
        tokens = [match.group(0).encode("utf-8") for match in _COMPILED_PAT.finditer(doc)]
        chunk_ids.extend([list(token) for token in tokens]) # list(bytes) -> list[int]
    return chunk_ids

class Tokenizer:
    def __init__(
        self,
        vocab: dict[int, bytes],
        merges: list[tuple[bytes, bytes]],
        special_tokens: list[str] | None = None,
    ):
        self.vocab = vocab
        self.byte_to_token_id = {v: k for k, v in vocab.items()}
        self.merges = merges
        self.bpe_ranks = dict(zip(merges, range(len(merges))))
        self.special_tokens = special_tokens or []
        self.special_token_bytes = [token.encode("utf-8") for token in self.special_tokens]
        for token_bytes in self.special_token_bytes:
            if token_bytes not in self.byte_to_token_id:
                new_id = len(self.vocab)
                self.vocab[new_id] = token_bytes
                self.byte_to_token_id[token_bytes] = new_id

    def encode(self, text: str) -> list[int]:
        tokens = []
        sorted_special_tokens = sorted(self.special_tokens, key=len, reverse=True)
        pattern = "|".join(map(re.escape, sorted_special_tokens))
        if pattern:
            parts = re.split(f"({pattern})", text)
        else:
            parts = [text]
        for part in parts:
            if part in self.special_tokens:
                tokens.append(self.byte_to_token_id[part.encode("utf-8")])
            else:
                tokens.extend(self._tokenize_normal(part))
        return tokens

    def encode_iterable(self, iterable: Iterable[str]) -> iter:
        for chunk in iterable:
            yield from self.encode(chunk)

    def decode(self, ids: list[int]) -> str:
        full_bytes = b"".join(self.vocab[token_id] for token_id in ids)
        return full_bytes.decode("utf-8", errors="replace")

    def _tokenize_normal(self, text: str) -> list[int]:
        pre_tokens = []
        PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
        for m in re.finditer(PAT, text):
            word = m.group(0)
            pre_tokens.append(word)
        token_ids = []
        def to_bytes_tuple(word: str) -> Tuple[bytes]:
            l = list(word.encode("utf-8"))
            l = [bytes([x]) for x in l]
            return tuple(l)
        for token in pre_tokens:
            byte_tuple = to_bytes_tuple(token)
            merged = self._apply_merges(byte_tuple)
            token_ids.extend(self.byte_to_token_id[b] for b in merged)
        return token_ids

    def _apply_merges(self, byte_tuple: tuple[bytes, ...]) -> list[bytes]:

        word: list[bytes] = list(byte_tuple)

        def get_pairs(word: list[bytes]):
            pairs = set()
            prev_char = word[0]
            for char in word[1:]:
                pairs.add((prev_char, char))
                prev_char = char
            return pairs

        pairs = get_pairs(word)

        if not pairs:
            return word

        while True:
            bigram = min(pairs, key=lambda pair: self.bpe_ranks.get(pair, float('inf')))
            if bigram not in self.bpe_ranks:
                break

            first, second = bigram
            new_word = []
            i = 0
            while i < len(word):
                try:
                    j = word.index(first, i)
                except ValueError:
                    new_word.extend(word[i:])
                    break
                else:
                    new_word.extend(word[i:j])
                    i = j

                if word[i] == first and i < len(word) - 1 and word[i + 1] == second:
                    new_word.append(first + second)
                    i += 2
                else:
                    new_word.append(word[i])
                    i += 1
            new_word = tuple(new_word)
            word = new_word
            if len(word) == 1:
                break
            else:
                pairs = get_pairs(word)

        return word

if __name__ == "__main__":
    input_path = "data/TinyStoriesV2-GPT4-train.txt"
    output_path = "data/TinyStoriesV2-GPT4-train.bin"

    vocab, merges = run_train_bpe(
        input_path = input_path,
        vocab_size = 10000,
        special_tokens = ["<|endoftext|>"],
    )
    tokenizer = Tokenizer(vocab, merges, special_tokens=["<|endoftext|>"])


    with open(output_path, "wb") as f_out:
        with open(input_path, "r", encoding="utf-8") as f_in:
            for line in f_in:
                tokens = tokenizer.encode(line)
                arr = np.array(tokens, dtype=np.uint32)
                f_out.write(arr.tobytes())