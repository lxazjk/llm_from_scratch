import os
from typing import BinaryIO
import regex as re
from collections import Counter
from multiprocessing import Pool
from cs336_basics.pretokenization_example import find_chunk_boundaries
PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""

def _read_context(
    input_path: str | os.PathLike,
    PAT: str,
    start: int,
    end: int
):
    Final_PAT = re.compile(PAT)
    local_count = Counter()
    with open(input_path, "rb") as f:
        f.seek(start)
        chunk = f.read(end - start).decode("utf-8", errors="ignore")
        # print(chunk)
        iters = re.finditer(PAT, chunk)
        for iter in iters:
            # print(list(iter.group().encode("utf-8")))
            local_count[tuple(iter.group().encode("utf-8"))] += 1
    return local_count

def run_train_bpe(
    input_path: str | os.PathLike,
    vocab_size: int,
    special_tokens: list[str],
    **kwargs,
) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    if special_tokens:
        special_pattern = "|".join(re.escape(tok) for tok in special_tokens)
        Final_PAT = f"{special_pattern}|{PAT}"
    else:
        Final_PAT = PAT
    # print(f"input_path{input_path}, vocab_size{vocab_size}, special_token{special_tokens}")
    num_processes = 64
    with open(input_path, "rb") as f:
        boundaries = find_chunk_boundaries(f, num_processes, b"<|endoftext|>")
        tasks = []
        for start, end in zip(boundaries[:-1], boundaries[1:]):
            tasks.append((input_path, Final_PAT, start, end))
            # print(f"{start} -> {end}")
    word_count = Counter()
    with Pool(num_processes) as pool:
        res = pool.starmap(_read_context, tasks)
        for _, local_count in enumerate(res):
            word_count.update(local_count)
    print(len(word_count))
    num_iteration = vocab_size - 256 - len(special_tokens)
    vocab = {}
    merge = []
    for i in range(256):
        vocab[i] = bytes([i])
    for i in range(256, 256 + len(special_tokens)):
        vocab[i] = special_tokens[i - 256].encode("utf-8")
    base = 256 + len(special_tokens)
    for offset in range(num_iteration):
        pair_count = Counter()
        for word, freq in word_count.items():
            for i in range(len(word) - 1):
                pair_count[(word[i], word[i + 1])] += freq
        if not pair_count:
            print(f"No more pairs to merge at iteration {offset}. Stopping.")
            break
        best_pair = max(pair_count, key=lambda p: (pair_count[p], -p[0], -p[1]))
        part1_bytes = vocab[best_pair[0]]
        part2_bytes = vocab[best_pair[1]]
        vocab[offset + base] = part1_bytes + part2_bytes
        merge.append((part1_bytes, part2_bytes))
        new_word_count = Counter()
        for word in word_count:
            i = 0
            new_word = []
            while i < len(word):
                if i + 1 < len(word) and (word[i], word[i + 1]) == best_pair:
                    new_word.append(offset + base)
                    i += 2
                else :
                    new_word.append(word[i])
                    i += 1
            new_word_count[tuple(new_word)] += word_count[word]
        word_count = new_word_count
    return vocab, merge

if __name__ == "__main__":
    run_train_bpe("./data/TinyStoriesV2-GPT4-train.txt", 32000, ["<|endoftext|>"])