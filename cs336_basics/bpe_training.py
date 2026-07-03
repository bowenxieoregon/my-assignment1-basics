import os
from typing import BinaryIO
from pathlib import Path
from multiprocessing import Pool
import regex as re
from collections import defaultdict, Counter
from functools import partial

def find_chunk_boundaries(
    file: BinaryIO,
    desired_num_chunks: int,
    split_special_token: bytes,
) -> list[int]:
    """
    Chunk the file into parts that can be counted independently.
    May return fewer chunks if the boundaries end up overlapping.
    """
    assert isinstance(split_special_token, bytes), "Must represent special token as a bytestring"

    # Get total file size in bytes
    file.seek(0, os.SEEK_END)
    file_size = file.tell()
    file.seek(0)

    """
    Bowen Q : Why must use bytes to represent the file not simply using str? Maybe due to mechanism
              of file.read(), file.seek(), etc. 
    """
    chunk_size = file_size // desired_num_chunks

    # Initial guesses for chunk boundary locations, uniformly spaced
    # Chunks start on previous index, don't include last index
    chunk_boundaries = [i * chunk_size for i in range(desired_num_chunks + 1)]
    chunk_boundaries[-1] = file_size

    mini_chunk_size = 4096  # Read ahead by 4k bytes at a time

    for bi in range(1, len(chunk_boundaries) - 1):
        initial_position = chunk_boundaries[bi]
        file.seek(initial_position)  # Start at boundary guess
        while True:
            mini_chunk = file.read(mini_chunk_size)  # Read a mini chunk

            # If EOF, this boundary should be at the end of the file
            if mini_chunk == b"":
                chunk_boundaries[bi] = file_size
                break

            # Find the special token in the mini chunk
            found_at = mini_chunk.find(split_special_token)
            if found_at != -1:
                chunk_boundaries[bi] = initial_position + found_at
                break
            initial_position += mini_chunk_size

    # Make sure all boundaries are unique, but might be fewer than desired_num_chunks
    return sorted(set(chunk_boundaries))

'''
We need to remove special tokens before pre-tokenization
'''
def remove_special_tokens(chunk: str, special_tokens: list[str]) -> list[str]:
    # pattern = "|".join(special_tokens)
    # return re.split(pattern=pattern,string=chunk)
    if not special_tokens:
        return [chunk]

    pattern = "|".join(re.escape(token) for token in sorted(special_tokens, key=len, reverse=True))
    return re.split(pattern=pattern,string=chunk)

def pre_tokenization(
        boundary:tuple[int, int],
        input_path:str,
        special_tokens:list[str],
        pattern: re.Pattern,
)->dict[bytes,int]:
    with open(input_path, "rb") as f:
        f.seek(boundary[0])
        chunk:str = f.read(boundary[1] - boundary[0]).decode(encoding="utf-8", errors="ignore")
    sub_chunks:list[str] = remove_special_tokens(chunk, special_tokens)
    c = defaultdict(int)
    for i, sub_chunk in enumerate(sub_chunks):
        #Pretokenization
        pretoked_sub_chunk = pattern.finditer(sub_chunk)
        for match in pretoked_sub_chunk:
            token_str: str = match.group(0)
            token_bytes: bytes = token_str.encode(encoding="utf-8")
            pieces = tuple(bytes([b]) for b in token_bytes)
            c[pieces] = c.get(pieces, 0) + 1
    return c

def training(
        vocab_size:int,
        freq_dict: dict[bytes, int],
        special_tokens: list[str]
)->tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    # vocabs = {(i+1):bytes(chr(i), encoding="utf-8") for i in range(256)}
    vocabs = {(i + 1): bytes([i]) for i in range(256)}
    # vocabs[0] = special_tokens[0].encode(encoding="utf-8")
    vocabs[0] = special_tokens[0].encode(encoding="utf-8")
    merges = []

    pairs_count = defaultdict(dict)
    word_state = {}

    def current_pair(word: tuple[bytes, ...], idx: int):
        state = word_state[word]
        right_idx = state["next"][idx]
        if right_idx == -1 or not state["alive"][idx] or not state["alive"][right_idx]:
            return ""
        return (state["tokens"][idx], state["tokens"][right_idx])

    def add_pair(word: tuple[bytes, ...], idx: int, left_pair="", right_pair=""):
        pair = current_pair(word, idx)
        if pair:
            pairs_count[pair][(word, idx)] = [True, freq_dict[word], left_pair, right_pair]
        return pair

    def deactivate_pair(pair, word: tuple[bytes, ...], idx: int):
        if pair and (word, idx) in pairs_count.get(pair, {}):
            pairs_count[pair][(word, idx)][0] = False

    def update_left(pair, word: tuple[bytes, ...], idx: int, left_pair):
        if pair and (word, idx) in pairs_count.get(pair, {}):
            pairs_count[pair][(word, idx)][-2] = left_pair

    def update_right(pair, word: tuple[bytes, ...], idx: int, right_pair):
        if pair and (word, idx) in pairs_count.get(pair, {}):
            pairs_count[pair][(word, idx)][-1] = right_pair

    def merge_at(word: tuple[bytes, ...], idx: int, max_pair: tuple[bytes, bytes], new_vocab: bytes):
        if current_pair(word, idx) != max_pair:
            return

        state = word_state[word]
        right_idx = state["next"][idx]
        left_idx = state["prev"][idx]
        after_idx = state["next"][right_idx]

        old_left_pair = current_pair(word, left_idx) if left_idx != -1 else ""
        old_right_pair = current_pair(word, right_idx) if after_idx != -1 else ""

        deactivate_pair(old_left_pair, word, left_idx)
        deactivate_pair(max_pair, word, idx)
        deactivate_pair(old_right_pair, word, right_idx)

        before_left_idx = state["prev"][left_idx] if left_idx != -1 else -1
        after_right_idx = state["next"][after_idx] if after_idx != -1 else -1
        before_left_pair = current_pair(word, before_left_idx) if before_left_idx != -1 else ""
        after_right_pair = current_pair(word, after_idx) if after_right_idx != -1 else ""

        state["tokens"][idx] = new_vocab
        state["alive"][right_idx] = False
        state["next"][idx] = after_idx
        if after_idx != -1:
            state["prev"][after_idx] = idx

        left_new_pair = add_pair(word, left_idx, before_left_pair, "") if left_idx != -1 else ""
        right_new_pair = add_pair(word, idx, left_new_pair, after_right_pair) if after_idx != -1 else ""

        if left_new_pair:
            pairs_count[left_new_pair][(word, left_idx)][-1] = right_new_pair
            update_right(before_left_pair, word, before_left_idx, left_new_pair)
        if right_new_pair:
            update_left(after_right_pair, word, after_idx, right_new_pair)

    for word_bytes,freq in freq_dict.items():
        word_state[word_bytes] = {
            "tokens": list(word_bytes),
            "prev": [i - 1 for i in range(len(word_bytes))],
            "next": [i + 1 for i in range(len(word_bytes))],
            "alive": [True for _ in word_bytes],
        }
        if word_bytes:
            word_state[word_bytes]["prev"][0] = -1
            word_state[word_bytes]["next"][-1] = -1

        prev_pair = ""
        prev_idx = -1
        for i, pair in enumerate(zip(word_bytes, word_bytes[1:])):
            if prev_pair:
                pairs_count[prev_pair][(word_bytes, prev_idx)][-1] = pair
            pairs_count[pair][(word_bytes, i)] = [True, freq_dict[word_bytes], prev_pair, ""]
            prev_pair = pair
            prev_idx = i
            
    total_epoch = vocab_size - len(vocabs.items())
    for epoch in range(total_epoch):
        for pair in list(pairs_count.keys()):
            if sum(sub_info[1] for sub_info in pairs_count[pair].values() if sub_info[0]) == 0:
                pairs_count.pop(pair)
        if not pairs_count:
            break
        max_pair = max((sum([(sub_info[1] * sub_info[0]) for _, sub_info in info.items()]), pair) for pair, info in pairs_count.items())[1]
        new_vocab = max_pair[0] + max_pair[1]
        vocabs[257 + epoch] = new_vocab
        merges.append(max_pair)
        # print(f"merging: {max_pair}")
        # print(f"new vocab: {new_vocab}")
        for (word, idx), info in list(pairs_count[max_pair].items()):
            if info[0]:
                merge_at(word, idx, max_pair, new_vocab)
        pairs_count.pop(max_pair, None)

    return vocabs, merges 

def train_bpe(
        input_path: str|os.PathLike,
        vocab_size: int,
        special_tokens: list[str],
        ):
    # print(f"We have {os.cpu_count()} available processes")
    PAT:str = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
    pattern: re.Pattern = re.compile(PAT)
    ## Step 1: Parallelizing the pretokenization and remove special tokens
    num_processes = os.cpu_count()
    with open(input_path, "rb") as f:
        boundaries = find_chunk_boundaries(f, num_processes, special_tokens[0].encode(encoding="utf-8"))

    boundaries_pair = list(zip(boundaries[:-1], boundaries[1:]))
    partial_pre_tokenization = partial(
        pre_tokenization, 
        input_path=input_path, 
        special_tokens=special_tokens,
        pattern=pattern)
    with Pool(processes=num_processes) as pool:
        results = pool.map(partial_pre_tokenization, boundaries_pair)
    c = Counter()
    for result in results:
        c.update(result)

    ## Step 2: Merging
    vocabs, merges = training(vocab_size=vocab_size, freq_dict=c, special_tokens=special_tokens)
    return vocabs, merges


if __name__ == "__main__":
    import time
    vocab_size = 32000
    folder = Path.cwd()
    # data_path = folder/"data"/"TinyStoriesV2-GPT4-train.txt"
    data_path = folder/"data"/"owt_train.txt"
    start = time.perf_counter()
    vocabs, merges = train_bpe(data_path, vocab_size,["<|endoftext|>"])
    end = time.perf_counter()
    print(f"train_bpe took {end - start:.4f} seconds")
