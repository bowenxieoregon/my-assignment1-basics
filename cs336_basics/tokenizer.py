from pathlib import Path
from typing import BinaryIO, Iterable, Iterator
import regex as re
from collections import defaultdict
import json
import math

class Tokenizer():
    def __init__(self, vocab: dict[int, bytes], merges: list[tuple[bytes, bytes]], special_tokens: list[str] | None = None):
        self.vocab: dict[int, bytes] = vocab
        self.merges: list[tuple[bytes, bytes]] = merges
        self.special_tokens: list[str] | None = special_tokens
        self.inverse_vocab: dict[bytes, int] = {val: idx for idx, val in self.vocab.items()}
    
    @classmethod
    def from_files(cls, vocab_filepath: str, merges_filepath: str, special_tokens: list[str] | None = None):
        # vocabs are stored in .json file dict[str, int]
        # merges are stored in .txt file

        merges: list[tuple[bytes, bytes]] = []
        with Path(vocab_filepath).open("r", encoding="utf-8") as f_vocab:
            vocab_raw: dict[str, int] = json.load(f_vocab)
            vocab: dict[int, bytes] = {idx: bytes(s, encoding="utf-8") for s, idx in vocab_raw.items()}
        with Path(merges_filepath).open("r", encoding="utf-8") as f_merges:
            for line in f_merges:
                line = line.strip() #remove /n 
                
                if not line:
                    continue

                pairs = line.split()
                merges.append((bytes(pairs[0], encoding="utf-8"), bytes(pairs[1], encoding="utf-8")))

        return cls(vocab=vocab, merges=merges, special_tokens=special_tokens)

    def pre_tokenization(self, text: str) -> list[str]:
        words = []
        res = []
        PAT:str = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
        pat: re.Pattern = re.compile(PAT)

        #conisder special tokens
        if self.special_tokens is not None:
            tokens = sorted(self.special_tokens, key = len, reverse=True)
            sp_pattern = re.compile("(" + "|".join(map(re.escape, tokens)) + ")")
            words = [piece for piece in sp_pattern.split(text) if piece]

            for piece in words:
                if piece in self.special_tokens:
                    res.append(piece)
                else:
                    res.extend(pat.findall(piece))
            return res
        else:
            return pat.findall(text)

    def encode(self, text: str) -> list[int]:
        ids: list[int] = []
        #1. pre-tokenization
        pre_tokens: list[str] = self.pre_tokenization(text)

        #2. merge
        for word in pre_tokens:
            byte_word: bytes = bytes(word, encoding="utf-8")

            if byte_word in self.inverse_vocab.keys():
                ids.append(self.inverse_vocab[byte_word])
            else:
                merged_word: list[bytes] = [bytes([b]) for b in byte_word]
                flag = True
                while flag:
                    min_idx = math.inf
                    for pairs in zip(merged_word, merged_word[1:]):
                        if pairs[0] + pairs[1] in self.inverse_vocab.keys():
                            min_idx = min(min_idx, self.inverse_vocab[pairs[0] + pairs[1]])
                    
                    if not (min_idx < math.inf):
                        min_idx = False
                        break

                    new_merged_word: list[bytes] = []
                    vocab = self.vocab[min_idx]
                    i = 0
                    while i < len(merged_word):
                        if i < len(merged_word) - 1:
                            if merged_word[i] + merged_word[i + 1] == vocab:
                                new_merged_word.append(vocab)
                                i += 2
                            else:
                                new_merged_word.append(merged_word[i])
                                i += 1
                        else:
                            new_merged_word.append(merged_word[i])
                            i += 1
                    merged_word = new_merged_word.copy()

                for p in merged_word:
                    ids.append(self.inverse_vocab[p])
        return ids
            
    def encode_iterable(self, iterable: Iterable[str]) -> Iterator[int]:
        for text_chunk in iterable:
            yield from self.encode(text_chunk)       

    def decode(self, ids: list[int]) -> str:
        res_bytes: bytes = bytes()
        for id in ids:
            res_bytes += self.vocab[id]
        return res_bytes.decode(encoding="utf-8", errors='replace')
    


