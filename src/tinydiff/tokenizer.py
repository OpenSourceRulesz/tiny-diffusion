"""Word-level tokenizer with protected special tokens for tool markup."""

import json
import re
from collections import Counter

SPECIALS = [
    "[PAD]", "[MASK]", "[UNK]",
    "[USER]", "[ASSIST]",
    "[THINK]", "[/THINK]",
    "[CALL]", "[/CALL]",
    "[RESULT]", "[/RESULT]",
    "[END]",
]

_SPLIT_RE = re.compile("(" + "|".join(re.escape(s) for s in SPECIALS) + ")")


def _emit(buf, was_digit):
    if was_digit:
        yield from buf  # one token per digit: any number is representable
    else:
        yield buf


def _split_mixed(word):
    """Split '47?' -> '4','7','?' and '2+3' -> '2','+','3'."""
    buf, was_digit = "", None
    for ch in word:
        is_digit = ch.isdigit()
        if was_digit is None:
            was_digit, buf = is_digit, ch
        elif is_digit == was_digit:
            buf += ch
        else:
            yield from _emit(buf, was_digit)
            was_digit, buf = is_digit, ch
    if buf:
        yield from _emit(buf, was_digit)


def _pieces(text):
    for part in _SPLIT_RE.split(text):
        part = part.strip()
        if not part:
            continue
        if part in SPECIALS:
            yield part
        else:
            for word in part.split():
                yield from _split_mixed(word)


class WordTokenizer:
    def __init__(self, stoi, itos):
        self.stoi = stoi
        self.itos = list(itos)
        self.pad_id = stoi["[PAD]"]
        self.mask_id = stoi["[MASK]"]
        self.unk_id = stoi["[UNK]"]

    def __len__(self):
        return len(self.itos)

    @classmethod
    def build(cls, texts, vocab_size=4000, min_freq=2):
        counter = Counter()
        for t in texts:
            for piece in _pieces(t):
                if piece in SPECIALS:
                    counter[piece] += 10 ** 9  # keep specials pinned at the top
                else:
                    counter[piece] += 1
        vocab = list(SPECIALS)
        for word, count in counter.most_common():
            if word in SPECIALS or count < min_freq:
                continue
            vocab.append(word)
            if len(vocab) >= vocab_size:
                break
        stoi = {w: i for i, w in enumerate(vocab)}
        return cls(stoi, vocab)

    def encode(self, text):
        return [self.stoi.get(p, self.unk_id) for p in _pieces(text)]

    def decode(self, ids):
        out = []
        for i in ids:
            if i == self.pad_id:
                continue
            t = self.itos[i]
            if out and t.isdigit() and out[-1].isdigit():
                out[-1] += t  # rejoin split digits: "2 3 8" -> "238"
            else:
                out.append(t)
        return " ".join(out)

    def save(self, path):
        with open(path, "w") as f:
            json.dump({"stoi": self.stoi, "itos": self.itos}, f)

    @classmethod
    def load(cls, path):
        with open(path) as f:
            d = json.load(f)
        return cls(d["stoi"], d["itos"])
