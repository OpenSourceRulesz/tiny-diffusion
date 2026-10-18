"""Synthetic, verifiable tool-use trajectories + simple language filler.

Every trajectory is deterministic: the tool call in the text, executed for real,
produces exactly the [RESULT] in the text. That makes tool-use correctness
mechanically checkable — the whole point of training on synthetic tools first.
"""

import random

from .tools import KB, run_call

CALC_THINKS = [
    "I will use the calculator for this.",
    "Let me compute that with the calc tool.",
    "This needs arithmetic, so I will call calc.",
    "I should not guess; I will calculate it.",
]
LOOKUP_THINKS = [
    "I will look that up in the knowledge base.",
    "Let me check the lookup tool for this fact.",
    "I should verify this with the lookup tool.",
]
ANSWER_LEADERS = [
    "The answer is",
    "So the answer is",
    "That gives",
]

_NOUNS = ["fox", "dog", "cat", "bird", "river", "tree", "house", "car", "book", "child",
          "garden", "mountain", "ocean", "star", "moon", "teacher", "farmer", "baker"]
_VERBS = ["runs", "jumps", "sleeps", "sings", "walks", "laughs", "works", "plays",
          "reads", "writes", "swims", "flies"]
_ADJS = ["quick", "lazy", "happy", "small", "big", "bright", "quiet", "brave",
         "clever", "gentle"]
_ADVS = ["quickly", "slowly", "happily", "quietly", "early", "late"]


def gen_calc(rng, lo=2, hi=500):
    a = rng.randint(lo, hi)
    b = rng.randint(lo, hi)
    op = rng.choice(["+", "-", "*"])
    expr = f"{a} {op} {b}"
    value = run_call(f"calc {expr}")
    q = rng.choice([
        f"What is {a} {op} {b}?",
        f"Compute {a} {op} {b}.",
        f"How much is {a} {op} {b}?",
    ])
    think = rng.choice(CALC_THINKS)
    leader = rng.choice(ANSWER_LEADERS)
    return (f"[USER] {q} [ASSIST] [THINK] {think} [/THINK] "
            f"[CALL] calc {expr} [/CALL] [RESULT] {value} [/RESULT] {leader} {value}. [END]")


def gen_lookup(rng):
    key = rng.choice(sorted(KB.keys()))
    value = run_call(f"lookup {key}")
    q = rng.choice([
        f"What is the {key}?",
        f"Tell me the {key}.",
        f"Do you know the {key}?",
    ])
    think = rng.choice(LOOKUP_THINKS)
    leader = rng.choice(ANSWER_LEADERS)
    return (f"[USER] {q} [ASSIST] [THINK] {think} [/THINK] "
            f"[CALL] lookup {key} [/CALL] [RESULT] {value} [/RESULT] {leader} {value}. [END]")


def gen_sentence(rng):
    return (f"The {rng.choice(_ADJS)} {rng.choice(_NOUNS)} {rng.choice(_VERBS)} "
            f"{rng.choice(_ADVS)}. [END]")


def make_dataset(n_tool=4000, n_lang=2000, seed=0, hi=500):
    rng = random.Random(seed)
    texts = []
    for _ in range(n_tool):
        texts.append(gen_calc(rng, hi=hi) if rng.random() < 0.6 else gen_lookup(rng))
    for _ in range(n_lang):
        texts.append(gen_sentence(rng))
    rng.shuffle(texts)
    return texts
