"""Smoke tests: the whole pipeline runs end to end."""

import torch

from tinydiff.data import make_dataset
from tinydiff.diffusion import diffusion_loss, generate
from tinydiff.model import TinyDiffusionLM
from tinydiff.tokenizer import WordTokenizer
from tinydiff.tools import run_call


def test_tokenizer_roundtrip():
    texts = make_dataset(n_tool=50, n_lang=20, seed=1)
    tok = WordTokenizer.build(texts, min_freq=1)
    t = texts[0]  # roundtrip real dataset text, specials included
    assert tok.decode(tok.encode(t)) == t
    assert tok.encode("[MASK]")[0] == tok.mask_id
    # digits split and rejoin: unseen numbers stay representable
    assert tok.decode(tok.encode("238 * 47")) == "238 * 47"


def test_tools():
    assert run_call("calc 238*47+12") == "11198"
    assert run_call("lookup capital of france") == "Paris"
    assert run_call("nope x").startswith("error")


def test_loss_and_generate():
    texts = make_dataset(n_tool=200, n_lang=100, seed=2)
    tok = WordTokenizer.build(texts)
    enc = [tok.encode(t)[:64] for t in texts]
    T = max(len(e) for e in enc)
    ids = torch.full((len(enc), T), tok.pad_id, dtype=torch.long)
    for i, e in enumerate(enc):
        ids[i, :len(e)] = torch.tensor(e)
    model = TinyDiffusionLM(len(tok), d_model=64, n_layers=2, n_heads=2, max_len=128)
    loss = diffusion_loss(model, ids[:8], tok)
    assert torch.isfinite(loss), loss
    prompt = tok.encode("[USER] What is 2 + 3? [ASSIST]")
    out = generate(model, tok, prompt, gen_len=16, steps=6, seed=0)
    assert len(out) == len(prompt) + 16
    # prompt region untouched
    assert out[:len(prompt)] == prompt
