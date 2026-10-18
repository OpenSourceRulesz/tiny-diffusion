"""Smoke-train the tiny diffusion LM on synthetic tool trajectories."""

import argparse
import json
import os
import random

import torch
from torch.utils.data import DataLoader, TensorDataset

from .data import make_dataset
from .diffusion import diffusion_loss
from .model import TinyDiffusionLM
from .tokenizer import WordTokenizer


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=600)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--lr", type=float, default=5e-4)
    ap.add_argument("--d-model", type=int, default=128)
    ap.add_argument("--layers", type=int, default=2)
    ap.add_argument("--heads", type=int, default=4)
    ap.add_argument("--max-len", type=int, default=128)
    ap.add_argument("--n-tool", type=int, default=4000)
    ap.add_argument("--n-lang", type=int, default=2000)
    ap.add_argument("--hi", type=int, default=500,
                    help="max operand for synthetic calc questions")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--ckpt", default="checkpoints/smoke")
    ap.add_argument("--save-every", type=int, default=500,
                    help="checkpoint every N steps so restarts don't lose progress")
    ap.add_argument("--resume", default=None,
                    help="checkpoint dir to resume from (continues step count)")
    args = ap.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)

    print("generating synthetic data...")
    texts = make_dataset(n_tool=args.n_tool, n_lang=args.n_lang, seed=args.seed,
                         hi=args.hi)
    tok = WordTokenizer.build(texts)
    print(f"vocab size: {len(tok)}")

    encoded = [tok.encode(t)[:args.max_len] for t in texts]
    max_t = max(len(e) for e in encoded)
    ids = torch.full((len(encoded), max_t), tok.pad_id, dtype=torch.long)
    for i, e in enumerate(encoded):
        ids[i, :len(e)] = torch.tensor(e)

    loader = DataLoader(TensorDataset(ids), batch_size=args.batch, shuffle=True)
    model = TinyDiffusionLM(len(tok), d_model=args.d_model, n_layers=args.layers,
                            n_heads=args.heads, max_len=args.max_len)
    start_step = 0
    if args.resume:
        ckpt = torch.load(f"{args.resume}/model.pt", map_location="cpu",
                          weights_only=False)
        model.load_state_dict(ckpt["state_dict"])
        with open(f"{args.resume}/meta.json") as f:
            start_step = json.load(f).get("step", 0)
        print(f"resumed from {args.resume} at step {start_step}")
    print(f"params: {model.n_params():,}")
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)
    gen = torch.Generator().manual_seed(args.seed)

    def save():
        os.makedirs(args.ckpt, exist_ok=True)
        torch.save({"config": vars(args), "vocab_size": len(tok),
                    "state_dict": model.state_dict()}, f"{args.ckpt}/model.pt")
        tok.save(f"{args.ckpt}/tokenizer.json")
        with open(f"{args.ckpt}/meta.json", "w") as f:
            json.dump({"final_loss": loss.item(), "step": step}, f)

    model.train()
    step = start_step
    batches = iter(loader)
    # fast-forward the data loader so resume sees fresh batches in the same order
    _seen = 0
    while _seen < start_step:
        try:
            (batch,) = next(batches)
        except StopIteration:
            batches = iter(loader)
            (batch,) = next(batches)
        _seen += 1
    while step < args.steps:
        try:
            (batch,) = next(batches)
        except StopIteration:
            batches = iter(loader)
            (batch,) = next(batches)
        loss = diffusion_loss(model, batch, tok, generator=gen)
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        step += 1
        if step % 50 == 0 or step == 1:
            print(f"step {step}/{args.steps}  loss {loss.item():.4f}", flush=True)
        if step % args.save_every == 0:
            save()
            print(f"checkpointed at step {step}", flush=True)

    save()
    print(f"saved to {args.ckpt}/")


if __name__ == "__main__":
    main()
