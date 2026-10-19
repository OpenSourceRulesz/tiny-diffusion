"""Smoke-train the tiny diffusion LM on synthetic tool trajectories."""

import argparse
import itertools
import json
import os
import random

import torch
from torch.utils.data import DataLoader, TensorDataset

from .data import make_dataset
from .diffusion import diffusion_loss
from .model import TinyDiffusionLM
from .tokenizer import WordTokenizer
from .tools import kb_words


def iter_stories(path, max_stories=None):
    """Yield stories one at a time; never holds the whole file in RAM."""
    buf = []
    n = 0
    with open(path, encoding="utf-8") as f:
        for line in f:
            if "<|endoftext|>" in line:
                s = "".join(buf).strip()
                buf = []
                if s:
                    yield s
                    n += 1
                    if max_stories and n >= max_stories:
                        return
            else:
                buf.append(line)


def count_stories(path):
    n = 0
    with open(path, encoding="utf-8") as f:
        for line in f:
            if "<|endoftext|>" in line:
                n += 1
    return n


def get_data_and_tokenizer(args):
    pinned = kb_words() if args.pin_kb else []
    if pinned:
        print(f"pinning {len(pinned)} KB words into vocab")
    if args.data == "synthetic":
        texts = make_dataset(n_tool=args.n_tool, n_lang=args.n_lang,
                             seed=args.seed, hi=args.hi)
        if args.tokenizer:
            tok = WordTokenizer.load(args.tokenizer)
        else:
            tok = WordTokenizer.build(texts, vocab_size=args.vocab_size,
                                      pinned=pinned)
        encoded = [tok.encode(t)[:args.max_len] for t in texts]
        max_t = max(len(e) for e in encoded)
        ids = torch.full((len(encoded), max_t), tok.pad_id, dtype=torch.long)
        for i, e in enumerate(encoded):
            ids[i, :len(e)] = torch.tensor(e)
        return ids, tok
    elif args.data in ("tinystories", "mixed"):
        # mix in tool trajectories so tool markup/words are in vocab
        tool_texts = make_dataset(n_tool=2000, n_lang=0, seed=args.seed, hi=args.hi)
        if args.tokenizer:
            tok = WordTokenizer.load(args.tokenizer)
        else:
            sample = list(itertools.islice(iter_stories(args.data_path),
                                           args.vocab_stories))
            print(f"building vocab from {len(sample)} stories...")
            tok = WordTokenizer.build(sample + tool_texts,
                                      vocab_size=args.vocab_size, pinned=pinned)
            del sample
        if args.data == "tinystories":
            n = args.max_stories or count_stories(args.data_path)
            story_iter = iter_stories(args.data_path, args.max_stories)
        else:
            n = args.mix_stories
            story_iter = iter_stories(args.data_path, args.mix_stories)
        ids = torch.full((n, args.max_len), tok.pad_id, dtype=torch.long)
        i = 0
        for s in story_iter:
            if i >= n:
                break
            e = tok.encode(s)[:args.max_len]
            ids[i, :len(e)] = torch.tensor(e)
            i += 1
        print(f"loaded {i} stories -> tensor {tuple(ids.shape)}")
        if args.data == "mixed":
            # interleave tool trajectories (replay keeps the story prior
            # from being wiped by narrow tool finetuning)
            tool_ids = get_data_and_tokenizer(
                argparse.Namespace(**{**vars(args), "data": "synthetic",
                                       "tokenizer": None}))[0]
            pad = args.max_len - tool_ids.shape[1]
            if pad > 0:
                tool_ids = torch.cat(
                    [tool_ids,
                     torch.full((tool_ids.shape[0], pad), tok.pad_id,
                                dtype=torch.long)], dim=1)
            ids = torch.cat([ids, tool_ids], dim=0)
            perm = torch.randperm(ids.shape[0])
            ids = ids[perm]
            print(f"mixed tensor {tuple(ids.shape)} "
                  f"({i} stories + {tool_ids.shape[0]} tool)")
        return ids, tok
    else:
        raise ValueError(args.data)


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
    ap.add_argument("--init-ckpt", default=None,
                    help="checkpoint dir to initialize weights from (fresh run, e.g. finetune)")
    ap.add_argument("--tokenizer", default=None,
                    help="tokenizer.json to use instead of building one")
    ap.add_argument("--data", default="synthetic",
                    choices=["synthetic", "tinystories", "mixed"])
    ap.add_argument("--data-path", default="data/TinyStories-valid.txt")
    ap.add_argument("--max-stories", type=int, default=None)
    ap.add_argument("--mix-stories", type=int, default=2000,
                    help="stories interleaved with tool data for --data mixed")
    ap.add_argument("--pin-kb", action="store_true",
                    help="pin KB words into vocab (appended, ids never shift)")
    ap.add_argument("--vocab-stories", type=int, default=200000,
                    help="stories used for vocab build (streamed, not all)")
    ap.add_argument("--vocab-size", type=int, default=4000)
    ap.add_argument("--device", default="auto",
                    help="'auto', 'cuda', or 'cpu'")
    args = ap.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)

    print("loading data...")
    ids, tok = get_data_and_tokenizer(args)
    print(f"vocab size: {len(tok)}")

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
    elif args.init_ckpt:
        ckpt = torch.load(f"{args.init_ckpt}/model.pt", map_location="cpu",
                          weights_only=False)
        sd = ckpt["state_dict"]
        # vocab may have grown (pinned KB words appended after the original
        # ids); copy the overlapping embedding rows, init the rest fresh.
        # head.weight is tied to tok_emb.weight, so fix both keys.
        old_n = sd["tok_emb.weight"].shape[0]
        new_n = model.tok_emb.weight.shape[0]
        if old_n != new_n:
            print(f"growing embeddings {old_n} -> {new_n} "
                  f"(copying {min(old_n, new_n)} rows)")
            new_emb = model.tok_emb.weight.data
            n = min(old_n, new_n)
            new_emb[:n] = sd["tok_emb.weight"][:n]
            sd["tok_emb.weight"] = new_emb
            sd["head.weight"] = new_emb
        model.load_state_dict(sd)
        print(f"initialized weights from {args.init_ckpt} (fresh run)")
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    model.to(device)
    print(f"device: {device}")
    print(f"params: {model.n_params():,}")
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)
    gen = torch.Generator(device=device).manual_seed(args.seed)

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
        batch = batch.to(device)
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
