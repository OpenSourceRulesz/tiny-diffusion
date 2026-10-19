"""Sample from the tiny diffusion LM: denoising frames + frozen-block tool loop."""

import argparse
import re
import sys

import torch

from .diffusion import generate
from .model import TinyDiffusionLM
from .tokenizer import WordTokenizer
from .tools import run_call


def load_ckpt(path):
    ckpt = torch.load(f"{path}/model.pt", map_location="cpu", weights_only=False)
    tok = WordTokenizer.load(f"{path}/tokenizer.json")
    cfg = ckpt["config"]
    model = TinyDiffusionLM(ckpt["vocab_size"], d_model=cfg["d_model"],
                            n_layers=cfg["layers"], n_heads=cfg["heads"],
                            max_len=cfg["max_len"])
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    return model, tok


def default_struct_ids(tok):
    """Token ids treated as structural in phase 1 of structure-first decoding:
    block markup plus the tool-name decision."""
    names = ["[THINK]", "[/THINK]", "[CALL]", "[/CALL]", "[END]", "calc", "lookup"]
    return [tok.stoi[n] for n in names if n in tok.stoi]


def tool_loop(model, tok, prompt_text, block_len=40, answer_len=16, steps=16,
              max_rounds=5, temperature=0.8, seed=0,
              phase1_steps=0, struct_ids=None, struct_min_conf=0.0, traces=None):
    """Frozen-block agentic loop.

    Each round: freeze everything generated so far (prompt + finished blocks),
    denoise only a fresh response block. If the block contains a complete
    [CALL]...[/CALL], execute it for real, freeze the [RESULT], and continue.
    The model never re-generates finished blocks — the diffusion-native answer
    to KV caching.

    With traces set to a list, each round appends its block's unmask log:
    a list of (step, position, token_id) events.
    """
    seq_ids = tok.encode(prompt_text)
    call_stop = tok.stoi["[/CALL]"]
    # [END] exists only from s5 on (it was added to stop answer rambling).
    # Older checkpoints just run the answer block to full length.
    end_stop = tok.stoi.get("[END]")
    transcript = []
    just_answered_tool = False
    for r in range(max_rounds):
        # the answer after a tool result only needs a few tokens; a long block
        # invites rambling past the answer. It ends at [END], which the model
        # learned as the trajectory terminator.
        blen = answer_len if just_answered_tool else block_len
        stop_id = end_stop if just_answered_tool else call_stop
        if len(seq_ids) + blen > model.max_len:
            break
        # stop the block the moment [/CALL] is unmasked: the environment owns
        # [RESULT], so anything the model would hallucinate after [/CALL] is cut.
        blog = [] if traces is not None else None
        gen_ids = generate(model, tok, seq_ids, gen_len=blen, steps=steps,
                           temperature=temperature, seed=seed + r, stop_on=stop_id,
                           phase1_steps=phase1_steps,
                           struct_ids=struct_ids or (),
                           struct_min_conf=struct_min_conf, log=blog)
        if traces is not None:
            traces.append(blog)
        new_text = tok.decode(gen_ids[len(seq_ids):])
        seq_ids = gen_ids
        transcript.append(("model", new_text))
        m = re.search(r"\[CALL\](.*?)\[/CALL\]", new_text)
        just_answered_tool = False
        if not m:
            break  # no tool call -> answer is complete
        result = run_call(m.group(1).strip())
        transcript.append(("tool", f"{m.group(1).strip()}  =>  {result}"))
        seq_ids = seq_ids + tok.encode(f" [RESULT] {result} [/RESULT]")
        just_answered_tool = True
    return tok.decode(seq_ids), transcript


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="checkpoints/smoke")
    ap.add_argument("--prompt", default="What is 238 * 47?")
    ap.add_argument("--gen-len", type=int, default=48)
    ap.add_argument("--steps", type=int, default=24)
    ap.add_argument("--temperature", type=float, default=0.8)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--frames", action="store_true",
                    help="show denoising trajectory snapshots")
    ap.add_argument("--tool-loop", action="store_true",
                    help="run the frozen-block agentic tool loop")
    args = ap.parse_args()

    model, tok = load_ckpt(args.ckpt)
    prompt = f"[USER] {args.prompt} [ASSIST]"
    prompt_ids = tok.encode(prompt)

    if args.tool_loop:
        final, transcript = tool_loop(
            model, tok, prompt, temperature=args.temperature, seed=args.seed)
        for role, text in transcript:
            tag = "MODEL" if role == "model" else "TOOL "
            print(f"[{tag}] {text}\n")
        print("FINAL:", final)
        return

    ids, snapshots = generate(model, tok, prompt_ids, gen_len=args.gen_len,
                              steps=args.steps, temperature=args.temperature,
                              seed=args.seed, frames=True)
    if args.frames:
        n = len(snapshots)
        picks = sorted(set([0, n // 3, 2 * n // 3, n - 1]))
        for p in picks:
            print(f"--- denoise step {p + 1}/{n} ---")
            print(snapshots[p], "\n")
    print("FINAL:", tok.decode(ids))


if __name__ == "__main__":
    sys.exit(main())
