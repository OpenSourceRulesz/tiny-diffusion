"""Uncertainty-gating probe: does the confidence trajectory of a denoising call
block predict whether the emitted tool call is correct?

Runs the frozen-block tool loop over the 60 eval items on a local checkpoint
(identical seeds/items to eval.py) and records, for the first generated block:
  - per-token birth confidence (confidence when the token unmasked)
  - per-step mean confidence of newly-unmasked tokens (the trajectory)
Then: do confidence features separate correct_tool from wrong calls?
"""

import re
import sys

import torch

from src.tinydiff.diffusion import _top_p_filter
from src.tinydiff.eval import make_items
from src.tinydiff.sample import load_ckpt
from src.tinydiff.tools import run_call


@torch.no_grad()
def generate_with_conf(model, tok, prompt_ids, gen_len=40, steps=16,
                       temperature=0.8, top_p=0.9, seed=0, stop_on=None):
    """diffusion.generate + confidence-trajectory recording.

    Returns (full_ids, conf) with conf = {
      'birth_conf': confidence of each token when unmasked (None for prompt),
      'birth_step': denoising step each position unmasked,
      'step_mean':  mean confidence of tokens unmasked at each step }
    """
    g = torch.Generator().manual_seed(seed)
    device = next(model.parameters()).device
    P = len(prompt_ids)
    seq = torch.full((1, P + gen_len), tok.pad_id, dtype=torch.long, device=device)
    seq[0, :P] = torch.tensor(prompt_ids, device=device)
    seq[0, P:] = tok.mask_id
    frozen = torch.zeros(1, P + gen_len, dtype=torch.bool, device=device)
    frozen[0, :P] = True

    birth_conf = [None] * (P + gen_len)
    birth_step = [-1] * (P + gen_len)
    step_mean = []

    for i in range(steps):
        logits = model(seq, pad_mask=(seq == tok.pad_id))[0] / max(temperature, 1e-6)
        logits = _top_p_filter(logits, top_p)
        probs = torch.softmax(logits, dim=-1)
        conf, _ = probs.max(dim=-1)
        active = (~frozen[0]) & (seq[0] == tok.mask_id)
        n_active = int(active.sum())
        if n_active == 0:
            break
        n_unmask = max(1, -(-n_active // (steps - i)))
        conf = conf.masked_fill(~active, -1.0)
        _, idx = torch.topk(conf, n_unmask)
        chosen = idx.tolist()
        step_mean.append(float(conf[chosen].mean()))
        for j in chosen:
            seq[0, j] = int(torch.multinomial(probs[j], 1, generator=g))
            birth_conf[j] = float(probs[j, int(seq[0, j])])
            birth_step[j] = i
        frozen[0, idx] = True
        if stop_on is not None:
            hit = ((seq[0, P:] == stop_on) & frozen[0, P:]).nonzero(as_tuple=True)[0]
            if len(hit):
                end = P + int(hit[0]) + 1
                seq, frozen = seq[:, :end], frozen[:, :end]
                break

    leftover = (seq[0] == tok.mask_id).nonzero(as_tuple=True)[0]
    if len(leftover):
        logits = model(seq, pad_mask=(seq == tok.pad_id))[0]
        logits[:, tok.mask_id] = float("-inf")
        probs = torch.softmax(logits, dim=-1)
        for j in leftover.tolist():
            seq[0, j] = int(logits[j].argmax())
            birth_conf[j] = float(probs[j, int(seq[0, j])])
            birth_step[j] = steps

    ids = seq[0].tolist()
    return ids, {"birth_conf": birth_conf[:len(ids)],
                 "birth_step": birth_step[:len(ids)],
                 "step_mean": step_mean}


def tool_loop_with_conf(model, tok, prompt_text, block_len=40, answer_len=16,
                        steps=16, max_rounds=5, temperature=0.8, seed=0):
    seq_ids = tok.encode(prompt_text)
    call_stop = tok.stoi["[/CALL]"]
    end_stop = tok.stoi.get("[END]")
    transcript = []
    rounds = []
    just_answered_tool = False
    for r in range(max_rounds):
        blen = answer_len if just_answered_tool else block_len
        stop_id = end_stop if just_answered_tool else call_stop
        if len(seq_ids) + blen > model.max_len:
            break
        gen_ids, conf = generate_with_conf(
            model, tok, seq_ids, gen_len=blen, steps=steps,
            temperature=temperature, seed=seed + r, stop_on=stop_id)
        new_text = tok.decode(gen_ids[len(seq_ids):])
        rounds.append({"text": new_text, "conf": conf, "ids": gen_ids,
                       "prompt_len": len(seq_ids)})
        seq_ids = gen_ids
        transcript.append(("model", new_text))
        m = re.search(r"\[CALL\](.*?)\[/CALL\]", new_text)
        just_answered_tool = False
        if not m:
            break
        result = run_call(m.group(1).strip())
        transcript.append(("tool", f"{m.group(1).strip()}  =>  {result}"))
        seq_ids = seq_ids + tok.encode(f" [RESULT] {result} [/RESULT]")
        just_answered_tool = True
    return tok.decode(seq_ids), transcript, rounds


def block_features(round_rec, tok):
    """Confidence features of the [CALL]...[/CALL] span inside a round."""
    ids = round_rec["ids"]
    P = round_rec["prompt_len"]
    text = round_rec["text"]
    m = re.search(r"\[CALL\](.*?)\[/CALL\]", text)
    if not m:
        return None
    gen_ids = ids[P:]
    bc = round_rec["conf"]["birth_conf"][P:]
    # map tokens to text offsets via incremental decode
    span_idx = []
    cum = ""
    for k, tid in enumerate(gen_ids):
        prev = len(cum)
        cum += tok.decode([tid])
        if prev < m.end() and len(cum) > m.start():
            span_idx.append(k)
    span_conf = [bc[k] for k in span_idx if bc[k] is not None]
    all_conf = [c for c in bc if c is not None]
    step_mean = round_rec["conf"]["step_mean"]
    return {
        "span_n": len(span_idx),
        "span_mean": sum(span_conf) / len(span_conf) if span_conf else None,
        "span_min": min(span_conf) if span_conf else None,
        "block_mean": sum(all_conf) / len(all_conf) if all_conf else None,
        "block_min": min(all_conf) if all_conf else None,
        "traj_min": min(step_mean) if step_mean else None,
        "traj_final": step_mean[-1] if step_mean else None,
    }


def auc(scores, labels):
    """Rank-based AUC of score predicting label==1."""
    paired = sorted(zip(scores, labels), key=lambda x: x[0])
    n1 = sum(labels)
    n0 = len(labels) - n1
    if not n1 or not n0:
        return None
    rank_sum = 0.0
    for r, (_, lab) in enumerate(paired, start=1):
        if lab:
            rank_sum += r
    return (rank_sum - n1 * (n1 + 1) / 2) / (n1 * n0)


def main():
    ckpt = sys.argv[1] if len(sys.argv) > 1 else "checkpoints/s4"
    n = int(sys.argv[2]) if len(sys.argv) > 2 else 60
    model, tok = load_ckpt(ckpt)
    items = make_items(n, 999)

    feats, labels, calls = [], [], []
    valid = tool_ok = 0
    for i, (q, exp_tool, exp_args, exp_val) in enumerate(items):
        final, transcript, rounds = tool_loop_with_conf(
            model, tok, f"[USER] {q} [ASSIST]", seed=1000 + i)
        t_calls = [t for role, t in transcript if role == "tool"]
        v = bool(t_calls) and "unknown" not in t_calls[0] and "error" not in t_calls[0]
        t_ok = v and t_calls[0].split()[0] == exp_tool
        if t_ok and exp_tool == "calc":
            t_ok = (t_calls[0].split("=>")[0].replace(" ", "")
                    == f"calc{exp_args.replace(' ', '')}")
        valid += v
        tool_ok += t_ok
        f = block_features(rounds[0], tok) if rounds else None
        if f and f["span_mean"] is not None:
            feats.append(f)
            labels.append(1 if t_ok else 0)
            calls.append(t_calls[0] if t_calls else "-")
        if (i + 1) % 15 == 0:
            print(f"... {i+1}/{n}  valid={valid} tool_ok={tool_ok}", flush=True)

    print(f"\nreplicated eval: valid={valid}/{n} tool_ok={tool_ok}/{n}")
    print(f"call blocks analyzed: {len(feats)} "
          f"({sum(labels)} correct, {len(labels)-sum(labels)} wrong)")

    print("\nAUC of each confidence feature predicting correct_tool:")
    for key in ["span_mean", "span_min", "block_mean", "block_min",
                "traj_min", "traj_final"]:
        vals = [f[key] for f in feats]
        print(f"  {key:10s} AUC={auc(vals, labels):.3f}")

    # gate simulation: block the call if span_min < threshold
    print("\nGate simulation (abort call if span_min < thr):")
    print("  thr    caught_wrong  kept_correct  precision_of_abort")
    wrong = [i for i, l in enumerate(labels) if not l]
    right = [i for i, l in enumerate(labels) if l]
    for thr in [0.3, 0.4, 0.5, 0.6, 0.7, 0.8]:
        aborted = [i for i in range(len(feats)) if feats[i]["span_min"] < thr]
        caught = sum(1 for i in aborted if i in wrong)
        kept = sum(1 for i in right if i not in aborted)
        prec = caught / len(aborted) if aborted else 0
        print(f"  {thr:.1f}    {caught:2d}/{len(wrong):2d}         "
              f"{kept:2d}/{len(right):2d}        {prec:.2f}")

    print("\nWorst-confidence wrong calls (span_min):")
    for i in sorted(wrong, key=lambda i: feats[i]["span_min"])[:8]:
        print(f"  span_min={feats[i]['span_min']:.3f}  call={calls[i]}")
    print("Lowest-confidence correct calls (span_min):")
    for i in sorted(right, key=lambda i: feats[i]["span_min"])[:5]:
        print(f"  span_min={feats[i]['span_min']:.3f}  call={calls[i]}")


if __name__ == "__main__":
    main()
