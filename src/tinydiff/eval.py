"""Evaluate tool-use: fresh questions, real tool loop, checkable answers.

Metrics per question:
  - valid_call: the loop executed a well-formed call that returned a real result
  - correct_tool: the model chose the right tool with the exact arguments
  - correct_answer: the MODEL's own answer span (text after the last tool
    [RESULT], not the tool output itself) contains the expected value
"""

import argparse
import random

from .diffusion import generate  # noqa: F401  (kept for API symmetry)
from .sample import load_ckpt, tool_loop
from .tools import KB, run_call


def make_items(n=60, seed=999):
    rng = random.Random(seed)
    items = []
    for _ in range(n // 2):
        a, b = rng.randint(100, 9999), rng.randint(100, 9999)
        op = rng.choice(["+", "-", "*"])
        expr = f"{a} {op} {b}"
        q = rng.choice([f"What is {a} {op} {b}?", f"Compute {a} {op} {b}."])
        items.append((q, "calc", expr, run_call(f"calc {expr}")))
    for _ in range(n - n // 2):
        key = rng.choice(sorted(KB.keys()))
        q = rng.choice([f"What is the {key}?", f"Tell me the {key}."])
        items.append((q, "lookup", key, run_call(f"lookup {key}")))
    rng.shuffle(items)
    return items


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="checkpoints/s2")
    ap.add_argument("--n", type=int, default=60)
    ap.add_argument("--seed", type=int, default=999)
    args = ap.parse_args()

    model, tok = load_ckpt(args.ckpt)
    items = make_items(args.n, args.seed)

    valid = tool_ok = ans_ok = 0
    for i, (q, exp_tool, exp_args, exp_val) in enumerate(items):
        final, transcript = tool_loop(
            model, tok, f"[USER] {q} [ASSIST]", seed=1000 + i)
        calls = [t for role, t in transcript if role == "tool"]
        v = bool(calls) and "unknown" not in calls[0] and "error" not in calls[0]
        t_ok = v and calls[0].split()[0] == exp_tool
        # for calc, also require the exact expression (unseen numbers!)
        if t_ok and exp_tool == "calc":
            t_ok = calls[0].split("=>")[0].replace(" ", "") == f"calc{exp_args.replace(' ', '')}"
        # the model's own answer: text after the last tool [RESULT]
        answer_span = final.rsplit("[/RESULT]", 1)[-1] if "[/RESULT]" in final else final
        a_ok = exp_val in answer_span
        valid += v
        tool_ok += t_ok
        ans_ok += a_ok
        flag = "" if (v and t_ok and a_ok) else "   <-- MISS"
        print(f"{i+1:2d}. [{q}] call={calls[0] if calls else '-'} "
              f"ans_in_final={a_ok}{flag}")

    n = len(items)
    print(f"\nvalid_call:     {valid}/{n} = {valid/n:.2f}")
    print(f"correct_tool:   {tool_ok}/{n} = {tool_ok/n:.2f}")
    print(f"correct_answer: {ans_ok}/{n} = {ans_ok/n:.2f}")


if __name__ == "__main__":
    main()
