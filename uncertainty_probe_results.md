# Uncertainty-gating probe — results

**Question:** does the denoising confidence trajectory of a call block predict
whether the emitted tool call is correct? (Idea #1 from the 2026-10-19
diffusion-ideas discussion; the "nearly free" prototype using existing eval data.)

**Method:** `uncertainty_probe.py` (untracked, local-only) replays the exact
60-question eval on the local s4 checkpoint (seeds identical to eval.py —
replicated scores: valid 51/60 = 85%, correct_tool 47/60 = 78%, matching the
reported s4 numbers). For each first call block it records per-token birth
confidence (confidence at the denoising step each token unmasked) and the
per-step mean confidence trajectory, then scores features against the
correct_tool label (47 correct / 13 wrong).

**Headline: yes — token-level confidence in the [CALL] span predicts
correctness. AUC = 0.876, bootstrap 95% CI [0.679, 0.994].**
Mean weakest-token confidence: 0.995 (correct calls) vs 0.661 (wrong calls).
Correct calls are near-certain on every span token; wrong calls almost always
contain at least one wobbly token.

**Refinement of the original idea:** the signal is in the *weakest token of the
call span*, not in the trajectory. Per-step trajectory features (min/mean
confidence per denoising step) score AUC 0.493 — coin flip. So it's not
"watch the trajectory collapse mid-denoise"; it's "watch the least-confident
token in the call span."

**Gate operating points** (abort the call if span_min < threshold):
| thr | wrong caught | correct kept | abort precision |
|-----|--------------|--------------|-----------------|
| 0.5 | 4/13 (31%)   | 47/47 (100%) | 1.00            |
| 0.8 | 8/13 (62%)   | 46/47 (98%)  | 0.89            |

**What it catches:** degenerate blocks (`calc 141 [RESULT] * answer 06`,
span_min 0.17), digit scrambles (`calc 89 - 22401`, `calc 8379 * 46`),
wrong-tool choices (`lookup capital of light`). Coherent mechanism: when the
model is unsure about exact tokens (digits, args), confidence drops on those
tokens — the same independent-unmasking weakness behind the copying failures.

**Caveats:** s4 only, 60 questions, 13 wrong cases (wide CI). Needs validation
on other checkpoints before it's a design claim. The abort *action* (ask for
clarification vs re-sample the block) is still an open design choice — the
numbers support confidence as a verifier signal, not a specific policy.

**Fit:** plugs directly into the structure-first framing as the abort signal
between the structure phase and the content phase.
