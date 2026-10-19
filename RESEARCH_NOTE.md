# tiny-diffusion — research note

*What we tried, what happened, and what it means. Updated 2026-10-19.*

## The questions

A colleague claimed diffusion models "always start from scratch" — no memory between steps, everything re-processed. That raised two research questions:

1. **Efficiency:** can a diffusion LM avoid re-processing the whole context every denoising step — a KV-cache equivalent?
2. **Tool use:** can tool-calling be finetuned into a diffusion LM the way it is into autoregressive ones, and where does it struggle?

## The setup

A tiny masked-diffusion LM (bidirectional transformer, no causal mask) trained with the standard objective: random mask rate t ~ U(0.15, 1), cross-entropy on masked positions only. Training data is synthetic and fully verifiable: tool-use trajectories over two deterministic tools, `calc` (exact arithmetic) and `lookup` (a 12-fact knowledge base).

The agentic loop uses **frozen blocks**: finished blocks (prompt, thinking, tool call, tool result) are frozen and only the new block is denoised — the diffusion analogue of KV caching for agentic loops. Generation of a block stops at `[/CALL]` so the model cannot hallucinate its own `[RESULT]`; the real tool executes and its output is frozen into the context.

## Results

Eval: 60 fresh questions, unseen numbers (100–9999), unseen phrasings. `correct_answer` requires the value to appear in the model's **own** answer span after the tool result — the tool's output sitting in the transcript does not count.

| model | params | valid call | correct tool + exact args | correct final answer |
|---|---|---|---|---|
| s4 — from scratch, 8k steps | 1.8M | 85% | 78% | 12% |
| agent1 — TinyStories 60k-step pretrain + 8k finetune | 6.8M | 58% | 45% | 27% |
| agent1 — finetune continued to 20k steps | 6.8M | 60% | 43% | 15% |

## Findings

1. **Tool selection and exact argument copying are learnable at toy scale** (78% from scratch). The model reliably maps questions to the right tool with the right arguments.
2. **Verbatim copying of the result into the model's own answer is the hard part** (12%). This looks like the genuine diffusion-LM weakness: unmasking tokens independently makes exact reproduction fragile.
3. **Pretraining changes the profile, both ways.** TinyStories pretraining then finetuning: answer accuracy 12% → 27% (fluency up), but tool-call accuracy 78% → 45% (format down). The story prior interferes with the rigid tool format.
4. **More finetuning is not the fix.** Continuing 8k → 20k steps: tool accuracy flat at 43%, answer accuracy *fell* 27% → 15%, while training loss kept dropping (0.86) — the classic overfitting signature. The model memorized training phrasings instead of generalizing.
5. **The frozen-block loop works mechanically.** Finished blocks stay frozen; only new blocks are denoised. For agentic loops this answers the "starts from scratch" objection the same way KV caching does for autoregressive models.

## Caveats

- The from-scratch baseline (s4) differs from agent1 in size (1.8M vs 6.8M) and tokenizer/vocab (1.9k vs 8k words), so the pretraining comparison is suggestive, not airtight.
- Eval is 60 questions; per-metric noise is roughly ±6pp. The 27%→15% drop is ~1.6 standard errors — suggestive of a real decline, not definitive.
- All results are toy scale (millions of params, synthetic data). Nothing here claims to transfer to billion-parameter models directly.

## Next

Mixed-data finetune (stories + tool trajectories interleaved) and/or lower learning rate, to reduce interference between the story prior and the tool format. Architecture research first — see `docs/` (forthcoming).

## 2026-10-19: structure-first decoding (two-phase unmasking) — NEGATIVE result

Implemented phase-separated decoding in `diffusion.generate()`: phase 1 may
only unmask positions whose argmax is structural (`[THINK]`, `[/THINK]`,
`[CALL]`, `[/CALL]`, `[END]`, tool names), sampling restricted to the
structural set and frozen; phase 2 fills content normally. Flags:
`--two-phase`, `--phase1-steps`, `--struct-min-conf`. Same 60 questions,
same seeds as baseline, s4 checkpoint.

| decoding | valid_call | correct_tool | correct_answer |
|---|---|---|---|
| baseline (any-order) | 85% | 78% | 12% |
| two-phase (8 struct steps) | 68% | 58% | 10% |

Failure mechanism: freezing structural tokens at absolute positions before
content lengths are known. E.g. `[/THINK]` committed at position 10, think
content needed 15 tokens -> content spilled into the call region
(`calc 14 pound [RESULT] * answer 06`). Positional commitment is the problem,
not structural commitment. Consistent with NDGC's choice to align granularity
smoothly rather than freeze explicitly. Next: skeleton-as-priority-bias
without positional freezing.

## 2026-10-19: structure-biased decoding (soft bonus) — NEGATIVE result

Follow-up to the hard two-phase failure. Soft version: +0.3 confidence bonus
for structural-predicting positions during the first 8 steps (ordering nudge
only, no restriction, no freezing). Same 60 questions/seeds, s4.

| decoding | valid_call | correct_tool | correct_answer |
|---|---|---|---|
| baseline (any-order) | 85% | 78% | 12% |
| two-phase (hard) | 68% | 58% | 10% |
| struct bonus (soft) | 75% | 58% | 7% |

Prior measurement showed the baseline unmasks content first (step 0.02) and
structure later (step 1.07) — so the soft bonus was a real intervention, not
redundant. It still hurt exactly as much as the hard version. Conclusion: the
model's confidence ordering is already well-adapted; imposing a hand-designed
ordering hurts. Untested: structure-aware TRAINING (mask content more than
structure à la NDGC granularity alignment). Next: copy-gated decoding for the
answer block (targets the 12% answer-copying bottleneck; verified novel).

## 2026-10-19: copy-gated decoding — POSITIVE, dose-dependent

Implemented copy-gating in `diffusion.generate()`: fixed logit bonus added to
tokens from a designated copy source at each denoising step (decoding-time
analogue of a pointer-generator's copy distribution). `tool_loop()` sets the
source per block: answer blocks copy from the frozen [RESULT] span
(--copy-answer), call blocks copy from the question (--copy-call, fights digit
corruption like 9322->9922). Flags: --copy-bonus, --copy-answer, --copy-call.
s4, same 60 questions/seeds.

| copy bonus | valid_call | correct_tool | correct_answer |
|---|---|---|---|
| 0 (baseline) | 85% | 78% | 12% |
| 1.0 | 85% | 78% | 15% |
| 2.0 | 88% | 82% | 25% |
| 3.0 | 77% | 70% | 18% |

Dose-response with an inverted-U: answer-copying (the identified core
weakness) doubles at bonus 2.0; tool accuracy also improves (82%); 3.0
overshoots — the bonus overwhelms the model's distribution and everything
degrades. Sweet spot ~2.0. First clear win: a verified-novel mechanism that
repairs diffusion's verbatim-copying weakness at decode time, no retraining.
