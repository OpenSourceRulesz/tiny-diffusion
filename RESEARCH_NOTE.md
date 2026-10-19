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
