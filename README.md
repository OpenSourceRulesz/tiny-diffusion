# tiny-diffusion

A tiny, hackable **text diffusion language model** — the `tiny-jepa` playbook applied to
diffusion LMs. Discrete masked diffusion (the LLaDA recipe) at toy scale: a small
bidirectional transformer trained to denoise masked token sequences, plus a
**frozen-block tool loop** that shows off what diffusion can do that autoregressive
models can't — surgically re-denoising only the tokens affected by a tool result,
instead of regenerating everything.

## Why

Open text diffusion models (LLaDA, Dream 7B, Fast-dLLM v2, DiffusionGemma) exist, but:

1. They re-process the whole context on every denoising step — no KV-cache-style
   efficiency for agentic loops (many short generations re-reading the trajectory).
2. None are fine-tuned for **tool use**.

tiny-diffusion is the small-scale lab for both ideas: block-wise frozen decoding
(keep finished blocks fixed, only denoise the new block — the diffusion-native
answer to KV caching) and tool-call trajectories as first-class training data.

## Quickstart

```bash
pip install -r requirements.txt        # torch (CPU build is fine)
python -m tinydiff.train --steps 600   # smoke-train on synthetic tool trajectories
python -m tinydiff.sample --prompt "What is 238 * 47 + 12?" --tool-loop --frames
```

## Layout

- `src/tinydiff/tokenizer.py` — word-level tokenizer with protected special tokens
- `src/tinydiff/model.py` — small bidirectional transformer (BERT-style, no causal mask)
- `src/tinydiff/diffusion.py` — masked-diffusion loss + confidence-based sampler
- `src/tinydiff/tools.py` — `calc` and `lookup` tools + call parser
- `src/tinydiff/data.py` — synthetic tool-use trajectory generator
- `src/tinydiff/train.py` — training loop
- `src/tinydiff/sample.py` — sampling, denoising frames, frozen-block tool loop
- `tests/test_smoke.py` — pipeline smoke tests

## The research bets

1. **Frozen-block agentic loop.** In a ReAct trajectory, freeze finished blocks as
   context and denoise only the new action block. Diffusion's native operation is
   masked infilling — this is cheaper than full re-denoising and impossible for AR.
2. **Surgical re-denoising.** When a tool result arrives, re-mask only the tokens
   that depended on it instead of regenerating the whole answer. (Implemented as
   block granularity here; token-level dependency tracking is the follow-up.)
3. **Tool-use fine-tuning for diffusion LMs.** Synthetic, verifiable tool
   trajectories (deterministic tools ⇒ checkable correctness) as training data —
   nobody has done this for an open diffusion LM at any scale.

## Results (s4 checkpoint, 1.8M params, 8000 steps)

Evaluated on 60 fresh questions with **unseen numbers** (never in training):

| metric | score | what it means |
|---|---|---|
| valid tool call | 51/60 = **0.85** | loop executed a well-formed call, real result returned |
| correct tool + exact args | 47/60 = **0.78** | right tool, verbatim expression/key copied |
| correct answer written by model | 7/60 = **0.12** | model's own answer span contains the value |

Honest read: the diffusion LM **learns tool selection and argument copying**
— the agentic loop works end to end on the decision side. The bottleneck is the
model reproducing the result value in its own answer span (verbatim copying
through denoising is the known hard part for diffusion LMs; s5 tested an `[END]`
terminator token — answers terminate cleanly but tool accuracy dipped to 57%,
so s4 remains the headline checkpoint).

Example end-to-end run (s4):

```
[model] [THINK] I will look that up in the knowledge base. [/THINK] [CALL] lookup speed of light [/CALL]
[tool]  lookup speed of light  =>  299792458 meters per second
[model] So the answer is 299792458 meters per second.
```

## Hard-won lessons

1. **Digit-split tokenization.** Word-level tokenizers map unseen numbers to
   `[UNK]` — the model literally cannot see them. Splitting numbers into digit
   tokens (rejoined on decode) makes any number representable.
2. **Stop the block at `[/CALL]`.** Otherwise the model hallucinates its own
   `[RESULT]` and the loop never gets to execute the real tool. The environment
   must own `[RESULT]`.
3. **Finished blocks stay frozen.** The prompt + prior turns are never
   re-denoised — the diffusion-native answer to KV caching. Only the new block
   costs compute.
4. **Capacity matters more than steps.** A 657K-param model memorized the format
   but not the question→tool mapping (2% tool accuracy); 1.8M params got 78%.

## Roadmap

- [x] Toy masked-diffusion LM + synthetic tool trajectories + tool loop (this repo)
- [x] Eval harness with checkable metrics (`python -m tinydiff.eval`)
- [x] `[END]` terminator experiment (s5: clean termination, but tool accuracy 57% < s4's 78%)
- [ ] TinyStories pre-training for fluent English, then tool-trajectory fine-tune
- [ ] Token-level surgical re-denoising (dependency-tracked remasking)
- [ ] Cost-per-task benchmark vs an autoregressive baseline on tool tasks
- [ ] Scale the recipe to LLaDA-8B: block-wise KV cache + tool loop on real weights
