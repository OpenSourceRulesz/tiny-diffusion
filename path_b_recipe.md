# Path B: frozen-block tool loop on a real open diffusion LM

Date: 2026-10-19. Status: PREP ONLY — no notebook written yet; waiting on the user's go-ahead.
Context: user asked (2026-10-19 17:59 UTC) whether the agent2 results say "scaling up a bit or
using a larger diffusion model" would make this useful. Main agent answered straight and offered
to prep a one-click notebook adapting the frozen-block tool loop to Hugging Face's LLaDA.
This doc is the verified groundwork so "yes" can turn into a notebook fast.

## The honest framing (already said to the user; do not repeat as new)

- The toy is done. Its job was recipe + traps: pin the vocab, intercept at [/CALL],
  watch digit-copying. Nothing the toy can add would change the scaling question.
- Scaling evidence points toward trying: 657K params → 2% tool accuracy, 1.8M → 78% (capacity
  was the biggest lever of the whole project), and LLaDA2.0-mini beats Qwen3-8B on BFCL v3
  (70.90 vs 70.08, verified via the deep-research report of 2026-10-19).
- The open question is whether precision-copying (digits into calls, values into answers)
  yields to capacity — plausibly yes, not proven.

## Verified facts

1. **Weights are public on HF.** Base repo `GSAI-ML/LLaDA-8B-Instruct`, loadable via
   transformers (verified loading pattern: `AutoModel.from_pretrained('GSAI-ML/LLaDA-8B-Instruct')`
   with `torch.bfloat16` + `trust_remote_code=True`; transformers==4.38.2 era).
   [source: DeepWiki grinco/LLaDA model-variants page, 2026-10-19 web check]
2. **Block-wise semi-AR decoding is native to LLaDA's official `generate()`.**
   `generate(model, prompt, steps, gen_length, block_length, remasking='low_confidence')` —
   when `block_length < gen_length` it decodes blocks left-to-right with diffusion inside each
   block. The LLaDA Instruct evals themselves use block_length=32 semi-AR.
   [source: DeepWiki shuyingte/DCD LLaDA-generation page + LLaDA paper §B, 2026-10-19 web check]
   → This maps 1:1 onto our frozen-block tool loop: finish a block, check whether it emits
   the call marker, run the tool, freeze [RESULT] back into the canvas, continue.
3. **Precedent exists.** `Proximile/LLaDA-8B-Tools` (HF, May 2025) is a LoRA SFT of
   LLaDA-8B-Instruct on ToolACE specifically for JSON tool invocation. Two uses: (a) proof the
   path has been walked; (b) we can test this checkpoint zero-shot as a shortcut — maybe no
   fine-tune is needed at all for the calc/lookup eval.
   [source: https://huggingface.co/Proximile/LLaDA-8B-Tools, 2026-10-19 web check]
4. **Bigger options (not T4-sized).** `inclusionAI/LLaDA2.0-mini` is open (Apache-2.0), 16B MoE
   with the same block_length API in `model.generate()` — but 16B needs ~32GB bf16, so this
   is an A100/rented-GPU step, not the free-Colab step. Mercury is commercial API-only
   (closed weights) → out of bounds for the $0 project rule.
   [source: https://huggingface.co/inclusionAI/LLaDA2.0-mini, 2026-10-19 web check]

## Colab T4 VRAM plan

- LLaDA-8B in bf16 ≈ 16GB. A Colab T4 has ~15-16GB → marginal: may run for short
  tool-loop sequences (a few hundred tokens), may OOM on longer ones.
- Fallback ladder: (1) try bf16 as the official example does; (2) bitsandbytes 8-bit
  (~8.6GB, comfortable) — standard practice but NOT yet verified on this model family, test
  at runtime; (3) if both fail, the experiment needs an A100 rental for an hour.
- Dream-7B (HKUNLP/Dream-base-7B, init from Qwen2.5-7B) is the alternative 7B checkpoint;
  same VRAM math, less tool-call precedent. LLaDA first.

## Proposed notebook recipe (when he says go)

1. Mount Drive, pip install transformers+torch (Colab default torch is fine; add bitsandbytes only if bf16 OOMs).
2. Load `GSAI-ML/LLaDA-8B-Instruct` bf16 on cuda.
3. Port the frozen-block loop from ~/workspace/tiny-diffusion/src (block_length=32, steps=64-128,
   low_confidence remasking, temp 0.0 as LLaDA2.0's best-practice).
4. Zero-shot baseline FIRST: run the same 60-question calc/lookup eval on the Instruct model
   with the loop — measure valid_call / correct_tool / correct_answer before any training.
   The Instruct model already knows what "call calc 1411 * 9402" means; the loop just
   enforces where the tool result goes. This is the test that answers "does the larger
   diffusion model get better" in one run, ~1-2 hours.
5. Optional branch (only if zero-shot disappoints): LoRA SFT on the 8k synthetic tool
   trajectories with the LLaDA 2.0 recipe pieces (complementary masking, clipped t-bandwidth,
   prompt-clean/response-masked) — or test Proximile's checkpoint instead of training.

## Decisions that stay with the user

- Go-ahead to write the notebook (nothing built until he says "prep it").
- If T4 OOMs and 8-bit is lossy/unusable: approve ~$1-3 of A100 rental for the run, or stop.
- Long-term: whether this becomes a tiny-diffusion Path B page/branch (local-only either way;
  tiny-diffusion stays unpushed until he approves anything public).
