"""Masked (discrete) diffusion: training loss and confidence-based sampler.

Training (the LLaDA recipe): sample t ~ Uniform(t_min, 1), mask each non-pad token
independently with probability t, and train the model to recover the original
tokens at masked positions only.

Sampling: start from an all-[MASK] response block with the prompt frozen, then
iteratively unmask the most confident predictions until the block is filled.
"""

import torch
import torch.nn.functional as F


def random_mask(ids, mask_id, pad_id, t_min=0.15, generator=None):
    """Return (masked_ids, target_mask). Each non-pad token masked w.p. t."""
    B, T = ids.shape
    t = torch.rand(B, 1, generator=generator, device=ids.device) * (1 - t_min) + t_min
    do_mask = (torch.rand(B, T, generator=generator, device=ids.device) < t)
    do_mask &= (ids != pad_id)
    masked = ids.clone()
    masked[do_mask] = mask_id
    return masked, do_mask


def diffusion_loss(model, ids, tok, t_min=0.15, generator=None):
    masked, target = random_mask(ids, tok.mask_id, tok.pad_id, t_min, generator)
    logits = model(masked, pad_mask=(ids == tok.pad_id))
    loss = F.cross_entropy(logits[target], ids[target])
    return loss


def _top_p_filter(logits, top_p):
    if top_p >= 1.0:
        return logits
    sorted_logits, sorted_idx = torch.sort(logits, descending=True, dim=-1)
    probs = torch.softmax(sorted_logits, dim=-1)
    cumulative = torch.cumsum(probs, dim=-1)
    remove = cumulative - probs > top_p
    sorted_logits[remove] = float("-inf")
    out = torch.full_like(logits, float("-inf")).scatter(-1, sorted_idx, sorted_logits)
    return out


@torch.no_grad()
def generate(model, tok, prompt_ids, gen_len=64, steps=24, temperature=1.0,
             top_p=0.9, seed=0, frames=False, stop_on=None,
             phase1_steps=0, struct_ids=(), struct_min_conf=0.0, log=None,
             struct_bonus=0.0, struct_bonus_steps=0):
    """Denoise a response block conditioned on a frozen prompt.

    Returns the full id list (prompt + generated). With frames=True, also returns
    decoded snapshots of the denoising trajectory. With stop_on set to a token
    id, the block ends right after that token is unmasked (lets a tool loop
    intercept [CALL]...[/CALL] before the model can hallucinate a [RESULT]).

    With phase1_steps > 0: structure-first decoding. The first phase1_steps
    iterations may only unmask positions whose argmax prediction is a structural
    token (from struct_ids); the sampled token is restricted to the structural
    set, then frozen. Remaining steps unmask normally. If struct_min_conf > 0,
    a structural position must also reach that confidence, otherwise phase 1
    ends early instead of forcing structure the model isn't sure about.

    With log set to a list, each unmask event appends (step, position,
    token_id) — leftover argmax fills are logged with step == steps.

    With struct_bonus > 0: structure-biased decoding. For the first
    struct_bonus_steps iterations, positions whose argmax prediction is a
    structural token get struct_bonus added to their selection confidence.
    Pure ordering nudge — sampling stays unrestricted and nothing is frozen
    ahead of schedule. (Do not combine with phase1_steps.)
    """
    g = torch.Generator().manual_seed(seed)
    device = next(model.parameters()).device
    if device.type != "cpu":
        g = torch.Generator(device=device).manual_seed(seed)
    P = len(prompt_ids)
    seq = torch.full((1, P + gen_len), tok.pad_id, dtype=torch.long, device=device)
    seq[0, :P] = torch.tensor(prompt_ids, device=device)
    seq[0, P:] = tok.mask_id
    frozen = torch.zeros(1, P + gen_len, dtype=torch.bool, device=device)
    frozen[0, :P] = True  # prompt never changes — the "frozen block"
    struct_set = set(struct_ids)
    sids = torch.tensor(sorted(struct_set), dtype=torch.long, device=device) \
        if struct_set else None

    def _finish_step(idx):
        frozen[0, idx] = True
        if frames:
            snapshots.append(tok.decode(seq[0].tolist()))
        if stop_on is not None:
            hit = ((seq[0, P:] == stop_on) & frozen[0, P:]).nonzero(as_tuple=True)[0]
            if len(hit):
                end = P + int(hit[0]) + 1
                return seq[:, :end], frozen[:, :end], True
        return seq, frozen, False

    snapshots = []
    for i in range(steps):
        logits = model(seq, pad_mask=(seq == tok.pad_id))[0] / max(temperature, 1e-6)
        logits = _top_p_filter(logits, top_p)
        probs = torch.softmax(logits, dim=-1)
        conf, pred = probs.max(dim=-1)
        active = (~frozen[0]) & (seq[0] == tok.mask_id)
        n_active = int(active.sum())
        if n_active == 0:
            break
        # --- phase 1: structure only ---
        if phase1_steps and i < phase1_steps and sids is not None:
            is_struct = torch.zeros_like(active)
            for sid in struct_set:
                is_struct |= (pred == sid)
            eligible = active & is_struct
            if struct_min_conf > 0:
                eligible &= conf >= struct_min_conf
            if int(eligible.sum()) == 0:
                phase1_steps = 0  # nothing structural to commit: go to phase 2
            else:
                n_unmask = max(1, -(-int(eligible.sum()) // (steps - i)))
                sel = conf.masked_fill(~eligible, -1.0)
                _, idx = torch.topk(sel, n_unmask)
                # sample within the structural set only, renormalized
                sp = probs[idx][:, sids]
                sp = sp / sp.sum(dim=-1, keepdim=True).clamp_min(1e-9)
                picks = torch.multinomial(sp, 1, generator=g).squeeze(-1)
                for j, k in zip(idx.tolist(), picks.tolist()):
                    seq[0, j] = int(sids[k])
                    if log is not None:
                        log.append((i, j, int(sids[k])))
                seq, frozen, done = _finish_step(idx)
                if done:
                    break
                continue
        # --- phase 2 (or normal): any-order confidence unmasking ---
        n_unmask = max(1, -(-n_active // (steps - i)))  # ceil division
        sel_conf = conf.masked_fill(~active, -1.0)
        if struct_bonus and struct_bonus_steps and i < struct_bonus_steps \
                and sids is not None:
            is_struct = torch.zeros_like(active)
            for sid in struct_set:
                is_struct |= (pred == sid)
            sel_conf = sel_conf + struct_bonus * is_struct.float()
        _, idx = torch.topk(sel_conf, n_unmask)
        for j in idx.tolist():
            tid = int(torch.multinomial(probs[j], 1, generator=g))
            seq[0, j] = tid
            if log is not None:
                log.append((i, j, tid))
        seq, frozen, done = _finish_step(idx)
        if done:
            break

    # fill any leftover masks with the argmax prediction (never [MASK] itself)
    leftover = (seq[0] == tok.mask_id).nonzero(as_tuple=True)[0]
    if len(leftover):
        logits = model(seq, pad_mask=(seq == tok.pad_id))[0]
        logits[:, tok.mask_id] = float("-inf")
        fill = logits[leftover].argmax(dim=-1)
        seq[0, leftover] = fill
        if log is not None:
            for j, tid in zip(leftover.tolist(), fill.tolist()):
                log.append((steps, j, int(tid)))
        if frames:
            snapshots.append(tok.decode(seq[0].tolist()))

    ids = seq[0].tolist()
    return (ids, snapshots) if frames else ids
