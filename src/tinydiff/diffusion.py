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
             top_p=0.9, seed=0, frames=False, stop_on=None):
    """Denoise a response block conditioned on a frozen prompt.

    Returns the full id list (prompt + generated). With frames=True, also returns
    decoded snapshots of the denoising trajectory. With stop_on set to a token
    id, the block ends right after that token is unmasked (lets a tool loop
    intercept [CALL]...[/CALL] before the model can hallucinate a [RESULT]).
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

    snapshots = []
    for i in range(steps):
        logits = model(seq, pad_mask=(seq == tok.pad_id))[0] / max(temperature, 1e-6)
        logits = _top_p_filter(logits, top_p)
        probs = torch.softmax(logits, dim=-1)
        conf, _ = probs.max(dim=-1)
        active = (~frozen[0]) & (seq[0] == tok.mask_id)
        n_active = int(active.sum())
        if n_active == 0:
            break
        n_unmask = max(1, -(-n_active // (steps - i)))  # ceil division
        conf = conf.masked_fill(~active, -1.0)
        _, idx = torch.topk(conf, n_unmask)
        for j in idx.tolist():
            seq[0, j] = int(torch.multinomial(probs[j], 1, generator=g))
        frozen[0, idx] = True
        if frames:
            snapshots.append(tok.decode(seq[0].tolist()))
        if stop_on is not None:
            hit = ((seq[0, P:] == stop_on) & frozen[0, P:]).nonzero(as_tuple=True)[0]
            if len(hit):
                end = P + int(hit[0]) + 1
                seq, frozen = seq[:, :end], frozen[:, :end]
                break

    # fill any leftover masks with the argmax prediction (never [MASK] itself)
    leftover = (seq[0] == tok.mask_id).nonzero(as_tuple=True)[0]
    if len(leftover):
        logits = model(seq, pad_mask=(seq == tok.pad_id))[0]
        logits[:, tok.mask_id] = float("-inf")
        seq[0, leftover] = logits[leftover].argmax(dim=-1)
        if frames:
            snapshots.append(tok.decode(seq[0].tolist()))

    ids = seq[0].tolist()
    return (ids, snapshots) if frames else ids
