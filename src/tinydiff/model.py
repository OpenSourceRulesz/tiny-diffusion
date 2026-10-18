"""Small bidirectional transformer LM (BERT-style: no causal mask)."""

import torch
import torch.nn as nn


class TinyDiffusionLM(nn.Module):
    def __init__(self, vocab_size, d_model=256, n_layers=4, n_heads=4,
                 max_len=256, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        self.max_len = max_len
        self.tok_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Embedding(max_len, d_model)
        layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads,
            dim_feedforward=d_model * 4, dropout=dropout,
            batch_first=True, activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=n_layers)
        self.ln = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size, bias=False)
        self.head.weight = self.tok_emb.weight  # tied embeddings

    def forward(self, ids, pad_mask=None):
        B, T = ids.shape
        if T > self.max_len:
            raise ValueError(f"sequence length {T} exceeds max_len {self.max_len}")
        x = self.tok_emb(ids) + self.pos_emb(
            torch.arange(T, device=ids.device))[None, :, :]
        h = self.encoder(x, src_key_padding_mask=pad_mask)
        return self.head(self.ln(h))

    def n_params(self):
        return sum(p.numel() for p in self.parameters())
