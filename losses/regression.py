from __future__ import annotations
import torch
import torch.nn as nn


class Model(nn.Module):
    def __init__(self, cfg_loss):
        super().__init__()
        self.cfg = cfg_loss
        emb_dim = int(cfg_loss.get("emb_dim", 128))
        hidden = max(emb_dim, 256)
        self.head = nn.Sequential(
            nn.Linear(emb_dim, hidden),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden, 1),
        )
        nn.init.zeros_(self.head[-1].bias)
        n_bins = int(cfg_loss.get("n_bins", 2100))
        self.register_buffer("_n_bins", torch.tensor(n_bins))

    def forward(self, tokens, mask, timeline, targets):
        B, P, NPC, D = tokens.shape
        n_bins = int(self._n_bins.item())
        flat = tokens.reshape(B, P * NPC, D)
        flat_mask = mask.reshape(B, P * NPC)
        per_tok_norm = self.head(flat).squeeze(-1)

        bin_to_norm_target = targets.float() / (n_bins - 1)
        target_per_tok = bin_to_norm_target.unsqueeze(-1).expand_as(per_tok_norm)

        per_tok_sq = (per_tok_norm - target_per_tok).pow(2)
        denom = flat_mask.sum().clamp_min(1.0)
        loss = (per_tok_sq * flat_mask).sum() / denom

        loss_comp = [loss.detach()]
        return loss, loss_comp
