from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F


class Model(nn.Module):
    def __init__(self, cfg_loss):
        super().__init__()
        self.cfg = cfg_loss
        emb_dim = int(cfg_loss.get("emb_dim", 128))
        n_bins = int(cfg_loss.get("n_bins", 2100))
        hidden = max(emb_dim, 256)
        self.head = nn.Sequential(
            nn.Linear(emb_dim, hidden),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden, n_bins),
        )
        self.register_buffer("_n_bins", torch.tensor(n_bins))

    def forward(self, tokens, mask, timeline, targets):
        B, P, NPC, D = tokens.shape
        n_bins = int(self._n_bins.item())
        flat = tokens.reshape(B, P * NPC, D)
        flat_mask = mask.reshape(B, P * NPC)

        logits = self.head(flat)

        N = logits.shape[1]
        target = targets.long().clamp(0, n_bins - 1)
        flat_logits = logits.reshape(B * N, n_bins)
        flat_target = target.view(B, 1).expand(-1, N).reshape(B * N)
        flat_w = flat_mask.reshape(B * N).clamp_min(0.0)
        ce = F.cross_entropy(flat_logits, flat_target, reduction="none")
        loss = (ce * flat_w).sum() / flat_w.sum().clamp_min(1e-6)

        loss_comp = [loss.detach()]
        return loss, loss_comp
