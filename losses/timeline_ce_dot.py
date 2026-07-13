from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F

class Model(nn.Module):
    def __init__(self, cfg_loss):
        super().__init__()
        self.cfg = cfg_loss
        temp = cfg_loss.get("temp", {})
        self.t_cross = float(temp.get("t_cross", 1.0))

    @staticmethod
    def _dot(emb, timeline):
        return emb @ timeline.T

    def forward(self, tokens, mask, timeline, targets):
        B, P, NPC, D = tokens.shape
        flat = tokens.reshape(B, P * NPC, D)
        flat_mask = mask.reshape(B, P * NPC)
        per_tok_logit = self._dot(flat, timeline)
        logits = per_tok_logit / self.t_cross

        n_bins = per_tok_logit.shape[-1]
        target = targets.long().clamp(0, n_bins - 1)
        N = per_tok_logit.shape[1]
        flat_logits = logits.reshape(B * N, n_bins)
        flat_target = target.view(B, 1).expand(-1, N).reshape(B * N)
        flat_w = flat_mask.reshape(B * N).clamp_min(0.0)
        ce = F.cross_entropy(flat_logits, flat_target, reduction='none')
        loss = (ce * flat_w).sum() / flat_w.sum().clamp_min(1e-6)

        loss_comp = [loss.detach()]
        return loss, loss_comp
