from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F

class Model(nn.Module):
    def __init__(self, cfg_loss):
        super().__init__()
        self.cfg = cfg_loss
        temp = cfg_loss.get("temp", {})
        self.t_cross = float(temp.get("t_cross", 0.1))

    @staticmethod
    def _cossim(emb, timeline):
        emb_n = F.normalize(emb, dim=-1)
        tl_n = F.normalize(timeline, dim=-1)
        return emb_n @ tl_n.T

    def _LiT(self, cossim, bin_mid, pad_mask):
        B, N, _ = cossim.shape
        logits = cossim / self.t_cross
        log_denom = torch.logsumexp(logits, dim=-1)
        pos = logits.gather(-1, bin_mid[:, None, None].expand(B, N, 1)).squeeze(-1)
        per_tok = -(pos - log_denom)
        if pad_mask is not None:
            return (per_tok * pad_mask).sum() / pad_mask.sum().clamp_min(1e-6)
        return per_tok.mean()

    def _Lii(self, cossim, bin_per_tok, pad_mask, device):
        M = cossim.shape[0]
        logits = cossim / self.t_cross
        bin_diff = (bin_per_tok.unsqueeze(0) - bin_per_tok.unsqueeze(1)).abs()
        valid = ~torch.eye(M, dtype=torch.bool, device=device)
        if pad_mask is not None:
            tok_valid = pad_mask > 0
            valid = valid & tok_valid.unsqueeze(0) & tok_valid.unsqueeze(1)
        pos_mask = valid & (bin_diff <= self.cfg.get("contrast_bin_thresh", 10))
        neg_inf = torch.finfo(logits.dtype).min / 2
        logits_masked = logits.masked_fill(~valid, neg_inf)
        log_denom = torch.logsumexp(logits_masked, dim=-1)
        log_p = logits_masked - log_denom.unsqueeze(-1)
        n_pos = pos_mask.sum(dim=-1).clamp_min(1)
        per_tok = -(log_p * pos_mask.float()).sum(dim=-1) / n_pos
        has_pos = pos_mask.any(dim=-1)
        if not has_pos.any():
            return torch.zeros((), device=device, dtype=logits.dtype)
        if pad_mask is not None:
            w = pad_mask * has_pos.float()
            return (per_tok * w).sum() / w.sum().clamp_min(1e-6)
        return per_tok[has_pos].mean()

    def forward(self, tokens, mask, timeline, targets):
        B, ppim, npc, D = tokens.shape
        N = ppim * npc
        flat_tok = tokens.reshape(B, N, D)
        n_bins = timeline.shape[0]
        bin_mid = targets.clamp(0, n_bins - 1)
        pad_mask = mask.reshape(B, N) if mask is not None else None
        pad_mask_flat = pad_mask.reshape(B * N) if pad_mask is not None else None

        # cossim
        cossim_tl = self._cossim(flat_tok, timeline)               # token->timeline (B, N, n_bins)
        all_tok = F.normalize(flat_tok.reshape(B * N, D), dim=-1)
        cossim_tok = all_tok @ all_tok.T                           # token<->token (M, M)

        # loss
        L_iT = self._LiT(cossim_tl, bin_mid, pad_mask)
        L_ii = self._Lii(cossim_tok, bin_mid.repeat_interleave(N), pad_mask_flat, tokens.device)
        loss = (L_iT + L_ii) / 2.0

        loss_comp = [loss.detach()]
        return loss, loss_comp
