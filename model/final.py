from __future__ import annotations

import torch

from .base import ModelBase

class Model(ModelBase):
    def forward_train(self, batch_data, cfg_train, is_last=False):
        device = cfg_train["device"]
        crops_ink, pos, bin_idx, _, valid_mask = batch_data

        crops_ink = crops_ink.to(device, non_blocking=True)
        pos = pos.to(device, non_blocking=True).float()
        bin_idx = bin_idx.to(device).long() if isinstance(bin_idx, torch.Tensor) else torch.tensor(bin_idx, device=device).long()

        if crops_ink.dim() == 4:
            crops_ink = crops_ink.unsqueeze(1)
        if pos.dim() == 2:
            pos = pos.unsqueeze(1)
        crops_ink = crops_ink.flatten(0, 1)
        pos = pos.flatten(0, 1)

        tokens, selector_mask, timeline_emb = self.encoder_sequence(crops_ink, pos)

        if valid_mask is not None:
            valid_mask = valid_mask.to(device, non_blocking=True).float()
            B, P, _ = selector_mask.shape
            selector_mask = selector_mask * valid_mask.view(B, P, 1)

        loss, loss_comp = self.loss(tokens, selector_mask, timeline_emb, bin_idx)

        return loss, loss_comp