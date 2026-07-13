from __future__ import annotations

import torch
import torch.nn as nn

from . import encoder
from losses import LOSSES

ENCODERS = {
    "encoder": encoder.Model,
}


class EncoderSequence(nn.Module):
    def __init__(self, cfg_enc_seq):
        super().__init__()
        self.net = ENCODERS[cfg_enc_seq["name"]](cfg_enc_seq)

    def forward(self, *args, **kwargs):
        return self.net(*args, **kwargs)


class Loss(nn.Module):
    def __init__(self, cfg_loss):
        super().__init__()
        self.net = LOSSES[cfg_loss["name"]](cfg_loss)

    def forward(self, *args, **kwargs):
        return self.net(*args, **kwargs)


class ModelBase(nn.Module):
    def __init__(self, cfg_model) -> None:
        super().__init__()
        self.encoder_sequence = EncoderSequence(cfg_model["encoder_sequence"])
        self.loss = Loss(cfg_model["loss"])

    def init_weights(self):
        for name, module in self.named_modules():
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    module.bias.data.fill_(0.01)

    def forward_train(self, batch_data, cfg_train, is_last=False):
        raise NotImplementedError

    def forward(self, batch_data, cfg_train, is_last=False):
        raise NotImplementedError
