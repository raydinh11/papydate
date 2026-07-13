import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def sinusoidal_cfg(n_bins, emb_dim, step=3, cst=10000):
    num_pos = n_bins * step
    position = torch.arange(num_pos).unsqueeze(1).float()
    div_term = torch.exp(torch.arange(0, emb_dim, 2).float() * (-math.log(cst) / emb_dim))
    pe = torch.zeros(num_pos, emb_dim)
    pe[:, 0::2] = torch.sin(position * div_term)
    pe[:, 1::2] = torch.cos(position * div_term)
    pe = pe[::step]
    return pe

class TimelineTokenizer(nn.Module):
    def __init__(self, emb_dim, n_bins, cst=10000, use_proj=False):
        super().__init__()
        basis = sinusoidal_cfg(n_bins, emb_dim, cst=cst)
        self.register_buffer("basis", basis)
        self.register_buffer("gaussian", torch.zeros(n_bins, n_bins))  # unused
        self.proj = nn.Linear(emb_dim, emb_dim) if use_proj else None

    def get_embeddings(self):
        b = self.proj(self.basis) if self.proj is not None else self.basis
        return F.normalize(b, dim=-1)
