from __future__ import annotations
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as tvm

from .timeline import TimelineTokenizer


class PerceiverBlock(nn.Module):
    def __init__(self, d_model: int = 64, n_self_attn: int = 2, dropout: float = 0.1):
        super().__init__()
        self.drop1 = nn.Dropout(dropout)
        self.drop2 = nn.Dropout(dropout)
        self.xnorm = nn.LayerNorm(d_model)
        self.cross_attn     = nn.MultiheadAttention(d_model, 1, batch_first=True, dropout=dropout)
        self.cross_norm     = nn.LayerNorm(d_model)
        self.cross_ffn      = nn.Sequential(
            nn.Linear(d_model, d_model), nn.Dropout(dropout), nn.GELU(), nn.Linear(d_model, d_model)
        )
        self.cross_ffn_norm = nn.LayerNorm(d_model)

        self.self_layers = nn.ModuleList([
            nn.MultiheadAttention(d_model, 2, batch_first=True, dropout=dropout)
            for _ in range(n_self_attn)
        ])
        self.self_norms = nn.ModuleList([nn.LayerNorm(d_model) for _ in range(n_self_attn)])
        self.ffns = nn.ModuleList([
            nn.Sequential(
                nn.Linear(d_model, d_model), nn.Dropout(dropout), nn.GELU(), nn.Linear(d_model, d_model)
            ) for _ in range(n_self_attn)
        ])
        self.ffn_norms = nn.ModuleList([nn.LayerNorm(d_model) for _ in range(n_self_attn)])

    def forward(self, latents, x, attn_mask=None):
        x_norm = self.xnorm(x)
        l_norm = self.cross_norm(latents)
        z, _ = self.cross_attn(l_norm, x_norm, x_norm)
        latents = latents + self.drop1(z)
        latents = latents + self.cross_ffn(self.cross_ffn_norm(latents))

        for attn, norm, ffn, ffn_norm in zip(self.self_layers, self.self_norms,
                                              self.ffns, self.ffn_norms):
            l_norm = norm(latents)
            z, _ = attn(l_norm, l_norm, l_norm, attn_mask=attn_mask)
            latents = latents + self.drop2(z)
            latents = latents + ffn(ffn_norm(latents))
        return latents


class ResNetBackbone(nn.Module):
    def __init__(self, arch: str = "resnet18", pretrained: bool = True,
                 freeze_bn: bool = True, pool: bool = True,
                 depth_stages: int = 4):
        super().__init__()
        assert depth_stages in (1, 2, 3, 4)
        self.pool = pool
        self.depth_stages = depth_stages

        rn = self._build(arch, pretrained)
        self.stem = nn.Sequential(rn.conv1, rn.bn1, rn.relu, rn.maxpool)
        self.layers = nn.ModuleList([rn.layer1, rn.layer2, rn.layer3, rn.layer4][:depth_stages])
        self.out_channels = self._stage_channels(arch)[depth_stages - 1]

        if freeze_bn:
            for sub in self.modules():
                if isinstance(sub, nn.BatchNorm2d):
                    sub.eval()
                    for p in sub.parameters():
                        p.requires_grad_(False)

    @staticmethod
    def _build(arch: str, pretrained: bool):
        if arch == "resnet18":
            w = tvm.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
            return tvm.resnet18(weights=w)
        if arch == "resnet34":
            w = tvm.ResNet34_Weights.IMAGENET1K_V1 if pretrained else None
            return tvm.resnet34(weights=w)
        if arch == "resnet50":
            w = tvm.ResNet50_Weights.IMAGENET1K_V1 if pretrained else None
            return tvm.resnet50(weights=w)
        raise ValueError(f"unknown arch: {arch}")

    @staticmethod
    def _stage_channels(arch: str):
        return {
            "resnet18": [64, 128, 256, 512],
            "resnet34": [64, 128, 256, 512],
            "resnet50": [256, 512, 1024, 2048],
        }[arch]

    def forward(self, x):
        x = self.stem(x)
        for layer in self.layers:
            x = layer(x)
        if self.pool:
            x = F.adaptive_avg_pool2d(x, 1).flatten(2).permute(0, 2, 1)
        return x

class InkEncoder(nn.Module):
    """ResNet backbone + projection producing one 'ink' token per crop."""

    def __init__(self, emb_dim: int, arch: str = "resnet18",
                 pretrained: bool = True, freeze_bn: bool = True,
                 depth_stages: int = 4):
        super().__init__()
        self.backbone = ResNetBackbone(
            arch=arch, pretrained=pretrained, freeze_bn=freeze_bn,
            pool=True, depth_stages=depth_stages,
        )
        hidden = max(emb_dim * 2, self.backbone.out_channels // 2)
        self.proj = nn.Sequential(
            nn.LayerNorm(self.backbone.out_channels),
            nn.Linear(self.backbone.out_channels, hidden),
            nn.GELU(),
            nn.Linear(hidden, emb_dim),
            nn.LayerNorm(emb_dim),
        )

    def forward(self, img: torch.Tensor) -> torch.Tensor:
        return self.proj(self.backbone(img))   # (B, 1, emb_dim)


class PerceiverGuidedSelector(nn.Module):
    def __init__(self, d_model: int):
        super().__init__()
        self.norm_x = nn.LayerNorm(d_model)
        self.norm_s = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor, latents: torch.Tensor):
        s = F.normalize(latents.mean(dim=1), dim=-1)
        x_n = F.normalize(x, dim=-1)
        logits = torch.einsum("bnd,bd->bn", x_n, s) / 0.6
        mask = torch.softmax(logits, dim=-1)
        m = mask.unsqueeze(-1)
        doc_emb = (x * m).sum(dim=1)
        return doc_emb, mask


def _pos_sincos_2d(pos: torch.Tensor, dim: int) -> torch.Tensor:
    d_each = dim // 2
    omega = torch.arange(d_each // 2, device=pos.device, dtype=pos.dtype)
    omega = 1.0 / (10000 ** (omega / max(d_each // 2 - 1, 1)))
    y, x = pos[:, 0:1], pos[:, 1:2]
    emb = torch.cat([
        torch.sin(y * omega), torch.cos(y * omega),
        torch.sin(x * omega), torch.cos(x * omega),
    ], dim=-1)
    if emb.shape[-1] < dim:
        emb = F.pad(emb, (0, dim - emb.shape[-1]))
    return emb


class Model(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        emb_dim = int(cfg["emb_dim"])
        n_bins = int(cfg["n_bins"])
        cfg["img_feat_dim"] = emb_dim

        branches = cfg.get("branches", ["ink"])
        if list(branches) != ["ink"]:
            raise ValueError(f"only the 'ink' branch is supported, got {branches!r}")
        self.branches = ["ink"]

        self.shape_bg = InkEncoder(
            emb_dim=emb_dim,
            arch=cfg.get("rn_arch", "resnet18"),
            pretrained=bool(cfg.get("rn_pretrained", True)),
            freeze_bn=bool(cfg.get("freeze_bn", True)),
            depth_stages=int(cfg.get("rn_depth_stages", 4)),
        )

        n_latents = cfg.get("n_latents", 16)
        assert n_latents % 2 == 0
        self.n_half = n_latents // 2
        self.latent_queries = nn.Parameter(torch.randn(1, n_latents, emb_dim) * 0.02)
        n_perceiver_blocks = cfg.get("n_perceiver_blocks", 2)
        self.perceivers = nn.ModuleList([
            PerceiverBlock(d_model=emb_dim, n_self_attn=2, dropout=cfg["drop_prob"])
            for _ in range(n_perceiver_blocks)
        ])
        self.final_norm = nn.LayerNorm(emb_dim)
        self.token_keep_ratio = float(cfg.get("token_keep_ratio", 0.7))
        self.use_batch_cossim = bool(cfg.get("use_batch_cossim", True))

        self.ppim = int(cfg["ppim"])
        self.n_toks_per_crop = len(self.branches)

        self.use_selector = bool(cfg.get("use_selector", True))
        self.selector = (
            PerceiverGuidedSelector(d_model=emb_dim)
            if self.use_selector else None
        )

        self.timeline = TimelineTokenizer(emb_dim, n_bins,
                                          use_proj=bool(cfg.get("timeline_proj", False)))

    def _build_tokens(self, img, pos):
        x = self.shape_bg(img)              # (B, 1, D)
        D = x.shape[-1]
        if pos is not None:
            x = x + _pos_sincos_2d(pos, D).unsqueeze(1)
        bsz = x.shape[0] // self.ppim
        return x.reshape(bsz, -1, D)

    def _run_perceiver(self, x):
        latents = self.latent_queries.expand(x.shape[0], -1, -1)
        for block in self.perceivers:
            latents = block(latents, x)
        return latents

    def _gate(self, x, latents):
        if self.selector is None:
            return x.new_ones(x.shape[:2])
        return self.selector(x, latents)[1]

    def forward(self, img, pos=None):
        x = self._build_tokens(img, pos)

        if self.training and x.shape[1] > 1 and self.token_keep_ratio < 1.0:
            keep = torch.bernoulli(torch.full(
                (x.shape[0], x.shape[1], 1), self.token_keep_ratio,
                device=x.device, dtype=x.dtype))
            x = x * keep + torch.randn_like(x) * 0.01

        if self.selector is None:
            m = x.new_ones(x.shape[:2])
        else:
            latents = self._run_perceiver(x)
            m = self._gate(x, latents)

        bsz, _, D = x.shape
        npc = self.n_toks_per_crop
        return (
            x.reshape(bsz, self.ppim, npc, D),
            m.reshape(bsz, self.ppim, npc),
            self.timeline.get_embeddings(),
        )
