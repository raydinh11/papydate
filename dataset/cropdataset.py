from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.transforms import InterpolationMode
from PIL import Image

from .cropcache import CropCache
from .cropsampler import CropSampler


# CropDataset.__getitem__ returns (crops_ink, pos, bin, src_idx, valid).
# crops are float32 RGB, ImageNet-normalized; pos is normalized (cy, cx).
_IMAGENET_MEAN = [0.485, 0.456, 0.406]
_IMAGENET_STD = [0.229, 0.224, 0.225]
_NORM = transforms.Normalize(mean=_IMAGENET_MEAN, std=_IMAGENET_STD)


class CropDataset(Dataset):
    def __init__(
        self,
        root: str,
        split: str, # "train" | "val" | "test"
        train: bool,
        n_bins: int,
        min_date: int,
        max_date: int,
        imsz: int = 128,
        ppim: int = 1,
        ink_ratio_min: float = 0.0,
        cache_dir_name: str = "cache",
        transform=None,
        anchor_sigma_year: float = 10.0,
        anchor_year: bool = False,
        group_by_tm: bool = False,
        ink_quantile: bool = False,
    ):
        self.crop_sz = int(imsz)
        self.transform = transform or transforms.Compose([
            transforms.Resize([self.crop_sz, self.crop_sz], interpolation=InterpolationMode.BILINEAR),
            transforms.ToTensor(),
        ])
        self.cache = CropCache(root, split, cache_dir_name, group_by_tm)
        self.sampler = CropSampler(
            self.cache, train, ppim, n_bins, min_date, max_date,
            ink_ratio_min=ink_ratio_min, anchor_year=anchor_year,
            anchor_sigma_year=anchor_sigma_year, ink_quantile=ink_quantile,
        )

    def __len__(self):
        return len(self.sampler)

    def __getitem__(self, idx):
        return self._materialize(*self.sampler.plan(idx))

    def _load_one(self, pool_idx: int):
        img_pil = Image.fromarray(np.ascontiguousarray(self.cache.read_rgb(pool_idx)))
        return _NORM(self.transform(img_pil).float())

    def _materialize(self, picks, src_idx, bin_idx, n_real):
        crops_ink = torch.stack([self._load_one(pi) for pi in picks])  # (ppim, 3, H, W)
        pos = torch.from_numpy(self.cache.positions[picks]).float()    # (ppim, 2)
        valid = torch.zeros(len(picks), dtype=torch.float32)
        valid[:n_real] = 1.0
        return crops_ink, pos, int(bin_idx), int(src_idx), valid
