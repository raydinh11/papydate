import os
import sys
import yaml
import random
import logging
import argparse
from datetime import datetime

import numpy as np
import torch
from torch.utils.data import DataLoader, WeightedRandomSampler
from torchvision import transforms

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import load_cfg
from model import build_model
from trainer import training_loop
from dataset import (
    CropDataset, ElasticDeformation, RandomAffine, _parse_year,
)

logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
log = logging.getLogger("train")


def _ds_kwargs(cfg):
    d = cfg["dataset"]
    t = cfg["train"]
    return dict(
        n_bins=int(t["n_bins"]), min_date=int(t["min_date"]), max_date=int(t["max_date"]),
        imsz=d["im_sz"], ppim=d["ppim"],
        ink_ratio_min=d.get("ink_ratio_min", 0.05),
        cache_dir_name=d["cache_dir_name"],
        anchor_sigma_year=float(d.get("anchor_sigma_year", 10.0)),
        anchor_year=bool(d.get("anchor_year", False)),
        group_by_tm=bool(d["group_by_tm"]),
        ink_quantile=bool(d.get("ink_quantile", False)),
    )


def build_datasets(cfg):
    d = cfg["dataset"]
    cs = int(d["im_sz"])
    bilinear = transforms.InterpolationMode.BILINEAR
    # Full per-crop pipeline (PIL -> tensor), one per split. train: geometric
    # (random-resized crop + elastic/affine) then photometric. val/test: resize.
    train_tf = transforms.Compose([
        transforms.RandomResizedCrop(cs, scale=(0.35, 1.0), ratio=(1.0, 1.0), interpolation=bilinear),
        ElasticDeformation(alpha=40, sigma=6, p=0.3),
        RandomAffine(p=0.3, rotation=8, scale=(0.95, 1.05)),
        transforms.RandomApply([transforms.ColorJitter(brightness=0.2, contrast=0.2)], p=0.3),
        transforms.RandomApply([transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 1.0))], p=0.5),
        transforms.RandomGrayscale(p=0.3),
        transforms.ToTensor(),
    ])
    eval_tf = transforms.Compose([
        transforms.Resize([cs, cs], interpolation=bilinear),
        transforms.ToTensor(),
    ])
    kw = _ds_kwargs(cfg)
    root = d["root"]
    train_set = CropDataset(root, split="train", train=True, transform=train_tf, **kw)
    valid_set = CropDataset(root, split="val", train=False, transform=eval_tf, **kw)
    test_set = CropDataset(root, split="test", train=False, transform=eval_tf, **kw)
    return train_set, valid_set, test_set


def build_sampler(train_set, cfg):
    mode = cfg["train"].get("sampler_mode", "uniform")
    if mode == "uniform":
        return None, True
    anchor_dates = []
    for anchor_pos in range(len(train_set.sampler.anchors)):
        pool_idx = int(train_set.sampler.anchors[anchor_pos])
        src = int(train_set.cache.sample_idxs[pool_idx])
        m = train_set.cache.metadata_list[src]
        ds_y = _parse_year(m.get("date_start", m["date"]), m["date"])
        de_y = _parse_year(m.get("date_end", m["date"]), m["date"])
        anchor_dates.append((ds_y + de_y) / 2.0)
    bins = np.floor(np.asarray(anchor_dates) / 100.0).astype(np.int64)
    bin_idx = bins - int(bins.min())
    counts = np.bincount(bin_idx)[bin_idx]
    if mode == "inverse":
        weights = 1.0 / counts.astype(np.float64)
    elif mode == "sqrt_inverse":
        weights = 1.0 / np.sqrt(counts.astype(np.float64))
    else:
        raise ValueError(f"unknown sampler_mode={mode}")
    weights = weights / weights.sum() * len(weights)
    sampler = WeightedRandomSampler(weights.tolist(), num_samples=len(train_set), replacement=True)
    return sampler, False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cfg", required=True)
    ap.add_argument("--ckpt-root", default="runs")
    args = ap.parse_args()

    cfg = load_cfg(args.cfg)
    SEED = int(cfg["train"].get("seed", 0))
    random.seed(SEED); np.random.seed(SEED)
    torch.manual_seed(SEED); torch.cuda.manual_seed_all(SEED)
    _g = torch.Generator(); _g.manual_seed(SEED)

    def _winit(wid):
        s = SEED + wid
        random.seed(s); np.random.seed(s); torch.manual_seed(s)

    date = datetime.now().strftime("%Y%m%d_%H%M%S")
    cfg["train"]["ckpt"] = os.path.join(args.ckpt_root, date)
    os.makedirs(cfg["train"]["ckpt"], exist_ok=True)
    src_cfg = yaml.safe_load(open(args.cfg))
    src_cfg["train"]["seed"] = SEED
    with open(os.path.join(cfg["train"]["ckpt"], "cfg.yaml"), "w") as f:
        yaml.dump(src_cfg, f, sort_keys=False, default_flow_style=False)
    log.info(f"[EXP_ID] {cfg['train']['ckpt']}")

    train_set, valid_set, test_set = build_datasets(cfg)
    log.info(f"TRAIN: {len(train_set)} VALID: {len(valid_set)} TEST: {len(test_set)}")
    sampler, shuffle = build_sampler(train_set, cfg)

    train_loader = DataLoader(train_set, shuffle=shuffle, sampler=sampler,
                              drop_last=True, batch_size=cfg["train"]["batch_size"],
                              num_workers=cfg["train"]["workers"], pin_memory=True,
                              persistent_workers=cfg["train"]["workers"] > 0,
                              worker_init_fn=_winit, generator=_g)
    valid_loader = DataLoader(valid_set, shuffle=False, batch_size=cfg["train"]["batch_size"], num_workers=4)

    model = build_model(cfg["model"]).to(cfg["train"]["device"])
    training_loop(model, train_loader, valid_loader, cfg["train"])


if __name__ == "__main__":
    main()
