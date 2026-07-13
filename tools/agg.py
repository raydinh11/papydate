#!/usr/bin/env python3
from collections import defaultdict
import numpy as np

DOWN = 10
TOPQ = 0.1
PIXEL = "median"
BUCKS = list(range(-300, 800, 100))
BW = 50

def majority_vote(yrs):
    arr = np.sort(np.asarray(yrs, dtype=np.float64))
    if arr.size == 0:
        return float("nan")
    half = BW / 2.0
    centers = np.arange(int(np.floor(arr[0])), int(np.ceil(arr[-1])) + 1)
    n = arr.size; best_c = centers[0]; best = -1; lo_i = hi_i = 0
    for c in centers:
        while lo_i < n and arr[lo_i] < c - half:
            lo_i += 1
        while hi_i < n and arr[hi_i] < c + half:
            hi_i += 1
        if hi_i - lo_i > best:
            best = hi_i - lo_i; best_c = c
    return float(best_c)


def heatmap_pixel_means(crops):
    if not crops:
        return np.array([], dtype=np.float64)
    H = int(crops[0]["img_h"]); W = int(crops[0]["img_w"])
    Hh = max(1, H // DOWN); Ww = max(1, W // DOWN)
    cidx = []; pv = []
    for c in crops:
        x0 = max(0, int(round(c["x"] / DOWN))); y0 = max(0, int(round(c["y"] / DOWN)))
        x1 = min(Ww, int(round((c["x"] + c["side"]) / DOWN)))
        y1 = min(Hh, int(round((c["y"] + c["side"]) / DOWN)))
        if x1 <= x0 or y1 <= y0:
            continue
        ys = np.arange(y0, y1); xs = np.arange(x0, x1)
        cid = (ys[:, None] * Ww + xs[None, :]).ravel()
        cidx.append(cid); pv.append(np.full(cid.size, float(c["pred_year"])))
    if not cidx:
        return np.array([], dtype=np.float64)
    cidx = np.concatenate(cidx); pv = np.concatenate(pv)
    o = np.lexsort((pv, cidx)); cidx = cidx[o]; pv = pv[o]
    _, st = np.unique(cidx, return_index=True)
    en = np.append(st[1:], len(cidx)); cnt = en - st
    lo = st + (cnt - 1) // 2; hi = st + cnt // 2
    return (pv[lo] + pv[hi]) / 2.0


def select(cr, ks, topq):
    sub = [c for c in cr if ks is None or c[1] in ks]
    if not sub:
        return sub
    cf = np.array([c[2] for c in sub], float)
    if not np.isfinite(cf).any():
        return sub
    thr = np.nanquantile(cf, 1 - topq)
    return [c for c in sub if c[2] >= thr]


def crop_pred(cr):
    return np.array([c[0] for c in cr])


def heat_pool(cr):
    bi = defaultdict(list)
    for p, k, cf, x, y, s, iid, h, w in cr:
        if iid:
            bi[iid].append({"x": x, "y": y, "side": s, "img_h": h, "img_w": w, "pred_year": p})
    return np.concatenate([heatmap_pixel_means(c) for c in bi.values()]) if bi else np.array([])
