import os
import sys
import json
import argparse

import numpy as np
import cv2 as cv
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))   # review root (parent of tools/)
from config import load_cfg
from model import load_checkpoint

_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def _norm(arr_bgr):
    a = arr_bgr[:, :, :, ::-1].astype(np.float32) / 255.0
    return ((a - _MEAN) / _STD).transpose(0, 3, 1, 2)


def load_caches(root, cache_names, split):
    out = []
    for cn in cache_names:
        cdir = os.path.join(root, cn, split)
        j = json.load(open(os.path.join(cdir, "index.json")))
        entries = j["index"]
        meta = j["metadata"]
        sid = np.array([int(e["sample_idx"]) for e in entries], dtype=np.int64)
        side = np.array([float(e.get("side", 192)) for e in entries], dtype=np.float32)
        x = np.array([float(e.get("x", 0)) for e in entries], dtype=np.float32)
        y = np.array([float(e.get("y", 0)) for e in entries], dtype=np.float32)
        H = np.array([float((meta[s] or {}).get("img_h", 1)) or 1.0 for s in sid], dtype=np.float32)
        W = np.array([float((meta[s] or {}).get("img_w", 1)) or 1.0 for s in sid], dtype=np.float32)
        image_id = np.array([str((meta[s] or {}).get("image_id", "")) for s in sid])
        out.append(dict(
            chunks_dir=os.path.join(cdir, "chunks"), metadata=meta,
            chunk_file=np.array([e["chunk_file"] for e in entries]),
            offset=np.array([int(e["offset"]) for e in entries], dtype=np.int64),
            sid=sid, ink=np.array([float(e.get("ink_ratio", 0.0)) for e in entries], dtype=np.float32),
            tm=np.array([str(e.get("tm", "")) for e in entries]),
            x=x, y=y, side=side, img_h=H, img_w=W, image_id=image_id,
            cy_norm=(y + side / 2.0) / H, cx_norm=(x + side / 2.0) / W, n=len(entries),
        ))
    return out


# Per-crop prediction depends on the model's head:
#   timeline -> argmax/max of cossim(tokens, timeline)   (timeline_contrastive / timeline_ce_dot)
#   mse      -> regression head scalar -> year            (regression)
#   ce       -> classifier head argmax / softmax-max      (ce_year / ce_century)
def _mode_for_loss(loss_name):
    if loss_name in ("regression",):
        return "mse"
    if loss_name in ("ce_year", "ce_century"):
        return "ce"
    return "timeline"


def collect(model, caches, max_chunk, n_bins, min_d, bin_to_yr, device, imsz, mode="timeline"):
    tm_pool, tm_year, seen = {}, {}, {}
    for k, c in enumerate(caches):
        for i in range(c["n"]):
            tm_pool.setdefault(str(c["tm"][i]), []).append((k, i, float(c["ink"][i])))
        for sid, m in enumerate(c["metadata"]):
            if m is None:
                continue
            tm = str(m.get("tm", ""))
            if sid in seen.get(tm, set()):
                continue
            seen.setdefault(tm, set()).add(sid)
            yv = float(m.get("date", 0))
            if yv != 0:
                tm_year.setdefault(tm, []).append(yv)

    chunk_cache = {}
    out = {}
    for tm in sorted(tm_pool.keys()):
        ys = tm_year.get(tm, [])
        if not ys:
            continue
        true_yr = float(np.median(ys))
        items = tm_pool[tm]                      # all crops of this TM (no selection cap)
        if not items:
            continue
        crops, pos, cidx = [], [], []
        gx, gy, gside, giid, gh, gw = [], [], [], [], [], []
        for (k, i, _) in items:
            c = caches[k]
            cf = str(c["chunk_file"][i])
            key = (id(c), cf)
            arr = chunk_cache.get(key)
            if arr is None:
                arr = np.load(os.path.join(c["chunks_dir"], cf), mmap_mode="r")
                chunk_cache[key] = arr
            img = np.asarray(arr[int(c["offset"][i])])
            if img.shape[0] != imsz:
                img = cv.resize(img, (imsz, imsz), interpolation=cv.INTER_LINEAR)
            crops.append(img)
            pos.append([float(c["cy_norm"][i]), float(c["cx_norm"][i])])
            cidx.append(int(k))
            gx.append(float(c["x"][i])); gy.append(float(c["y"][i])); gside.append(float(c["side"][i]))
            giid.append(str(c["image_id"][i])); gh.append(float(c["img_h"][i])); gw.append(float(c["img_w"][i]))
        am_all, mx_all, yr_all = [], [], []
        for st in range(0, len(crops), max_chunk):
            sub = np.stack(crops[st:st + max_chunk], 0)
            xt = torch.from_numpy(_norm(sub)).to(device)
            pt = torch.tensor(pos[st:st + max_chunk], device=device, dtype=torch.float)
            model.encoder_sequence.net.ppim = xt.shape[0]
            with torch.no_grad():
                tokens, _, tl = model.encoder_sequence(xt, pt)
                B, P, NPC, D = tokens.shape
                flat = tokens.reshape(B, P * NPC, D)
                if mode == "mse":
                    # regression head -> target in [0,1] -> bin = norm*(n-1)
                    norm = model.loss.net.head(flat).squeeze(-1)[0].cpu().numpy()
                    yr_all.append(norm * (n_bins - 1) * bin_to_yr + min_d)
                elif mode == "ce":
                    logits = model.loss.net.head(flat)[0]
                    am_all.append(logits.argmax(-1).cpu().numpy())
                    mx_all.append(F.softmax(logits, -1).amax(-1).cpu().numpy())
                else:
                    per_tok = (F.normalize(flat, dim=-1) @ F.normalize(tl, dim=-1).T)[0]
                    am_all.append(per_tok.argmax(-1).cpu().numpy())
                    mx_all.append(per_tok.amax(-1).cpu().numpy())
        rec = {"true": int(round(true_yr)), "all_cache_idxs": list(map(int, cidx)),
               "all_x": [round(v, 1) for v in gx], "all_y": [round(v, 1) for v in gy],
               "all_side": [round(v, 1) for v in gside], "all_image_id": giid,
               "all_img_h": [round(v, 1) for v in gh], "all_img_w": [round(v, 1) for v in gw]}
        if mode == "mse":
            ys = np.concatenate(yr_all).astype(np.float32)
            rec["all_years"] = [round(float(y_), 2) for y_ in ys]
        else:
            rec["all_confs"] = np.concatenate(mx_all).astype(np.float32).tolist()
            rec["all_bins"] = np.concatenate(am_all).astype(np.int32).tolist()
        out[tm] = rec
    return out


def generate(ckpt_path, out_dir, splits=("val", "test"), root=None, caches=None,
             max_chunk=200, force_mode=None):
    ckpt_dir = os.path.dirname(ckpt_path)
    cfg = load_cfg(os.path.join(ckpt_dir, "cfg.yaml"))
    n_bins = int(cfg["train"]["n_bins"])
    min_d = int(cfg["train"]["min_date"])
    max_d = int(cfg["train"]["max_date"])
    bin_to_yr = (max_d - min_d) / n_bins
    device = cfg["train"]["device"]
    imsz = int(cfg["dataset"].get("im_sz", 128))
    cache_names = caches or cfg["dataset"]["cache_dir_name"]
    if isinstance(cache_names, str):
        cache_names = [cache_names]
    data_root = root or cfg["dataset"]["root"]
    cfg["model"]["global"]["ppim"] = 1
    cfg["model"]["encoder_sequence"]["ppim"] = 1
    model = load_checkpoint(cfg["model"], ckpt_path, device).eval()
    mode = force_mode or _mode_for_loss(cfg["model"]["loss"]["name"])
    os.makedirs(out_dir, exist_ok=True)
    print(f"  dump mode = {mode}  (loss={cfg['model']['loss']['name']})")
    for split in splits:
        cc = load_caches(data_root, cache_names, split)
        tms = collect(model, cc, max_chunk, n_bins, min_d, bin_to_yr, device, imsz, mode)
        p = os.path.join(out_dir, f"per_crop_{split}.json")
        json.dump({"tms": tms}, open(p, "w"))
        print(f"  wrote {p}  ({len(tms)} TMs)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--out", default=None,
                    help="output dir for per_crop_{split}.json (default: the checkpoint's own dir)")
    ap.add_argument("--splits", nargs="+", default=["val", "test"])
    ap.add_argument("--root", default=None)
    ap.add_argument("--caches", nargs="+", default=None)
    args = ap.parse_args()
    out = args.out or os.path.dirname(os.path.abspath(args.ckpt))
    generate(args.ckpt, out, args.splits, args.root, args.caches)


if __name__ == "__main__":
    main()
