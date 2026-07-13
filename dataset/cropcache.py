from __future__ import annotations

import os
import json
import functools

import numpy as np


# Cache layout (produced by preprocess/build_caches.py):
#   <root>/<cache_dir_name>/<split>/
#       index.json                     per-crop entries + per-doc metadata
#       chunks/<NNNNN>_<NNN>.npy        image chunk: (k, P, P, 3) uint8 BGR


@functools.lru_cache(maxsize=512)
def _open_npy(path: str):
    return np.load(path, mmap_mode="r")


class CropCache:
    """Storage: on-disk crop index, grouping maps, and pixel reads."""

    def __init__(self, root, split, cache_dir_name, group_by_tm=False):
        self.root = root
        self.split = split
        self.group_by_tm = bool(group_by_tm)
        self._load_caches(cache_dir_name)
        self._build_sibling_map()
        self._build_tm_maps()

    def __len__(self):
        return len(self.entries)

    def _load_caches(self, cache_dir_name):
        if isinstance(cache_dir_name, (list, tuple)):
            cache_names = list(cache_dir_name)
        else:
            cache_names = [cache_dir_name]
        self.cache_names = cache_names
        self.chunks_dirs: list[str] = []
        entries = []
        entry_dir_idx: list[int] = []
        entry_meta: list[list[dict | None]] = []   # per-cache metadata list
        precompute_szs: list[int] = []
        for k, cn in enumerate(cache_names):
            cdir = os.path.join(self.root, cn, self.split)
            ipath = os.path.join(cdir, "index.json")
            if not os.path.isfile(ipath):
                raise FileNotFoundError(f"Missing cache index: {ipath}")
            with open(ipath, "r") as f:
                d = json.load(f)
            self.chunks_dirs.append(os.path.join(cdir, "chunks"))
            entry_meta.append(d["metadata"])
            precompute_szs.append(int(d.get("patch_sz", 256)))
            for e in d["index"]:
                entries.append(e)
                entry_dir_idx.append(k)
        self.entries = entries
        self.entry_dir_idx = np.asarray(entry_dir_idx, dtype=np.int32)
        self.entry_meta = entry_meta
        # Cache 0 is canonical for metadata_list (src_idx lookups, external callers).
        self.metadata_list = entry_meta[0]
        self._precompute_szs = precompute_szs

        n = len(self.entries)
        self.chunk_files = [e["chunk_file"] for e in self.entries]
        self.offsets = np.asarray([e["offset"] for e in self.entries], dtype=np.int32)
        # sample_idxs is cache-local — only meaningful with entry_dir_idx.
        self.sample_idxs = np.asarray(
            [int(e["sample_idx"]) for e in self.entries], dtype=np.int32
        )
        self.ink_ratios = np.asarray(
            [float(e.get("ink_ratio", 0.0)) for e in self.entries], dtype=np.float32
        )
        self.positions = np.zeros((n, 2), dtype=np.float32)
        for i, e in enumerate(self.entries):
            k = int(self.entry_dir_idx[i])
            sid = int(e["sample_idx"])
            m = (self.entry_meta[k][sid] if sid < len(self.entry_meta[k]) else None) or {}
            H = float(m.get("img_h", 1)) or 1.0
            W = float(m.get("img_w", 1)) or 1.0
            ps = float(self._precompute_szs[k])
            self.positions[i, 0] = (float(e.get("y", 0)) + ps / 2) / H
            self.positions[i, 1] = (float(e.get("x", 0)) + ps / 2) / W

    def _build_sibling_map(self):
        # src_idx -> [pool_idx, ...], within cache 0 only (sample_idx is cache-local).
        self.siblings: dict[int, list[int]] = {}
        for pool_idx, (sid, k) in enumerate(zip(self.sample_idxs, self.entry_dir_idx)):
            if int(k) == 0:
                self.siblings.setdefault(int(sid), []).append(pool_idx)

    def _build_tm_maps(self):
        self.tm_per_entry: list[str] = [str(e.get("tm", "")) for e in self.entries]
        self.tm_to_pool: dict[str, list[int]] = {}
        for pool_idx, tm in enumerate(self.tm_per_entry):
            self.tm_to_pool.setdefault(tm, []).append(pool_idx)
        # tm_to_docs: TM -> [src_idx, ...] in cache 0; de-dup across caches.
        self.tm_to_docs: dict[str, list[int]] = {}
        seen_per_tm: dict[str, set] = {}
        for k, mlist in enumerate(self.entry_meta):
            for src, m in enumerate(mlist):
                if m is None:
                    continue
                tm = str(m.get("tm", ""))
                key = (k, int(src))
                if tm in seen_per_tm and key in seen_per_tm[tm]:
                    continue
                seen_per_tm.setdefault(tm, set()).add(key)
                if k == 0:
                    self.tm_to_docs.setdefault(tm, []).append(int(src))
        # Synthetic negative id for any TM absent from cache 0.
        synth = -1
        for tm in self.tm_to_pool.keys():
            if tm not in self.tm_to_docs:
                self.tm_to_docs[tm] = [synth]
                synth -= 1
        # src_idx -> tm built from cache 0 only.
        self.src_to_tm: dict[int, str] = {}
        for src, m in enumerate(self.metadata_list):
            if m is None:
                continue
            self.src_to_tm[int(src)] = str(m.get("tm", ""))
        if self.group_by_tm and not any(self.tm_per_entry):
            raise RuntimeError("group_by_tm=True but cache entries carry no `tm` field")

    def read_rgb(self, pool_idx: int) -> np.ndarray:
        chunk = self.chunk_files[pool_idx]
        offset = int(self.offsets[pool_idx])
        k = int(self.entry_dir_idx[pool_idx])
        img_arr = _open_npy(os.path.join(self.chunks_dirs[k], chunk))
        # cache stores BGR uint8 -> convert to RGB for ImageNet-pretrained net.
        return np.asarray(img_arr[offset])[:, :, ::-1]   # BGR -> RGB
