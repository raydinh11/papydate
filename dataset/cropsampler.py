from __future__ import annotations

import os
import random

import numpy as np

from .cropcache import CropCache
from .utils import _parse_year


class CropSampler:
    """Planning: bag plans (picks, src_idx, bin_idx, n_real) over a CropCache."""

    def __init__(self, cache: CropCache, train, ppim, n_bins, min_date, max_date,
                 ink_ratio_min=0.0, anchor_year=False, anchor_sigma_year=10.0,
                 ink_quantile=False):
        self.cache = cache
        self.train = bool(train)
        self.ppim = int(ppim)
        self.n_bins = int(n_bins)
        self.min_date = int(min_date)
        self.max_date = int(max_date)
        self._date_range = self.max_date - self.min_date
        if self._date_range <= 0:
            raise ValueError(f"max_date ({max_date}) must be > min_date ({min_date})")
        self.ink_ratio_min = float(ink_ratio_min)

        self.ink_quantile = bool(ink_quantile)
        self.group_by_tm = cache.group_by_tm
        self._build_anchor_year_table(anchor_year, anchor_sigma_year)
        self._build_anchor_pool()
        if not self.train:
            self._build_val_pools()

    def __len__(self):
        if self.train:
            return len(self.anchors)
        return len(self._val_tms) if self.group_by_tm else len(self._val_docs)

    def plan(self, idx):
        return self._get_train(idx) if self.train else self._get_val(idx)

    # Pools
    def _build_anchor_year_table(self, anchor_year, anchor_sigma_year):
        self.anchor_year = bool(anchor_year) and self.train
        self.anchor_sigma_year = float(anchor_sigma_year)
        self._cm_doc_ids: list[int] = []
        self._cm_doc_years: np.ndarray = np.zeros(0, dtype=np.float32)
        if self.anchor_year:
            self._cm_doc_ids = sorted(self.cache.siblings.keys())
            ys = []
            for d in self._cm_doc_ids:
                m = self.cache.metadata_list[int(d)] or {}
                ds_y = _parse_year(m.get("date_start", m.get("date", 0)), m.get("date", 0))
                de_y = _parse_year(m.get("date_end", m.get("date", 0)), m.get("date", 0))
                ys.append((ds_y + de_y) / 2.0)
            self._cm_doc_years = np.asarray(ys, dtype=np.float32)

    def _build_anchor_pool(self):
        anchor_mask = self.cache.ink_ratios >= self.ink_ratio_min
        self.anchors = np.where(anchor_mask)[0].astype(np.int64)
        if len(self.anchors) == 0:
            raise RuntimeError(
                f"No crops with ink_ratio >= {self.ink_ratio_min} in "
                f"{[os.path.join(self.cache.root, cn, self.cache.split) for cn in self.cache.cache_names]}"
            )

    def _build_val_pools(self):
        anchor_set = set(int(p) for p in self.anchors)
        if self.group_by_tm:
            
            tm_to_anchors: dict[str, list[int]] = {}
            for tm, pool in self.cache.tm_to_pool.items():
                a = [pi for pi in pool if pi in anchor_set]
                if a:
                    a.sort(key=lambda pi: -float(self.cache.ink_ratios[pi]))
                    tm_to_anchors[tm] = a
            self._val_tms = sorted(tm_to_anchors.keys())
            self._val_tm_picks = tm_to_anchors
            
            self._val_tm_repr: dict[str, int] = {}
            self._val_tm_year: dict[str, float] = {}
            tm_years: dict[str, list[float]] = {}
            tm_repr: dict[str, int] = {}
            for k, mlist in enumerate(self.cache.entry_meta):
                for src, m in enumerate(mlist):
                    if m is None:
                        continue
                    tm = str(m.get("tm", ""))
                    if tm not in tm_to_anchors:
                        continue
                    tm_years.setdefault(tm, []).append(float(m.get("date", 0)))
                    if k == 0 and tm not in tm_repr:
                        tm_repr[tm] = int(src)
            for tm in tm_to_anchors.keys():
                self._val_tm_repr[tm] = tm_repr.get(tm, 0)
                ys = tm_years.get(tm, [])
                self._val_tm_year[tm] = float(np.median(ys)) if ys else 0.0
        else:
            doc_to_anchors: dict[int, list[int]] = {}
            for src, sibs in self.cache.siblings.items():
                a = [pi for pi in sibs if pi in anchor_set]
                if a:
                    a.sort(key=lambda pi: -float(self.cache.ink_ratios[pi]))
                    doc_to_anchors[src] = a
            self._val_docs = sorted(doc_to_anchors.keys())
            self._val_doc_picks = doc_to_anchors

    # Indexing
    def _year_to_bin(self, year) -> int:
        b = int((float(year) - self.min_date) / self._date_range * self.n_bins)
        if b < 0:
            return 0
        if b >= self.n_bins:
            return self.n_bins - 1
        return b

    def _bin_from_meta(self, meta) -> int:
        return self._year_to_bin((meta or {}).get("date", 0))

    def _pad_picks(self, pool):
        # First ppim picks, padded with pool[0] if short. -> (picks, n_real).
        if len(pool) >= self.ppim:
            return pool[: self.ppim], self.ppim
        return pool + [pool[0]] * (self.ppim - len(pool)), len(pool)

    def _get_train(self, idx):
        if self.anchor_year:
            return self._get_train_anchor_year(idx)
        anchor_pool_idx = int(self.anchors[idx])
        k = int(self.cache.entry_dir_idx[anchor_pool_idx])
        sid = int(self.cache.sample_idxs[anchor_pool_idx])
        meta = self.cache.entry_meta[k][sid] if sid < len(self.cache.entry_meta[k]) else None
        if k == 0:
            src_idx = sid                                      # cache 0: sample_idx is the doc id
        else:
            tm = self.cache.tm_per_entry[anchor_pool_idx]      # other caches: resolve via TM
            src_idx = int(self.cache.tm_to_docs[tm][0])
        return self._plan_anchor(anchor_pool_idx, src_idx, self._bin_from_meta(meta))

    def _get_train_anchor_year(self, idx):
        n = len(self._cm_doc_ids)
        t_lo = float(self._cm_doc_years.min())
        t_hi = float(self._cm_doc_years.max())
        t = random.uniform(t_lo, t_hi)
        diff = self._cm_doc_years - t
        log_w = -(diff * diff) / (2.0 * self.anchor_sigma_year ** 2)
        log_w -= log_w.max()
        w = np.exp(log_w)
        s = w.sum()
        if not np.isfinite(s) or s <= 0:
            i = int(np.random.randint(0, n))
        else:
            i = int(np.random.choice(n, p=(w / s)))
        src_idx = int(self._cm_doc_ids[i])
        anchor = int(self.cache.siblings[src_idx][0])
        bin_idx = self._bin_from_meta(self.cache.metadata_list[src_idx])
        return self._plan_anchor(anchor, src_idx, bin_idx)

    def _sample_siblings(self, anchor_pool_idx: int, src_idx: int):
        # -> (picks, n_real). Sample siblings of the anchor's doc/TM; pad if few.
        if self.group_by_tm:
            tm = self.cache.src_to_tm.get(int(src_idx), "")
            sibs = self.cache.tm_to_pool.get(tm, self.cache.siblings[src_idx])
        else:
            sibs = self.cache.siblings[src_idx]
        if self.ink_quantile:
            return self._sample_ink_quantile(sibs)
        others = [i for i in sibs if i != anchor_pool_idx]
        need = self.ppim - 1
        if not others:
            return [anchor_pool_idx] * self.ppim, 1
        if need <= len(others):
            picks = random.sample(others, need)
            return [anchor_pool_idx] + picks, self.ppim
        picks = random.choices(others, k=need)
        return [anchor_pool_idx] + picks, 1 + len(others)

    def _sample_ink_quantile(self, sibs):
        pool = list(sibs)
        if len(pool) <= self.ppim:
            n_real = len(pool)
            picks = pool + [pool[0]] * (self.ppim - len(pool)) if pool else []
            return picks, max(1, n_real)
        order = sorted(pool, key=lambda pi: float(self.cache.ink_ratios[pi]))
        n = len(order)
        picks = []
        for q in range(self.ppim):
            lo, hi = q * n // self.ppim, (q + 1) * n // self.ppim
            grp = order[lo:hi] if hi > lo else order[lo:lo + 1]
            picks.append(random.choice(grp))
        random.shuffle(picks)
        return picks, self.ppim

    def _plan_anchor(self, anchor: int, src_idx: int, bin_idx: int):
        if self.ppim <= 1:
            picks, n_real = [anchor], 1
        else:
            picks, n_real = self._sample_siblings(anchor, src_idx)
        return picks, src_idx, bin_idx, n_real

    def _get_val(self, idx):
        if self.group_by_tm:
            tm = self._val_tms[idx]
            picks, n_real = self._pad_picks(self._val_tm_picks[tm])
            src_idx = self._val_tm_repr[tm]
            bin_idx = self._year_to_bin(self._val_tm_year[tm])
            return picks, src_idx, bin_idx, n_real
        src_idx = self._val_docs[idx]
        picks, n_real = self._pad_picks(self._val_doc_picks[src_idx])
        bin_idx = self._bin_from_meta(self.cache.metadata_list[src_idx])
        return picks, src_idx, bin_idx, n_real
