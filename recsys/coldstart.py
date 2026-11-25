"""New users with no history: recommend what's popular with people like them.

MovieLens 1M records each user's gender and age group. Popularity within that
segment, recently, blended with overall popularity, beats the same list for
everyone. It also gives the reranker a "people like you" feature for warm users.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .data import AGE_GROUPS
from .retrieval import DAY, Context

AGE_CODES = sorted(AGE_GROUPS)


def segment_ids(users: pd.DataFrame) -> np.ndarray:
    """One id per (gender, age group): 14 segments. `users` is in catalog order."""
    age_idx = np.searchsorted(AGE_CODES, users["age"].to_numpy())
    return (users["gender"].to_numpy() == "F").astype(int) * len(AGE_CODES) + age_idx


def segment_key(gender: str, age: int) -> str:
    return f"{gender}|{age}"


class SegmentPopular:
    """Share of each movie among recent ratings by the user's segment, plus a bit of overall share."""

    name = "segment"

    def __init__(self, days: int = 180, blend: float = 0.3):
        self.days, self.blend = days, blend

    def fit(self, ctx: Context, users: pd.DataFrame) -> "SegmentPopular":
        self.seg = segment_ids(users)
        n_seg = 2 * len(AGE_CODES)
        recent = ctx.history[ctx.history["ts"] >= ctx.cutoff - self.days * DAY]
        if len(recent) < 1000:  # too little recent data: use everything
            recent = ctx.history
        u = ctx.catalog.user_pos(recent["user"])
        i = ctx.catalog.item_pos(recent["item"])
        counts = np.zeros((n_seg, ctx.catalog.n_items), dtype=np.float64)
        np.add.at(counts, (self.seg[u], i), 1.0)
        self.sizes = counts.sum(axis=1)  # ratings per segment
        overall = counts.sum(axis=0)
        overall_share = overall / max(overall.sum(), 1.0)
        seg_share = counts / np.maximum(counts.sum(axis=1, keepdims=True), 1.0)
        self.share = (seg_share + self.blend * overall_share[None, :]).astype(np.float32)
        self.X = ctx.X
        return self

    def scores(self, rows: np.ndarray) -> np.ndarray:
        return self.share[self.seg[rows]]
