"""Features for the reranker, computed only from history before the cutoff.

For each (user, candidate movie): how strongly each retrieval model proposed it,
how popular and well rated the movie is (overall and lately), how active the user
is, how well the movie's genres match the user's taste, and how popular it is with
people of the same gender and age group.
"""

from __future__ import annotations

import datetime as dt

import numpy as np
import pandas as pd

from .coldstart import SegmentPopular
from .data import genre_matrix
from .retrieval import DAY, Context

FEATURE_LABELS = {
    "itemcf_score": "Item-to-item score",
    "itemcf_rank": "Rank in item-to-item",
    "svd_score": "SVD score",
    "svd_rank": "Rank in SVD",
    "ease_score": "EASE score",
    "ease_rank": "Rank in EASE",
    "n_sources": "Number of models proposing it",
    "item_log_pop": "Movie: times rated",
    "item_log_recent": "Movie: times rated in the last 30 days",
    "item_trend": "Movie: rated more or less than usual lately",
    "item_avg_rating": "Movie: average rating",
    "item_age": "Movie: years since release",
    "user_log_hist": "User: movies rated so far",
    "user_avg_rating": "User: average rating given",
    "user_mainstream": "User: how mainstream their movies are",
    "user_days_since_last": "User: days since last rating",
    "user_days_active": "User: days since first rating",
    "gender": "User: gender",
    "age": "User: age group",
    "occupation": "User: occupation",
    "genre_affinity": "Genre match with the user's history",
    "segment_share": "Popular with people of the same age and gender",
}
FEATURES = list(FEATURE_LABELS)
USER_LEVEL = [f for f in FEATURES if f.startswith("user_")] + ["gender", "age", "occupation"]
ITEM_LEVEL = [f for f in FEATURES if f.startswith("item_")]
CATEGORICAL = ["gender", "age", "occupation"]


class FeatureBuilder:
    def __init__(self, ctx: Context, movies: pd.DataFrame, users: pd.DataFrame, settings):
        """`movies` and `users` must be in catalog order (one row per item / user position)."""
        hist = ctx.history
        cat = ctx.catalog
        ui = cat.user_pos(hist["user"])
        ii = cat.item_pos(hist["item"])
        rating = hist["rating"].to_numpy(dtype=np.float64)
        prior = float(rating.mean()) if len(rating) else 3.5

        # Movies
        counts = np.bincount(ii, minlength=cat.n_items).astype(np.float64)
        recent_mask = (hist["ts"] >= ctx.cutoff - settings.recent_days * DAY).to_numpy()
        recent = np.bincount(ii[recent_mask], minlength=cat.n_items).astype(np.float64)
        span_days = max((ctx.cutoff - hist["ts"].min()) / DAY, 1.0) if len(hist) else 1.0
        expected_recent = counts * settings.recent_days / span_days
        rating_sum = np.bincount(ii, weights=rating, minlength=cat.n_items)
        cutoff_year = dt.datetime.fromtimestamp(ctx.cutoff, dt.timezone.utc).year
        year = movies["year"].to_numpy(dtype=np.float64)
        self.genres = genre_matrix(movies)
        self.item = {
            "item_log_pop": np.log1p(counts),
            "item_log_recent": np.log1p(recent),
            "item_trend": np.log1p(recent) - np.log1p(expected_recent),
            "item_avg_rating": (rating_sum + prior * 10) / (counts + 10),  # shrunk towards the mean
            "item_age": cutoff_year - year,
        }

        # Users
        u_counts = np.bincount(ui, minlength=cat.n_users).astype(np.float64)
        u_sum = np.bincount(ui, weights=rating, minlength=cat.n_users)
        last = pd.Series(hist["ts"].to_numpy(), index=ui).groupby(level=0).max()
        first = pd.Series(hist["ts"].to_numpy(), index=ui).groupby(level=0).min()
        days_since_last = np.full(cat.n_users, np.nan)
        days_active = np.full(cat.n_users, np.nan)
        days_since_last[last.index] = (ctx.cutoff - last.to_numpy()) / DAY
        days_active[first.index] = (ctx.cutoff - first.to_numpy()) / DAY
        mainstream = np.asarray(ctx.X @ np.log1p(counts), dtype=np.float64).ravel() / np.maximum(u_counts, 1)
        self.user = {
            "user_log_hist": np.log1p(u_counts),
            "user_avg_rating": (u_sum + prior * 5) / (u_counts + 5),
            "user_mainstream": np.where(u_counts > 0, mainstream, np.nan),
            "user_days_since_last": days_since_last,
            "user_days_active": days_active,
            "gender": (users["gender"].to_numpy() == "F").astype(np.int64),
            "age": users["age"].to_numpy(dtype=np.int64),
            "occupation": users["occupation"].to_numpy(dtype=np.int64),
        }

        # Taste: the genre mix of what each user watched, and each movie's genres (unit length)
        profile = np.asarray(ctx.X @ self.genres, dtype=np.float32)
        self.profile = profile / np.maximum(np.linalg.norm(profile, axis=1, keepdims=True), 1e-9)
        self.item_genres = self.genres / np.maximum(np.linalg.norm(self.genres, axis=1, keepdims=True), 1e-9)

        self.segment = SegmentPopular(settings.segment_days).fit(ctx, users)

    def transform(self, frame: pd.DataFrame) -> pd.DataFrame:
        out = frame.copy()
        u = out["user"].to_numpy()
        i = out["item"].to_numpy()
        for name, values in self.item.items():
            out[name] = values[i]
        for name, values in self.user.items():
            out[name] = values[u]
        out["genre_affinity"] = np.einsum("ij,ij->i", self.profile[u], self.item_genres[i])
        out["segment_share"] = np.log1p(1e4 * self.segment.share[self.segment.seg[u], i])
        return out


def add_labels(frame: pd.DataFrame, truth: dict[int, set[int]]) -> pd.DataFrame:
    """label = 1 if the user went on to watch the candidate in the target period."""
    pairs = pd.DataFrame([(u, i) for u, items in truth.items() for i in items], columns=["user", "item"])
    pairs["label"] = 1
    out = frame.merge(pairs, on=["user", "item"], how="left")
    out["label"] = out["label"].fillna(0).astype(np.int8)
    return out
