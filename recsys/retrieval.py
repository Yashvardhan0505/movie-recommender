"""Stage 1, candidate retrieval: several cheap models each propose a few hundred movies.

Each model scores every movie for a user; we keep its top N unseen movies. The
reranker (stage 2) then only has to order the union of these short lists, which is
what makes two-stage systems scale: the expensive model never sees the whole catalog.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.decomposition import TruncatedSVD

log = logging.getLogger(__name__)
DAY = 86_400


class Catalog:
    """Maps MovieLens user and movie ids to matrix rows and columns."""

    def __init__(self, user_ids, item_ids):
        self.user_ids = np.sort(np.unique(np.asarray(user_ids, dtype=np.int64)))
        self.item_ids = np.sort(np.unique(np.asarray(item_ids, dtype=np.int64)))
        self.n_users, self.n_items = len(self.user_ids), len(self.item_ids)

    def user_pos(self, ids) -> np.ndarray:
        return np.searchsorted(self.user_ids, np.asarray(ids, dtype=np.int64))

    def item_pos(self, ids) -> np.ndarray:
        return np.searchsorted(self.item_ids, np.asarray(ids, dtype=np.int64))


@dataclass
class Context:
    """Everything a model may use: the interactions before one cutoff."""

    catalog: Catalog
    history: pd.DataFrame
    cutoff: int
    X: sparse.csr_matrix  # users x items, 1 where the user rated the movie before the cutoff

    @classmethod
    def build(cls, catalog: Catalog, history: pd.DataFrame, cutoff: int) -> "Context":
        rows = catalog.user_pos(history["user"])
        cols = catalog.item_pos(history["item"])
        X = sparse.csr_matrix((np.ones(len(rows), dtype=np.float32), (rows, cols)),
                              shape=(catalog.n_users, catalog.n_items))
        X.data[:] = 1.0  # duplicates would have summed
        return cls(catalog, history, cutoff, X)


class MostPopular:
    """The baseline: the most-rated movies, the same list for everyone."""

    name = "popular"

    def fit(self, ctx: Context) -> "MostPopular":
        self.pop = np.asarray(ctx.X.sum(axis=0), dtype=np.float32).ravel()
        return self

    def scores(self, rows: np.ndarray) -> np.ndarray:
        return np.tile(self.pop, (len(rows), 1))


class RecentPopular(MostPopular):
    """Most-rated in the last `days` before the cutoff: popularity that follows trends."""

    name = "recent"

    def __init__(self, days: int = 30):
        self.days = days

    def fit(self, ctx: Context) -> "RecentPopular":
        recent = ctx.history[ctx.history["ts"] >= ctx.cutoff - self.days * DAY]
        self.pop = np.bincount(ctx.catalog.item_pos(recent["item"]),
                               minlength=ctx.catalog.n_items).astype(np.float32)
        return self


class ItemCF:
    """Item-to-item collaborative filtering: "people who watched X also watched Y".

    Cosine similarity between movies' audiences, keeping each movie's top neighbours.
    A user's score for a movie is the summed similarity to everything they watched,
    which also makes the recommendation easy to explain.
    """

    name = "itemcf"

    def __init__(self, neighbors: int = 100):
        self.neighbors = neighbors

    def fit(self, ctx: Context) -> "ItemCF":
        self.sim = item_similarity(ctx.X, self.neighbors)
        self.X = ctx.X
        return self

    def scores(self, rows: np.ndarray) -> np.ndarray:
        return np.asarray(self.X[rows] @ self.sim, dtype=np.float32)


def item_similarity(X: sparse.csr_matrix, neighbors: int) -> np.ndarray:
    dense = X.toarray().astype(np.float32)
    co = dense.T @ dense                         # how many users rated both movies
    norm = 1.0 / np.sqrt(np.maximum(np.diag(co), 1.0))
    sim = co * norm[:, None] * norm[None, :]     # cosine similarity
    np.fill_diagonal(sim, 0.0)
    if neighbors < sim.shape[1]:
        drop = np.argpartition(-sim, neighbors, axis=1)[:, neighbors:]
        np.put_along_axis(sim, drop, 0.0, axis=1)
    return sim.astype(np.float32)


class PureSVD:
    """Matrix factorisation: users and movies as 64-number taste vectors (truncated SVD)."""

    name = "svd"

    def __init__(self, factors: int = 64, seed: int = 0):
        self.factors, self.seed = factors, seed

    def fit(self, ctx: Context) -> "PureSVD":
        k = min(self.factors, min(ctx.X.shape) - 1)
        svd = TruncatedSVD(n_components=k, algorithm="randomized", n_iter=7, random_state=self.seed)
        svd.fit(ctx.X)
        self.V = svd.components_.T.astype(np.float32)  # items x factors
        self.X = ctx.X
        return self

    def scores(self, rows: np.ndarray) -> np.ndarray:
        return np.asarray((self.X[rows] @ self.V) @ self.V.T, dtype=np.float32)


class EASE:
    """EASE (Steck, 2019): a linear item-to-item model with a closed-form solution.

    It learns how much each movie predicts each other movie, with one regularisation
    knob. Simple, and among the strongest classic models on MovieLens.
    """

    name = "ease"

    def __init__(self, lam: float = 300.0):
        self.lam = lam

    def fit(self, ctx: Context) -> "EASE":
        dense = ctx.X.toarray().astype(np.float32)
        gram = (dense.T @ dense).astype(np.float64)
        gram[np.diag_indices_from(gram)] += self.lam
        P = np.linalg.inv(gram)
        B = P / (-np.diag(P))
        np.fill_diagonal(B, 0.0)
        self.B = B.astype(np.float32)
        self.X = ctx.X
        return self

    def scores(self, rows: np.ndarray) -> np.ndarray:
        return np.asarray(self.X[rows] @ self.B, dtype=np.float32)


def fit_models(ctx: Context, settings) -> dict:
    models = [MostPopular(), RecentPopular(settings.recent_days), ItemCF(settings.itemcf_neighbors),
              PureSVD(settings.svd_factors, settings.seed), EASE(settings.ease_lambda)]
    out = {}
    for m in models:
        out[m.name] = m.fit(ctx)
        log.info("  fitted %s", m.name)
    return out


def _best(s: np.ndarray, n: int) -> tuple[np.ndarray, np.ndarray]:
    """Column positions and values of each row's n largest entries, best first."""
    idx = np.argpartition(-s, n - 1, axis=1)[:, :n]
    part = np.take_along_axis(s, idx, axis=1)
    order = np.argsort(-part, axis=1, kind="stable")
    return np.take_along_axis(idx, order, axis=1), np.take_along_axis(part, order, axis=1)


def _unseen_scores(model, ctx: Context, rows: np.ndarray) -> np.ndarray:
    s = np.array(model.scores(rows), dtype=np.float32, copy=True)
    seen_r, seen_c = ctx.X[rows].nonzero()
    s[seen_r, seen_c] = -np.inf  # never recommend something already watched
    return s


def top_n(model, ctx: Context, rows: np.ndarray, n: int, batch: int = 1024) -> tuple[np.ndarray, np.ndarray]:
    """Each user's n best unseen movies by this model: (item positions, scores), best first."""
    n = min(n, ctx.catalog.n_items - 1)
    parts = [_best(_unseen_scores(model, ctx, rows[i:i + batch]), n) for i in range(0, len(rows), batch)]
    if not parts:
        return np.empty((0, n), dtype=np.int64), np.empty((0, n), dtype=np.float32)
    return np.vstack([p[0] for p in parts]), np.vstack([p[1] for p in parts])


def candidates(models: dict, ctx: Context, rows: np.ndarray, sizes: dict[str, int],
               batch: int = 512) -> tuple[pd.DataFrame, dict]:
    """The union of every model's top-N lists, one row per (user, movie).

    Every candidate gets every model's score and exact rank among the user's unseen
    movies (1 = that model's favourite), whichever model proposed it, so the reranker
    can weigh all opinions. n_sources counts the top-N lists it appeared in. Also
    returns each model's top lists as {name: (item positions, scores)}.
    """
    n_items = ctx.catalog.n_items
    sizes = {name: min(n, n_items - 1) for name, n in sizes.items()}
    tops = {name: ([], []) for name in sizes}
    frames = []
    for start in range(0, len(rows), batch):
        r = rows[start:start + batch]
        b = np.arange(len(r))[:, None]
        dense = {}
        proposed = np.zeros((len(r), n_items), dtype=np.int8)
        for name, n in sizes.items():
            s = _unseen_scores(models[name], ctx, r)
            idx, best = _best(s, n)
            tops[name][0].append(idx)
            tops[name][1].append(best)
            np.add.at(proposed, (np.broadcast_to(b, idx.shape)[np.isfinite(best)], idx[np.isfinite(best)]), 1)
            dense[name] = s
        ub, ui = np.nonzero(proposed)
        cols = {"user": r[ub], "item": ui, "n_sources": proposed[ub, ui].astype(np.int16)}
        for name, s in dense.items():
            rank = np.empty(s.shape, dtype=np.int32)
            np.put_along_axis(rank, np.argsort(-s, axis=1, kind="stable"), np.arange(1, n_items + 1, dtype=np.int32), axis=1)
            cols[f"{name}_score"] = s[ub, ui]
            cols[f"{name}_rank"] = rank[ub, ui]
        frames.append(pd.DataFrame(cols))
    empty = {name: (np.empty((0, n), dtype=np.int64), np.empty((0, n), dtype=np.float32)) for name, n in sizes.items()}
    tops = {name: (np.vstack(i), np.vstack(s)) if i else empty[name] for name, (i, s) in tops.items()}
    frame = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=["user", "item", "n_sources"])
    return frame, tops
