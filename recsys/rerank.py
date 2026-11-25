"""Stage 2, reranking: a gradient-boosted model orders each user's candidates.

It learns from the validation period: for users who were active then, which of the
candidates the retrieval models proposed (from history before that period) did they
actually go on to watch? That's a classification target per (user, movie) pair, and
sorting by the predicted probability gives the final top-k (pointwise learning to rank).
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier

from .features import CATEGORICAL, FEATURES, ITEM_LEVEL, USER_LEVEL

log = logging.getLogger(__name__)


class Reranker:
    def __init__(self, seed: int = 0, max_rows: int = 600_000):
        self.seed, self.max_rows = seed, max_rows

    def fit(self, frame: pd.DataFrame) -> "Reranker":
        # Users none of whose candidates were watched can't teach the model how to order
        # candidates, only how active people are, so they are left out.
        useful = frame.groupby("user")["label"].transform("max") == 1
        frame = frame[useful]
        pos = frame[frame["label"] == 1]
        neg = frame[frame["label"] == 0]
        if len(frame) > self.max_rows:  # keep every positive, sample the negatives
            neg = neg.sample(n=max(self.max_rows - len(pos), len(pos)), random_state=self.seed)
        data = pd.concat([pos, neg])
        self.features = [c for c in FEATURES if c in frame.columns]
        self.model = HistGradientBoostingClassifier(
            learning_rate=0.08, max_iter=500, early_stopping=True, validation_fraction=0.1,
            n_iter_no_change=20, max_leaf_nodes=31, min_samples_leaf=100, l2_regularization=1.0,
            categorical_features=[c for c in CATEGORICAL if c in self.features], random_state=self.seed,
        )
        self.model.fit(data[self.features], data["label"])
        log.info("  reranker: %d rows (%d positives), %d trees", len(data), len(pos), self.model.n_iter_)
        return self

    def score(self, frame: pd.DataFrame) -> np.ndarray:
        return self.model.predict_proba(frame[self.features])[:, 1]

    def importance(self, frame: pd.DataFrame, k: int = 10, users: int = 800, repeats: int = 2) -> pd.Series:
        """How much NDCG@k drops when one feature is shuffled (permutation importance).

        Measured on each user's own candidate list, so it reflects what the reranker is
        for: ordering one person's candidates. Only users with a hit among them count.
        User features are shuffled between users and movie features between movies, so a
        person (or movie) keeps one consistent value across their rows.
        """
        rng = np.random.default_rng(self.seed)
        hit_users = frame.loc[frame["label"] == 1, "user"].unique()
        if len(hit_users) == 0:
            return pd.Series(dtype=float)
        pick = rng.choice(hit_users, size=min(users, len(hit_users)), replace=False)
        data = frame[frame["user"].isin(pick)].reset_index(drop=True)
        base = list_ndcg(data, self.score(data), k)
        drops = {}
        for f in self.features:
            losses = []
            for _ in range(repeats):
                shuffled = data.copy()
                shuffled[f] = _shuffle(data, f, rng)
                losses.append(base - list_ndcg(shuffled, self.score(shuffled), k))
            drops[f] = float(np.mean(losses))
        return pd.Series(drops).sort_values(ascending=False)


def _shuffle(data: pd.DataFrame, feature: str, rng: np.random.Generator) -> np.ndarray:
    key = "user" if feature in USER_LEVEL else "item" if feature in ITEM_LEVEL else None
    if key is None:
        return rng.permutation(data[feature].to_numpy())
    values = data.groupby(key)[feature].first()
    swapped = pd.Series(rng.permutation(values.to_numpy()), index=values.index)
    return swapped.loc[data[key]].to_numpy()


def list_ndcg(frame: pd.DataFrame, scores: np.ndarray, k: int) -> float:
    """Mean NDCG@k when each user's candidates are sorted by `scores` (hits = label 1)."""
    d = pd.DataFrame({"user": frame["user"].to_numpy(), "label": frame["label"].to_numpy(), "s": scores})
    d = d.sort_values(["user", "s"], ascending=[True, False], kind="stable")
    rank = d.groupby("user", sort=False).cumcount().to_numpy()
    gain = np.where(rank < k, d["label"].to_numpy() / np.log2(rank + 2), 0.0)
    dcg = pd.Series(gain).groupby(d["user"].to_numpy()).sum()
    positives = d.groupby("user")["label"].sum()
    discounts = np.concatenate([[0.0], np.cumsum(1.0 / np.log2(np.arange(2, k + 2)))])
    ideal = discounts[np.minimum(positives.to_numpy(), k)]
    keep = ideal > 0
    return float(np.mean(dcg.loc[positives.index].to_numpy()[keep] / ideal[keep])) if keep.any() else 0.0


def top_k(frame: pd.DataFrame, score_col: str, k: int) -> dict[int, list[int]]:
    """user -> their k highest-scoring items."""
    ordered = frame.sort_values(["user", score_col], ascending=[True, False], kind="stable")
    head = ordered.groupby("user", sort=False).head(k)
    return {u: list(g) for u, g in head.groupby("user", sort=False)["item"]}
