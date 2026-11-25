"""Top-K ranking metrics. Every method is scored on exactly the same users."""

from __future__ import annotations

from typing import Mapping, Sequence

import numpy as np


def ranking_metrics(recs: Mapping[int, Sequence[int]], truth: Mapping[int, set[int]], k: int,
                    n_items: int) -> dict:
    """Recall, NDCG, hit rate and precision at k (averaged over users), plus catalog coverage.

    recall@k  = share of a user's next movies that made it into their top k
    ndcg@k    = like recall, but a hit at rank 1 counts more than one at rank 10
    hit@k     = share of users with at least one hit in their top k
    coverage  = share of the whole catalog recommended to anyone (popularity bias check)
    """
    users = [u for u, items in truth.items() if items]
    discounts = 1.0 / np.log2(np.arange(2, k + 2))
    recall, ndcg, hit, precision = [], [], [], []
    shown: set[int] = set()
    for u in users:
        top = list(recs.get(u, []))[:k]
        shown.update(top)
        relevant = truth[u]
        gains = np.array([1.0 if item in relevant else 0.0 for item in top] + [0.0] * (k - len(top)))
        hits = gains.sum()
        ideal = discounts[: min(k, len(relevant))].sum()
        recall.append(hits / len(relevant))
        ndcg.append(float(gains @ discounts) / ideal)
        hit.append(1.0 if hits > 0 else 0.0)
        precision.append(hits / k)
    if not users:
        return {"users": 0, "recall": None, "ndcg": None, "hit": None, "precision": None, "coverage": None}
    return {
        "users": len(users),
        "recall": float(np.mean(recall)),
        "ndcg": float(np.mean(ndcg)),
        "hit": float(np.mean(hit)),
        "precision": float(np.mean(precision)),
        "coverage": len(shown) / n_items,
    }


def candidate_recall(candidates: Mapping[int, set[int]], truth: Mapping[int, set[int]]) -> float:
    """Share of each user's next movies present in their candidate set (the reranker's ceiling)."""
    users = [u for u, items in truth.items() if items]
    if not users:
        return float("nan")
    return float(np.mean([len(candidates.get(u, set()) & truth[u]) / len(truth[u]) for u in users]))
