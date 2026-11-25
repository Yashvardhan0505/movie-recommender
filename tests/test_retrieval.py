from types import SimpleNamespace

import numpy as np
import pytest

from recsys.retrieval import EASE, Catalog, Context, candidates, fit_models, top_n
from recsys.split import temporal_split

SETTINGS = SimpleNamespace(recent_days=30, itemcf_neighbors=50, svd_factors=16, seed=0, ease_lambda=100.0)


@pytest.fixture(scope="module")
def fitted(dataset):
    _, test = temporal_split(dataset.ratings)
    catalog = Catalog(dataset.users["user"], dataset.movies["item"])
    ctx = Context.build(catalog, test.history, test.cutoff)
    rows = catalog.user_pos(test.warm_users())
    return ctx, fit_models(ctx, SETTINGS), rows


def test_never_recommends_seen_movies(fitted):
    ctx, models, rows = fitted
    for model in models.values():
        idx, scores = top_n(model, ctx, rows, 20)
        seen = ctx.X[rows].toarray() > 0
        assert not np.take_along_axis(seen, idx, axis=1).any(), model.name
        assert (np.diff(scores, axis=1) <= 1e-6).all(), "best first"


def test_ease_has_no_self_similarity(fitted):
    ctx, models, _ = fitted
    assert isinstance(models["ease"], EASE)
    assert np.allclose(np.diag(models["ease"].B), 0)


def test_candidates_union_scores_and_ranks(fitted):
    ctx, models, rows = fitted
    sizes = {"popular": 10, "itemcf": 20, "ease": 20}
    frame, tops = candidates(models, ctx, rows[:50], sizes)
    assert not frame.duplicated(["user", "item"]).any()
    assert frame[[c for c in frame.columns if c.endswith(("_score", "_rank"))]].notna().all().all()
    for name, (idx, _) in tops.items():
        pairs = set(zip(np.repeat(rows[:50], idx.shape[1]), idx.ravel()))
        assert pairs <= set(zip(frame["user"], frame["item"])), name
        # each model's favourite has rank 1 in its own column
        best = dict(zip(rows[:50], idx[:, 0]))
        firsts = frame[frame[f"{name}_rank"] == 1]
        assert dict(zip(firsts["user"], firsts["item"])) == best
    assert frame["n_sources"].between(1, len(sizes)).all()
