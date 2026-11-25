from types import SimpleNamespace

import numpy as np
import pandas as pd

from recsys.features import FEATURES, FeatureBuilder, add_labels
from recsys.retrieval import Catalog, Context
from recsys.split import temporal_split

SETTINGS = SimpleNamespace(recent_days=30, segment_days=180)


def _build(dataset, ratings):
    catalog = Catalog(dataset.users["user"], dataset.movies["item"])
    movies = dataset.movies.set_index("item").loc[catalog.item_ids].reset_index()
    users = dataset.users.set_index("user").loc[catalog.user_ids].reset_index()
    _, test = temporal_split(ratings)
    ctx = Context.build(catalog, test.history, test.cutoff)
    pairs = pd.DataFrame({"user": np.repeat(np.arange(50), 30), "item": np.tile(np.arange(30), 50)})
    return FeatureBuilder(ctx, movies, users, SETTINGS).transform(pairs), test


def test_features_ignore_the_future(dataset):
    """Scrambling everything after the test cutoff must not change a single feature."""
    before, test = _build(dataset, dataset.ratings)
    future = dataset.ratings["ts"] >= test.cutoff
    scrambled = dataset.ratings.copy()
    scrambled.loc[future, "item"] = np.random.default_rng(0).permutation(scrambled.loc[future, "item"].to_numpy())
    scrambled.loc[future, "rating"] = 1
    after, after_test = _build(dataset, scrambled)
    assert after_test.cutoff == test.cutoff
    cols = [c for c in FEATURES if c in before.columns]
    pd.testing.assert_frame_equal(before[cols], after[cols])


def test_every_reranker_feature_is_built(dataset):
    built, _ = _build(dataset, dataset.ratings)
    retrieval = {f for f in FEATURES if f.split("_")[0] in {"itemcf", "svd", "ease"} or f == "n_sources"}
    assert set(FEATURES) - retrieval <= set(built.columns)


def test_add_labels():
    frame = pd.DataFrame({"user": [0, 0, 1], "item": [5, 6, 5]})
    out = add_labels(frame, {0: {6}, 1: {7}})
    assert out["label"].tolist() == [0, 1, 0]
