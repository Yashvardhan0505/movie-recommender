"""Synthetic data in MovieLens 1M's exact file format, for tests and the offline demo.

Users have tastes over genres (shaped by gender and age group), movies have genres,
popularity and a release date, and people tend to watch what's new and popular.
That gives the models real structure to find. The numbers mean nothing about real
movies; the page built from it says so.
"""

from __future__ import annotations

import datetime as dt
from pathlib import Path

import numpy as np

from .data import AGE_GROUPS, GENRES

START = int(dt.datetime(2000, 4, 25, tzinfo=dt.timezone.utc).timestamp())
DAY = 86_400


def write_synthetic(folder: str | Path, n_users: int = 1200, n_items: int = 600, mean_ratings: int = 60,
                    seed: int = 0, span_days: int = 700) -> Path:
    """Write ratings.dat, users.dat and movies.dat. Defaults are small (for tests);
    n_users=6040, n_items=3706, mean_ratings=165, span_days=1040 is about MovieLens 1M's size."""
    rng = np.random.default_rng(seed)
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    n_g = len(GENRES)

    # Movies: 1-3 genres, a popularity level, and a release day within the timeline.
    item_genres = np.zeros((n_items, n_g), dtype=np.float32)
    for i in range(n_items):
        item_genres[i, rng.choice(n_g, size=rng.integers(1, 4), replace=False)] = 1.0
    item_extra = rng.normal(0, 0.6, (n_items, 4))
    popularity = rng.permutation(1.0 / (np.arange(n_items) + 8.0) ** 0.9)
    early = int(span_days * 0.3)  # most of the activity happens in the first months, as in MovieLens
    release = np.where(rng.random(n_items) < 0.75, -rng.integers(0, 3000, n_items), rng.integers(0, early + 60, n_items))
    years = np.clip(2000 + release // 365, 1930, 2001)

    # Users: gender, age group, occupation; taste = their segment's taste plus their own.
    ages = np.array(sorted(AGE_GROUPS))
    gender = np.where(rng.random(n_users) < 0.28, "F", "M")
    age = rng.choice(ages, size=n_users, p=[0.04, 0.18, 0.35, 0.2, 0.09, 0.08, 0.06])
    occupation = rng.integers(0, 21, n_users)
    segment_taste = rng.normal(0, 1.0, (2, len(ages), n_g))
    taste = (segment_taste[(gender == "F").astype(int), np.searchsorted(ages, age)]
             + rng.normal(0, 0.8, (n_users, n_g)))
    extra = rng.normal(0, 0.6, (n_users, 4))

    # When people show up: most join early; a few keep trickling in later (the new users
    # of the test period). About a third come back to rate more, weeks or months later.
    late = np.minimum(early + rng.exponential(span_days * 0.2, n_users), span_days - 1)
    join = np.where(rng.random(n_users) < 0.86, rng.integers(0, early, n_users), late).astype(int)
    returner = rng.random(n_users) < 0.35
    n_ratings = np.clip((20 + rng.lognormal(np.log(mean_ratings - 15), 0.8, n_users)).astype(int), 20,
                        n_items // 2)

    lines = []
    for u in range(n_users):
        t_main = join[u]
        affinity = (item_genres @ taste[u]) / np.maximum(item_genres.sum(axis=1), 1) + item_extra @ extra[u]
        age_days = t_main - release
        fresh = np.where(age_days < 0, 0.02, 1.0 + 2.0 * np.exp(-np.maximum(age_days, 0) / 60.0))
        weight = popularity ** 0.7 * np.exp(1.4 * affinity) * fresh
        gumbel = -np.log(-np.log(rng.random(n_items)))
        chosen = np.argsort(-(np.log(weight) + gumbel))[: n_ratings[u]]
        later = returner[u] & (rng.random(len(chosen)) < 0.3)
        back = np.minimum(t_main + 1 + rng.exponential(span_days * 0.15, len(chosen)), span_days)
        days = np.where(later, back, t_main + rng.random(len(chosen)) * 3)
        ts = START + (days * DAY).astype(np.int64) + rng.integers(0, 3600, len(chosen))
        z = (affinity[chosen] - affinity.mean()) / (affinity.std() + 1e-9)
        stars = np.clip(np.round(3.4 + 0.9 * z + rng.normal(0, 0.8, len(chosen))), 1, 5).astype(int)
        for item, t, r in zip(chosen, ts, stars):
            lines.append(f"{u + 1}::{item + 1}::{r}::{t}")
    (folder / "ratings.dat").write_text("\n".join(lines) + "\n", encoding="latin-1")

    (folder / "users.dat").write_text("\n".join(
        f"{u + 1}::{gender[u]}::{age[u]}::{occupation[u]}::{30000 + u}" for u in range(n_users)) + "\n",
        encoding="latin-1")

    names = []
    for i in range(n_items):
        genres = "|".join(GENRES[g] for g in np.flatnonzero(item_genres[i]))
        title = f"Synthetic Movie {i + 1}"
        if i % 17 == 3:
            title = f"Synthetic Picture {i + 1}, The"
        if i == 5:
            title = "Amélie of Synthetica"
        names.append(f"{i + 1}::{title} ({years[i]})::{genres}")
    (folder / "movies.dat").write_text("\n".join(names) + "\n", encoding="latin-1")
    return folder
