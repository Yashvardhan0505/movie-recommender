import numpy as np
import pytest

from recsys.data import GENRES, clean_title, genre_matrix


@pytest.mark.parametrize("raw, expected", [
    ("Toy Story (1995)", "Toy Story"),
    ("Matrix, The (1999)", "The Matrix"),
    ("Few Good Men, A (1992)", "A Few Good Men"),
    ("Godfather: Part II, The (1974)", "The Godfather: Part II"),
    # alternate titles stay, so a search for the original title still finds the film
    ("City of Lost Children, The (Cité des enfants perdus, La) (1995)",
     "The City of Lost Children (Cité des enfants perdus, La)"),
    ("Seven (Se7en) (1995)", "Seven (Se7en)"),
    ("Star Wars: Episode IV - A New Hope (1977)", "Star Wars: Episode IV - A New Hope"),
])
def test_clean_title(raw, expected):
    assert clean_title(raw) == expected


def test_load_reads_movielens_format(dataset):
    r, u, m = dataset.ratings, dataset.users, dataset.movies
    assert list(r.columns) == ["user", "item", "rating", "ts"]
    assert r["rating"].between(1, 5).all()
    assert set(u["gender"]) <= {"F", "M"}
    assert "Amélie of Synthetica" in set(m["title"])          # latin-1 decoded
    assert any(t.startswith("The Synthetic Picture") for t in m["title"])  # ", The" moved to the front
    assert m["year"].notna().all()
    assert not r.duplicated(["user", "item"]).any()


def test_genre_matrix(dataset):
    g = genre_matrix(dataset.movies)
    assert g.shape == (len(dataset.movies), len(GENRES))
    assert (g.sum(axis=1) >= 1).all()
    first = dataset.movies["genres"].iat[0].split("|")
    assert set(np.array(GENRES)[g[0] > 0]) == set(first)
