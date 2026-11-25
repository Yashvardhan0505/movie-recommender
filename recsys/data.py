"""Download and read MovieLens 1M from GroupLens.

The licence doesn't allow redistributing the data, so it isn't stored in this repo:
it's downloaded on the first run into data/ (which git ignores).
Harper & Konstan (2015), The MovieLens Datasets: History and Context, ACM TiiS 5(4).
"""

from __future__ import annotations

import csv
import io
import logging
import re
import urllib.request
import zipfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)

ML1M_URL = "https://files.grouplens.org/datasets/movielens/ml-1m.zip"

GENRES = ("Action", "Adventure", "Animation", "Children's", "Comedy", "Crime", "Documentary", "Drama",
          "Fantasy", "Film-Noir", "Horror", "Musical", "Mystery", "Romance", "Sci-Fi", "Thriller", "War",
          "Western")
AGE_GROUPS = {1: "under 18", 18: "18–24", 25: "25–34", 35: "35–44", 45: "45–49", 50: "50–55", 56: "56+"}


@dataclass
class Dataset:
    ratings: pd.DataFrame  # user, item, rating (1-5), ts (unix seconds)
    users: pd.DataFrame    # user, gender (M/F), age (code), occupation (0-20)
    movies: pd.DataFrame   # item, title, year, genres ("Comedy|Drama")


def download(data_dir: str | Path) -> Path:
    """The extracted ml-1m folder, downloading the 6 MB zip the first time."""
    folder = Path(data_dir) / "ml-1m"
    if (folder / "ratings.dat").exists():
        return folder
    folder.parent.mkdir(parents=True, exist_ok=True)
    archive = folder.parent / "ml-1m.zip"
    partial = archive.with_suffix(".part")
    log.info("Downloading MovieLens 1M from %s", ML1M_URL)
    urllib.request.urlretrieve(ML1M_URL, partial)
    try:
        with zipfile.ZipFile(partial) as z:
            z.extractall(folder.parent)
    except zipfile.BadZipFile as err:
        partial.unlink(missing_ok=True)
        raise RuntimeError(f"The download from {ML1M_URL} isn't a valid zip file; try again.") from err
    partial.replace(archive)
    return folder


def _read(path: Path, names: list[str]) -> pd.DataFrame:
    # Fields are separated by "::". Swapping it for a tab lets pandas use its fast parser;
    # titles contain single colons but never "::" or tabs.
    text = path.read_text(encoding="latin-1").replace("::", "\t")
    return pd.read_csv(io.StringIO(text), sep="\t", header=None, names=names, quoting=csv.QUOTE_NONE)


def load(folder: str | Path) -> Dataset:
    folder = Path(folder)
    ratings = _read(folder / "ratings.dat", ["user", "item", "rating", "ts"]).astype("int64")
    users = _read(folder / "users.dat", ["user", "gender", "age", "occupation", "zip"])
    users = users[["user", "gender", "age", "occupation"]].astype({"user": "int64", "age": "int64",
                                                                   "occupation": "int64"})
    movies = _read(folder / "movies.dat", ["item", "raw_title", "genres"])
    movies["year"] = pd.to_numeric(movies["raw_title"].str.extract(r"\((\d{4})\)\s*$")[0], errors="coerce")
    movies["title"] = movies["raw_title"].map(clean_title)
    movies["genres"] = movies["genres"].fillna("").astype(str)
    log.info("Loaded %d ratings from %d users on %d movies", len(ratings), len(users), len(movies))
    return Dataset(ratings, users, movies[["item", "title", "year", "genres"]])


_TRAILING_ARTICLE = re.compile(r"^(.*), (The|A|An|Les|La|Le|L'|Il|Das|Der|Die|El)$")
_ALT_TITLE = re.compile(r"^(.*?)(\s+\([^()]*\))$")


def clean_title(raw: str) -> str:
    """'Matrix, The (1999)' -> 'The Matrix'; keeps an alternate title in brackets."""
    title = re.sub(r"\s*\(\d{4}\)\s*$", "", str(raw)).strip()
    alt = ""
    m = _ALT_TITLE.match(title)
    if m and not re.fullmatch(r"\s+\(\d{4}\)", m.group(2)):
        title, alt = m.group(1), m.group(2)
    m = _TRAILING_ARTICLE.match(title)
    if m:
        article = m.group(2)
        title = f"{article}{m.group(1)}" if article.endswith("'") else f"{article} {m.group(1)}"
    return title + alt


def genre_matrix(movies: pd.DataFrame) -> np.ndarray:
    """Items x 18 genre indicators, in the order of `movies`."""
    out = np.zeros((len(movies), len(GENRES)), dtype=np.float32)
    lookup = {g: k for k, g in enumerate(GENRES)}
    for row, genres in enumerate(movies["genres"]):
        for g in str(genres).split("|"):
            if g in lookup:
                out[row, lookup[g]] = 1.0
    return out
