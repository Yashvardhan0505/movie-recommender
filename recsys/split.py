"""Split ratings by time, the way a recommender is actually used.

Everything before a cutoff is history; what users watch after it is what we try to
predict. A random split would let the model learn from the future (other people's
later ratings), which flatters offline numbers. Users whose first rating comes after
the cutoff have no history at all: they are the new-user (cold-start) test set.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class Period:
    name: str
    cutoff: int             # unix seconds; history is strictly before it
    history: pd.DataFrame   # what a model may learn from
    target: pd.DataFrame    # what users went on to watch in this period

    def truth(self) -> dict[int, set[int]]:
        """user -> movies they rated in the period."""
        return {u: set(g) for u, g in self.target.groupby("user")["item"]}

    def warm_users(self) -> np.ndarray:
        """Users active in the period who also have history before the cutoff."""
        active = self.target["user"].unique()
        return np.sort(active[np.isin(active, self.history["user"].unique())])

    def new_users(self) -> np.ndarray:
        """Users active in the period with no history at all (cold start)."""
        active = self.target["user"].unique()
        return np.sort(active[~np.isin(active, self.history["user"].unique())])


def temporal_split(ratings: pd.DataFrame, val_quantile: float = 0.8,
                   test_quantile: float = 0.9) -> tuple[Period, Period]:
    """Validation = ratings between the two time quantiles; test = everything after the second.

    The reranker learns on the validation period with retrieval fitted on what came
    before it; the test period is then scored with everything before the test cutoff.
    """
    ratings = ratings.sort_values("ts", kind="stable")
    t_val = int(ratings["ts"].quantile(val_quantile))
    t_test = int(ratings["ts"].quantile(test_quantile))
    val = Period("validation", t_val, ratings[ratings["ts"] < t_val],
                 ratings[(ratings["ts"] >= t_val) & (ratings["ts"] < t_test)])
    test = Period("test", t_test, ratings[ratings["ts"] < t_test], ratings[ratings["ts"] >= t_test])
    return val, test
