import json

import numpy as np
import pandas as pd
import pytest

from recsys import pipeline
from recsys.rerank import list_ndcg, top_k


@pytest.fixture(scope="module")
def run(dataset, tmp_path_factory):
    out = tmp_path_factory.mktemp("run")
    readme = out / "README.md"
    readme.write_text("# Title\n\n<!-- results:start -->\nold\n<!-- results:end -->\n\nMore text.\n")
    results = pipeline.run(site_dir=out / "docs", results_dir=out / "results", dataset=dataset,
                           readme=readme, note="test run", source="synthetic")
    return results, out


def test_two_stage_beats_the_popularity_baseline(run):
    results, _ = run
    warm = {m["method"]: m for m in results["warm"]}
    assert warm["two_stage"]["recall"] > 1.5 * warm["popular"]["recall"]
    assert warm["two_stage"]["ndcg"] > 1.5 * warm["popular"]["ndcg"]
    users = {m["users"] for m in results["warm"]}
    assert len(users) == 1, "every method is scored on the same users"


def test_new_users_get_segment_popularity(run):
    results, _ = run
    cold = {m["method"]: m for m in results["cold"]}
    assert cold["segment"]["users"] > 0
    assert cold["segment"]["recall"] >= cold["popular"]["recall"]


def test_shortlist_recall_is_the_ceiling(run):
    results, _ = run
    retrieval = {m["source"]: m["recall"] for m in results["retrieval"]}
    warm = {m["method"]: m["recall"] for m in results["warm"]}
    assert retrieval["union"] >= max(v for k, v in retrieval.items() if k != "union")
    assert warm["two_stage"] <= retrieval["union"] + 1e-9


def test_outputs(run):
    results, out = run
    saved = json.loads((out / "results" / "metrics.json").read_text())
    assert saved["warm"] == results["warm"]
    assert "| Two-stage: retrieve + rerank |" in (out / "results" / "metrics.md").read_text()
    page = (out / "docs" / "index.html").read_text(encoding="utf-8")
    assert "{{" not in page and "}}" not in page
    assert "synthetic ratings" in page
    demo = json.loads(page.split('<script id="demo-data" type="application/json">')[1].split("</script>")[0])
    assert set(demo) == {"movies", "nb", "ns", "quick", "segments", "ages"}  # no ratings or users
    assert len(demo["nb"]) == len(demo["movies"]) == len(demo["ns"])
    assert all(len(v) == 10 for v in demo["segments"].values())
    readme = (out / "README.md").read_text()
    assert "old" not in readme and "Recall@10" in readme and readme.endswith("More text.\n")


def test_list_ndcg_and_top_k():
    frame = pd.DataFrame({"user": [1, 1, 1, 2, 2], "item": [10, 11, 12, 20, 21], "label": [0, 1, 0, 1, 0]})
    scores = np.array([0.9, 0.8, 0.1, 0.2, 0.7])
    expected_1 = 1 / np.log2(3)       # hit at rank 2, one relevant
    expected_2 = 1 / np.log2(3)
    assert list_ndcg(frame, scores, k=10) == pytest.approx((expected_1 + expected_2) / 2)
    frame["p"] = scores
    assert top_k(frame, "p", 2) == {1: [10, 11], 2: [21, 20]}
