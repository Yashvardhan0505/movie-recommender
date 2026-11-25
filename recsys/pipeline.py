"""Train, evaluate and publish: the whole experiment in one function."""

from __future__ import annotations

import datetime as dt
import json
import logging
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from . import data, site
from .coldstart import SegmentPopular, segment_key
from .features import FEATURE_LABELS, FeatureBuilder, add_labels
from .metrics import candidate_recall, ranking_metrics
from .rerank import Reranker, top_k
from .retrieval import Catalog, Context, candidates, fit_models, item_similarity, top_n
from .split import temporal_split

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class Settings:
    val_quantile: float = 0.8      # validation period starts at this share of the timeline
    test_quantile: float = 0.9     # test period starts here
    k: int = 10                    # length of the recommendation list being scored
    recent_days: int = 30
    segment_days: int = 180
    itemcf_neighbors: int = 100
    svd_factors: int = 64
    ease_lambda: float = 300.0
    retrieval_n: tuple = (("popular", 50), ("recent", 50), ("itemcf", 100), ("svd", 100), ("ease", 100))
    max_train_rows: int = 600_000
    demo_movies: int = 1500        # movies searchable in the demo
    demo_neighbors: int = 20
    seed: int = 0


METHOD_LABELS = {
    "popular": "Most popular (baseline)",
    "recent": "Most popular, last 30 days",
    "itemcf": "Item-to-item",
    "svd": "Matrix factorisation (SVD)",
    "ease": "EASE",
    "two_stage": "Two-stage: retrieve + rerank",
    "segment": "Popular with similar people",
    "union": "All five combined",
}


def run(data_dir: str | Path = "data", site_dir: str | Path = "docs", results_dir: str | Path = "results",
        settings: Settings = Settings(), dataset: data.Dataset | None = None, repo_url: str = "",
        note: str = "", readme: str | Path | None = None, source: str = "MovieLens 1M") -> dict:
    started = time.time()
    s = settings
    ds = dataset or data.load(data.download(data_dir))
    catalog = Catalog(ds.users["user"], ds.movies["item"])
    movies = ds.movies.set_index("item").loc[catalog.item_ids].reset_index()
    users = ds.users.set_index("user").loc[catalog.user_ids].reset_index()
    ratings = ds.ratings[ds.ratings["user"].isin(catalog.user_ids) & ds.ratings["item"].isin(catalog.item_ids)]
    val, test = temporal_split(ratings, s.val_quantile, s.test_quantile)
    sizes = dict(s.retrieval_n)

    # 1. Learn the reranker on the validation period (retrieval sees only what came before it).
    log.info("Validation period from %s: fitting retrieval models", _date(val.cutoff))
    ctx_v = Context.build(catalog, val.history, val.cutoff)
    models_v = fit_models(ctx_v, s)
    rows_v = catalog.user_pos(val.warm_users())
    frame_v, _ = candidates(models_v, ctx_v, rows_v, sizes)
    frame_v = FeatureBuilder(ctx_v, movies, users, s).transform(frame_v)
    frame_v = add_labels(frame_v, _positions(val.truth(), catalog))
    reranker = Reranker(s.seed, s.max_train_rows).fit(frame_v)
    del models_v, frame_v

    # 2. Score the test period with models refit on everything before it.
    log.info("Test period from %s: refitting and scoring", _date(test.cutoff))
    ctx_t = Context.build(catalog, test.history, test.cutoff)
    models_t = fit_models(ctx_t, s)
    rows_t = catalog.user_pos(test.warm_users())
    truth_all = _positions(test.truth(), catalog)
    truth_warm = {u: truth_all[u] for u in rows_t}
    frame_t, tops = candidates(models_t, ctx_t, rows_t, sizes)
    builder = FeatureBuilder(ctx_t, movies, users, s)
    frame_t = add_labels(builder.transform(frame_t), truth_warm)
    frame_t["p"] = reranker.score(frame_t)

    recs = {name: {u: idx[j, :s.k].tolist() for j, u in enumerate(rows_t)} for name, (idx, _) in tops.items()}
    recs["two_stage"] = top_k(frame_t, "p", s.k)
    order = ["popular", "recent", "itemcf", "svd", "ease", "two_stage"]
    warm = [{"method": m, "label": METHOD_LABELS[m], **ranking_metrics(recs[m], truth_warm, s.k, catalog.n_items)}
            for m in order]

    retrieval = []
    for name, (idx, _) in tops.items():
        lists = {u: set(idx[j].tolist()) for j, u in enumerate(rows_t)}
        retrieval.append({"source": name, "label": METHOD_LABELS[name], "n": int(idx.shape[1]),
                          "recall": candidate_recall(lists, truth_warm)})
    union = {u: set(g) for u, g in frame_t.groupby("user")["item"]}
    retrieval.append({"source": "union", "label": METHOD_LABELS["union"],
                      "n": int(round(np.mean([len(v) for v in union.values()]))) if union else 0,
                      "recall": candidate_recall(union, truth_warm)})

    # 3. New users: no history, only gender and age group.
    rows_new = catalog.user_pos(test.new_users())
    truth_new = {u: truth_all[u] for u in rows_new}
    cold_models = {"popular": models_t["popular"], "recent": models_t["recent"], "segment": builder.segment}
    cold = []
    for name, model in cold_models.items():
        idx, _ = top_n(model, ctx_t, rows_new, s.k)
        lists = {u: idx[j].tolist() for j, u in enumerate(rows_new)}
        cold.append({"method": name, "label": METHOD_LABELS[name],
                     **ranking_metrics(lists, truth_new, s.k, catalog.n_items)})

    importance = reranker.importance(frame_t, s.k).head(12)
    results = {
        "generated_at": dt.datetime.now(dt.timezone.utc).isoformat(timespec="minutes"),
        "note": note,
        "k": s.k,
        "dataset": {"source": source, "users": int(ratings["user"].nunique()), "movies": int(ratings["item"].nunique()),
                    "ratings": int(len(ratings)), "first": _date(int(ratings["ts"].min())),
                    "last": _date(int(ratings["ts"].max()))},
        "split": {"validation_start": _date(val.cutoff), "test_start": _date(test.cutoff),
                  "validation_users": int(len(rows_v)), "test_users": int(len(rows_t)),
                  "new_users": int(len(rows_new)),
                  "test_movies_per_user": float(np.mean([len(v) for v in truth_warm.values()])) if truth_warm else 0.0},
        "warm": warm,
        "retrieval": retrieval,
        "cold": cold,
        "importance": [{"feature": f, "label": FEATURE_LABELS.get(f, f), "importance": float(v)}
                       for f, v in importance.items()],
        "settings": {k: (dict(v) if k == "retrieval_n" else v) for k, v in asdict(s).items()},
        "runtime_seconds": round(time.time() - started, 1),
    }
    demo = build_demo(ratings, catalog, movies, users, s)
    write_results(Path(results_dir), results)
    site.write(Path(site_dir), results, demo, repo_url)
    if readme and update_readme(Path(readme), results):
        log.info("Updated the results in %s", readme)
    _log_summary(results)
    return results


def build_demo(ratings: pd.DataFrame, catalog: Catalog, movies: pd.DataFrame, users: pd.DataFrame,
               s: Settings) -> dict:
    """What the in-browser demo needs: the most-rated movies, each one's most similar
    movies, and what's popular per age group and gender. Derived statistics only:
    no ratings and no users leave the build."""
    ctx = Context.build(catalog, ratings, int(ratings["ts"].max()) + 1)
    counts = np.asarray(ctx.X.sum(axis=0)).ravel()
    top = np.argsort(-counts, kind="stable")[: min(s.demo_movies, int((counts > 0).sum()))]
    sim = item_similarity(ctx.X, catalog.n_items)[np.ix_(top, top)]
    k = min(s.demo_neighbors, len(top) - 1)
    nb, ns = [], []
    for row in sim:
        best = [int(j) for j in np.argsort(-row, kind="stable")[:k] if row[j] > 0]
        nb.append(best)
        ns.append([max(1, int(round(float(row[j]) * 1000))) for j in best])

    demo_movies = [[int(catalog.item_ids[i]), str(movies["title"].iat[i]),
                    None if pd.isna(movies["year"].iat[i]) else int(movies["year"].iat[i]),
                    str(movies["genres"].iat[i]).replace("|", ", "), int(counts[i])] for i in top]

    # Quick picks: the most-rated movie of a few genres, for a one-click start.
    quick = []
    for genre in ("Sci-Fi", "Animation", "Comedy", "Drama", "Romance", "Horror"):
        for j, i in enumerate(top):
            if genre in str(movies["genres"].iat[i]).split("|") and j not in quick:
                quick.append(j)
                break

    seg = SegmentPopular(days=10**6).fit(ctx, users)  # all ratings, not just recent ones
    position = {int(i): j for j, i in enumerate(top)}
    ages = sorted(data.AGE_GROUPS)

    def best_of(share: np.ndarray) -> list[int]:
        order = np.argsort(-share, kind="stable")
        return [position[int(i)] for i in order if int(i) in position][:10]

    segments = {}
    for a, age in enumerate(ages):
        m, f = a, len(ages) + a  # segment ids: men first, then women (see coldstart.segment_ids)
        segments[segment_key("M", age)] = best_of(seg.share[m])
        segments[segment_key("F", age)] = best_of(seg.share[f])
        weight = seg.sizes[[m, f]] / max(seg.sizes[[m, f]].sum(), 1.0)
        segments[segment_key("any", age)] = best_of(weight[0] * seg.share[m] + weight[1] * seg.share[f])
    return {"movies": demo_movies, "nb": nb, "ns": ns, "quick": quick, "segments": segments,
            "ages": [[age, data.AGE_GROUPS[age]] for age in ages]}


README_START, README_END = "<!-- results:start -->", "<!-- results:end -->"


def update_readme(path: Path, results: dict) -> bool:
    """Replace the block between the results markers in the README with fresh numbers."""
    if not path.exists():
        return False
    text = path.read_text(encoding="utf-8")
    if README_START not in text or README_END not in text:
        return False
    head, rest = text.split(README_START, 1)
    _, tail = rest.split(README_END, 1)
    path.write_text(head + README_START + "\n" + readme_block(results) + README_END + tail, encoding="utf-8")
    return True


def readme_block(r: dict) -> str:
    k = r["k"]
    warm = {m["method"]: m for m in r["warm"]}
    cold = {m["method"]: m for m in r["cold"]}
    two, pop = warm["two_stage"], warm["popular"]
    lines = [f"| Method | Recall@{k} | NDCG@{k} |", "|---|---:|---:|"]
    for m in sorted(r["warm"], key=lambda m: -m["recall"]):
        name = f"**{m['label']}**" if m["method"] == "two_stage" else m["label"]
        lines.append(f"| {name} | {m['recall']:.4f} | {m['ndcg']:.4f} |")
    sp = r["split"]
    lines += ["", f"Returning users: {sp['test_users']:,}, scored on everything they rated after {sp['test_start']}."]
    seg, cpop = cold.get("segment", {}), cold.get("popular", {})
    if seg.get("recall") is not None and cpop.get("recall") is not None:
        lines.append(f"New users ({sp['new_users']:,}, no history): popular with their age group and gender "
                     f"Recall@{k} {seg['recall']:.4f}, against {cpop['recall']:.4f} for most popular overall.")
    ratio = two["recall"] / pop["recall"] if pop["recall"] else float("nan")
    new_users = "serves new users from age and gender segments"
    if seg.get("recall") is not None and cpop.get("recall") and seg["recall"] > cpop["recall"]:
        new_users += f" ({seg['recall'] / cpop['recall']:.1f}× the baseline's recall for them)"
    lines += ["", "Resume line, with these numbers:", "",
              f"> Built a two-stage movie recommender on {_source(r['dataset'])} ({r['dataset']['ratings']:,} "
              f"ratings): five retrieval models (two popularity lists, item-to-item, SVD and EASE) shortlist "
              f"candidates and a gradient-boosted reranker orders them. On a time-based split it reaches "
              f"Recall@{k} {two['recall']:.3f} and NDCG@{k} {two['ndcg']:.3f}, {ratio:.1f}× a most-popular "
              f"baseline, and {new_users}.",
              "", f"_Updated {r['generated_at'][:10]}. Full tables: [results/metrics.md](results/metrics.md)._", ""]
    if r.get("note"):
        lines.insert(0, f"_{r['note']}_\n")
    return "\n".join(lines)


def write_results(folder: Path, results: dict) -> None:
    folder.mkdir(parents=True, exist_ok=True)
    (folder / "metrics.json").write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
    (folder / "metrics.md").write_text(results_markdown(results), encoding="utf-8")


def results_markdown(r: dict) -> str:
    k = r["k"]
    d, sp = r["dataset"], r["split"]
    lines = [
        "# Results",
        "",
        f"{_source(d, sentence_start=True)}: {d['ratings']:,} ratings from {d['users']:,} users on {d['movies']:,} movies "
        f"({d['first']} to {d['last']}). Test period from {sp['test_start']}: {sp['test_users']:,} returning users "
        f"(on average {sp['test_movies_per_user']:.0f} movies each to predict) and {sp['new_users']:,} new users.",
        "",
        f"## Returning users (top {k})",
        "",
        f"| Method | Recall@{k} | NDCG@{k} | Hit rate@{k} | Catalog coverage |",
        "|---|---:|---:|---:|---:|",
    ]
    for m in r["warm"]:
        lines.append(f"| {m['label']} | {m['recall']:.4f} | {m['ndcg']:.4f} | {m['hit']:.3f} | {m['coverage']:.1%} |")
    lines += ["", "## Candidate retrieval (how many of the next movies made the shortlist)", "",
              "| Source | Candidates per user | Recall |", "|---|---:|---:|"]
    for m in r["retrieval"]:
        lines.append(f"| {m['label']} | {m['n']} | {m['recall']:.3f} |")
    lines += ["", f"## New users with no history (top {k})", "",
              f"| Method | Recall@{k} | NDCG@{k} | Hit rate@{k} |", "|---|---:|---:|---:|"]
    for m in r["cold"]:
        if m["recall"] is not None:
            lines.append(f"| {m['label']} | {m['recall']:.4f} | {m['ndcg']:.4f} | {m['hit']:.3f} |")
    lines += ["", "## What the reranker relies on (permutation importance)", ""]
    for f in r["importance"]:
        lines.append(f"- {f['label']} (`{f['feature']}`): {f['importance']:.4f}")
    if r.get("note"):
        lines += ["", f"_{r['note']}_"]
    return "\n".join(lines) + "\n"


def _source(d: dict, sentence_start: bool = False) -> str:
    if d.get("source") == "synthetic":
        return ("Synthetic" if sentence_start else "synthetic") + " data in MovieLens 1M format"
    return "MovieLens 1M"


def _positions(truth: dict[int, set[int]], catalog: Catalog) -> dict[int, set[int]]:
    out = {}
    for u, items in truth.items():
        out[int(catalog.user_pos([u])[0])] = set(catalog.item_pos(sorted(items)).tolist())
    return out


def _date(ts: int) -> str:
    return dt.datetime.fromtimestamp(ts, dt.timezone.utc).strftime("%Y-%m-%d")


def _log_summary(r: dict) -> None:
    for m in r["warm"]:
        log.info("%-30s recall@%d %.4f  ndcg@%d %.4f", m["label"], r["k"], m["recall"], r["k"], m["ndcg"])
    for m in r["cold"]:
        if m["recall"] is not None:
            log.info("New users, %-26s recall@%d %.4f", m["label"], r["k"], m["recall"])
