"""Build the static page: the demo, the results and how it works, in one HTML file.

Results are rendered here in Python, so they show without JavaScript; the demo's
data rides along as inline JSON and runs in the browser.
"""

from __future__ import annotations

import datetime as dt
import html
import json
from pathlib import Path

from .features import FEATURES

TEMPLATE = Path(__file__).with_name("page.html")


def write(site_dir: str | Path, results: dict, demo: dict, repo_url: str = "") -> Path:
    site_dir = Path(site_dir)
    site_dir.mkdir(parents=True, exist_ok=True)
    out = site_dir / "index.html"
    out.write_text(render(results, demo, repo_url), encoding="utf-8")
    return out


def render(r: dict, demo: dict, repo_url: str = "") -> str:
    k = r["k"]
    warm = {m["method"]: m for m in r["warm"]}
    cold = {m["method"]: m for m in r["cold"]}
    sources = {m["source"]: m for m in r["retrieval"]}
    two, pop = warm["two_stage"], warm["popular"]
    best = max((m for m in r["warm"] if m["method"] not in ("two_stage", "popular", "recent")),
               key=lambda m: m["recall"])
    d, sp = r["dataset"], r["split"]
    union = sources.get("union", {"n": 0, "recall": 0.0})
    test_start, val_start = _day(sp["test_start"]), _day(sp["validation_start"])

    real = d.get("source", "MovieLens 1M") != "synthetic"
    lead = (f"A two-stage recommender trained on {d['ratings']:,} {'MovieLens' if real else 'synthetic'} ratings. "
            f"Five quick models shortlist "
            f"about {union['n']} movies for each person, then a gradient-boosted model reranks them. On ratings "
            f"made after the training cutoff, its top {k} found {_pct(two['recall'])} of the movies people went "
            f"on to rate, against {_pct(pop['recall'])} for a most-popular list.")
    ratio = two["recall"] / pop["recall"] if pop["recall"] else float("nan")
    results_lead = (
        f"Every model learned only from ratings made before {test_start}. Then it had to predict what "
        f"{sp['test_users']:,} returning users rated after that, about {sp['test_movies_per_user']:.0f} movies "
        f"each. Recall@{k} is the share of those movies that made a person's top {k}. NDCG@{k} also gives more "
        f"credit for ranking them near the top. The two-stage model's recall is {ratio:.1f}× the baseline's, and "
        f"the best single model, {best['label']}, reaches {_pct(best['recall'])} on its own.")

    seg, cpop = cold.get("segment", {}), cold.get("popular", {})
    if seg.get("recall") is not None and cpop.get("recall") is not None:
        cold_lead = (f"{sp['new_users']:,} people joined after {test_start}, so all the system knows about them "
                     f"is the age group and gender they gave when signing up. Recommending what's popular in "
                     f"that group found {_pct(seg['recall'])} of their first movies; the overall most-popular "
                     f"list found {_pct(cpop['recall'])}.")
    else:
        cold_lead = f"No new users joined after {test_start} in this data, so there is nothing to score here."

    repo_link = (f'<a href="{html.escape(repo_url)}">Code on GitHub</a>' if repo_url else "")
    banner = f'<p class="banner" role="note">{html.escape(r["note"])}</p>' if r.get("note") else ""
    fill = {
        "banner": banner,
        "lead": html.escape(lead),
        "repo_link": repo_link,
        "n_demo": f"{len(demo['movies']):,}",
        "k": str(k),
        "demo_note": html.escape(
            f"The demo runs the item-to-item model in your browser, over the {len(demo['movies']):,} most-rated "
            f"movies. The numbers below are for the full two-stage system."),
        "results_lead": html.escape(results_lead),
        "warm_chart": _warm_chart(r["warm"], k, pop),
        "warm_table": _warm_table(r["warm"], k),
        "retrieval_lead": html.escape(
            f"Share of the movies people went on to rate that made it into each model's shortlist. The reranker "
            f"can only pick from the combined list, so {_pct(union['recall'])} is as good as it can get."),
        "retrieval_chart": _retrieval_chart(r["retrieval"]),
        "retrieval_table": _retrieval_table(r["retrieval"]),
        "cold_lead": html.escape(cold_lead),
        "cold_chart": _cold_chart(r["cold"], k, cpop),
        "cold_table": _cold_table(r["cold"], k),
        "importance_lead": html.escape(
            f"How much the reranker's NDCG@{k} drops when one signal is shuffled between users or movies. "
            f"The bigger the drop, the more it relies on that signal."),
        "importance_chart": _importance_chart(r["importance"][:10], k),
        "importance_table": _importance_table(r["importance"], k),
        "step_split": html.escape(
            f"Ratings before {val_start} are history. The reranker learns from what people rated between "
            f"{val_start} and {test_start}, and everything after {test_start} is the test. A random split "
            f"would let the models peek at the future."),
        "step_shortlist": html.escape(
            f"Most popular overall, most popular in the last 30 days, item-to-item similarity, matrix "
            f"factorisation (SVD) and EASE each propose their top 50 to 100 unseen movies. Together that's "
            f"about {union['n']} per person, holding {_pct(union['recall'])} of the movies they'll rate next."),
        "step_rerank": html.escape(
            f"A gradient-boosted classifier scores each candidate from {len(FEATURES)} signals: the models' "
            f"scores and ranks, the movie's popularity, trend, age and average rating, how mainstream the "
            f"person's taste is, how well the genres match, and how popular it is with people of the same age "
            f"and gender. The top {k} are the recommendations."),
        "step_new": html.escape(
            "Someone with no ratings gets what has been popular lately with their age group and gender, mixed "
            "with overall popularity, until they've rated a few movies."),
        "dataset_line": html.escape(
            f"{'MovieLens 1M' if real else 'Synthetic data in MovieLens 1M format'}: {d['ratings']:,} ratings "
            f"(1 to 5 stars) from {d['users']:,} people on {d['movies']:,} movies, {_month(d['first'])} to "
            f"{_month(d['last'])}."),
        "built": html.escape(_day(r["generated_at"][:10])),
        "repo_footer": (f' Code: <a href="{html.escape(repo_url)}">{html.escape(_short(repo_url))}</a>, '
                        f'MIT licence.' if repo_url else " Code: MIT licence."),
        "data": _json(demo),
    }
    page = TEMPLATE.read_text(encoding="utf-8")
    for key, value in fill.items():
        page = page.replace("{{" + key + "}}", value)
    return page


# Charts: horizontal bars drawn with HTML, one hover/focus tooltip per bar.

def _cell(value: float, top: float, text: str, tip: str, label: str) -> str:
    share = value / top if top > 0 else 0.0
    return (f'<span class="cell" role="img" tabindex="0" aria-label="{html.escape(label)}" '
            f'data-tip="{html.escape(tip)}"><span class="bar" style="--f:{max(share, 0.0):.4f}"></span>'
            f'<span class="val">{html.escape(text)}</span></span>')


def _times(value: float, base: dict) -> str:
    if not base or not base.get("recall"):
        return ""
    return f"\n{value / base['recall']:.1f}× the most-popular baseline"


def _warm_chart(rows: list[dict], k: int, pop: dict) -> str:
    rows = sorted(rows, key=lambda m: -m["recall"])
    top_r = max(m["recall"] for m in rows)
    top_n = max(m["ndcg"] for m in rows)
    out = ['<div class="chart multiples" role="group" aria-label="Recall and NDCG for each method">',
           f'<span class="head" aria-hidden="true"></span><span class="head">Recall@{k}</span>'
           f'<span class="head">NDCG@{k}</span>']
    for m in rows:
        accent = " is-accent" if m["method"] == "two_stage" else ""
        name = html.escape(m["label"])
        hit = f"\n{_pct(m['hit'], 0)} of people had at least one hit"
        out.append(f'<span class="label{accent}">{name}</span>')
        out.append(f'<span class="track{accent}">' + _cell(
            m["recall"], top_r, _pct(m["recall"]),
            f"{m['label']}\nRecall@{k} {_pct(m['recall'])}{_times(m['recall'], pop)}{hit}",
            f"{m['label']}, recall at {k}: {_pct(m['recall'])}") + "</span>")
        out.append(f'<span class="track{accent}">' + _cell(
            m["ndcg"], top_n, f"{m['ndcg']:.3f}", f"{m['label']}\nNDCG@{k} {m['ndcg']:.3f}",
            f"{m['label']}, NDCG at {k}: {m['ndcg']:.3f}") + "</span>")
    out.append("</div>")
    return "\n".join(out)


def _bars(rows: list[tuple[str, float, str, str, bool]], aria: str) -> str:
    top = max((v for _, v, _, _, _ in rows), default=0.0)
    out = [f'<div class="chart single" role="group" aria-label="{html.escape(aria)}">']
    for label, value, text, tip, accent in rows:
        a = " is-accent" if accent else ""
        out.append(f'<span class="label{a}">{html.escape(label)}</span>')
        out.append(f'<span class="track{a}">' + _cell(value, top, text, tip, f"{label}: {text}") + "</span>")
    out.append("</div>")
    return "\n".join(out)


def _retrieval_chart(rows: list[dict]) -> str:
    rows = sorted(rows, key=lambda m: -m["recall"])
    return _bars([(f"{m['label']} ({m['n']})", m["recall"], _pct(m["recall"], 0),
                   f"{m['label']}\n{m['n']} candidates per person\nholds {_pct(m['recall'])} of the next movies",
                   m["source"] == "union") for m in rows], "Shortlist recall for each retrieval model")


def _cold_chart(rows: list[dict], k: int, base: dict) -> str:
    rows = [m for m in rows if m["recall"] is not None]
    if not rows:
        return ""
    rows = sorted(rows, key=lambda m: -m["recall"])
    return _bars([(m["label"], m["recall"], _pct(m["recall"]),
                   f"{m['label']}\nRecall@{k} {_pct(m['recall'])}{_times(m['recall'], base)}"
                   f"\nNDCG@{k} {m['ndcg']:.3f}", m["method"] == "segment") for m in rows],
                 "Recall for new users with no history")


def _importance_chart(rows: list[dict], k: int) -> str:
    rows = [f for f in rows if f["importance"] > 0]
    if not rows:
        return '<p class="muted">No signal made a measurable difference.</p>'
    return _bars([(f["label"], f["importance"], _drop(f["importance"]),
                   f"{f['label']}\nNDCG@{k} drops by {f['importance']:.4f} when shuffled", False) for f in rows],
                 "Drop in NDCG when each signal is shuffled")


# Tables: the same numbers, exact.

def _table(head: list[str], rows: list[list[str]], numeric_from: int = 1) -> str:
    th = "".join(f'<th scope="col"{" class=num" if i >= numeric_from else ""}>{html.escape(h)}</th>'
                 for i, h in enumerate(head))
    body = "".join("<tr>" + "".join(
        (f'<th scope="row">{c}</th>' if i == 0 else f'<td class="num">{c}</td>') for i, c in enumerate(row)) + "</tr>"
        for row in rows)
    return (f'<details class="numbers"><summary>Show the numbers</summary><div class="scroll">'
            f'<table><thead><tr>{th}</tr></thead><tbody>{body}</tbody></table></div></details>')


def _warm_table(rows: list[dict], k: int) -> str:
    rows = sorted(rows, key=lambda m: -m["recall"])
    return _table(["Method", f"Recall@{k}", f"NDCG@{k}", f"Hit rate@{k}", f"Precision@{k}", "Catalog shown"],
                  [[html.escape(m["label"]), f"{m['recall']:.4f}", f"{m['ndcg']:.4f}", f"{m['hit']:.3f}",
                    f"{m['precision']:.3f}", _pct(m["coverage"])] for m in rows])


def _retrieval_table(rows: list[dict]) -> str:
    rows = sorted(rows, key=lambda m: -m["recall"])
    return _table(["Shortlist from", "Candidates per person", "Share of next movies held"],
                  [[html.escape(m["label"]), str(m["n"]), f"{m['recall']:.3f}"] for m in rows])


def _cold_table(rows: list[dict], k: int) -> str:
    rows = sorted([m for m in rows if m["recall"] is not None], key=lambda m: -m["recall"])
    return _table(["Method", f"Recall@{k}", f"NDCG@{k}", f"Hit rate@{k}"],
                  [[html.escape(m["label"]), f"{m['recall']:.4f}", f"{m['ndcg']:.4f}", f"{m['hit']:.3f}"]
                   for m in rows]) if rows else ""


def _importance_table(rows: list[dict], k: int) -> str:
    return _table(["Signal", "Feature", f"NDCG@{k} drop"],
                  [[html.escape(f["label"]), f"<code>{html.escape(f['feature'])}</code>", f"{f['importance']:.4f}"]
                   for f in rows], numeric_from=2)


# Formatting

def _pct(x: float | None, digits: int = 1) -> str:
    return "n/a" if x is None else f"{100 * x:.{digits}f}%"


def _drop(x: float) -> str:
    return "<0.001" if x < 0.001 else f"{x:.3f}"


def _day(iso: str) -> str:
    d = dt.date.fromisoformat(iso[:10])
    return f"{d.day} {d:%B %Y}"


def _month(iso: str) -> str:
    return f"{dt.date.fromisoformat(iso[:10]):%B %Y}"


def _short(url: str) -> str:
    return url.split("://", 1)[-1]


def _json(obj) -> str:
    # Inline <script> data: "</" would end the script element early.
    return json.dumps(obj, ensure_ascii=False, separators=(",", ":")).replace("</", "<\\/")
