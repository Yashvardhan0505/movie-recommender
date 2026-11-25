"""Command line: `python -m recsys run` for MovieLens 1M, `python -m recsys demo` offline."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from . import data, pipeline
from .synthetic import write_synthetic

SYNTHETIC_NOTE = ("Synthetic preview: these numbers come from generated test data, not MovieLens. "
                  "Run python -m recsys run (or the GitHub Action) for real results.")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="python -m recsys", description="Two-stage movie recommender on MovieLens 1M.")
    sub = parser.add_subparsers(dest="command", required=True)
    run = sub.add_parser("run", help="download MovieLens 1M, train, evaluate and build the page")
    run.add_argument("--data", default="data", help="where MovieLens is downloaded (default: data)")
    run.add_argument("--site", default="docs", help="where the page is written (default: docs)")
    run.add_argument("--results", default="results", help="where metrics are written (default: results)")
    run.add_argument("--readme", default="README.md", help="README whose results block is refreshed")
    run.add_argument("--repo-url", default="", help="link to the code, shown on the page")
    demo = sub.add_parser("demo", help="the whole pipeline on synthetic data, no download (writes to preview/)")
    demo.add_argument("--out", default="preview", help="output folder (default: preview)")
    demo.add_argument("--full", action="store_true", help="MovieLens-sized synthetic data (about a minute)")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%H:%M:%S")
    if args.command == "run":
        pipeline.run(args.data, args.site, args.results, repo_url=args.repo_url, readme=args.readme)
        print(f"Done. Open {Path(args.site) / 'index.html'} in a browser.")
    else:
        out = Path(args.out)
        size = (dict(n_users=6040, n_items=3706, mean_ratings=165, span_days=1040) if args.full
                else dict(n_users=1500, n_items=800, mean_ratings=70))
        folder = write_synthetic(out / "data" / "ml-1m", **size)
        pipeline.run(site_dir=out / "site", results_dir=out / "results", dataset=data.load(folder),
                     note=SYNTHETIC_NOTE, source="synthetic")
        print(f"Done. Open {out / 'site' / 'index.html'} in a browser.")


if __name__ == "__main__":
    main()
