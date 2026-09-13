#!/usr/bin/env python3
"""Generate a small LaTeX result table from experiment JSON files."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def latex_escape(value: str) -> str:
    return value.replace("_", r"\_").replace("%", r"\%")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("metrics", nargs="+")
    parser.add_argument("--output", default="report/generated_results.tex")
    args = parser.parse_args()
    rows = []
    for path in args.metrics:
        with open(path, encoding="utf-8") as stream:
            result = json.load(stream)
        rows.append(
            "{} & {} & {} & {:.4f} & {:.4f} & {:.2f}".format(
                latex_escape(result["feature"]),
                latex_escape(result["model"]),
                result["samples"]["test"],
                result["test"]["accuracy"],
                result["test"]["macro_f1"],
                result["training_seconds"],
            )
        )
    destination = Path(args.output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(" \\\\\n".join(rows), encoding="utf-8")
    print(f"Generated {destination}")


if __name__ == "__main__":
    main()
