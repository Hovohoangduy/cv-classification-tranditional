#!/usr/bin/env python3
"""Run one or more JSON experiment configurations sequentially."""

from __future__ import annotations

import argparse
import subprocess
import sys


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("configs", nargs="+")
    args = parser.parse_args()
    for config in args.configs:
        print(f"Running {config}")
        subprocess.run([sys.executable, "-m", "scripts.train", "--config", config], check=True)


if __name__ == "__main__":
    main()
