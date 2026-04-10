#!/usr/bin/env python3
"""
Scaffold for the mainline supplementary stage 04.

Purpose:
- score synthetic samples with preservation and utility proxies
- build reusable policy manifests
- separate generation quality from downstream utility evidence
"""

import argparse


def main() -> None:
    parser = argparse.ArgumentParser(description="Scaffold for mainline synthetic-pool scoring.")
    parser.add_argument("--config", type=str, default=None, help="Path to future scoring config.")
    parser.parse_args()
    raise SystemExit("Scaffold only: implement the new mainline scoring pipeline here.")


if __name__ == "__main__":
    main()
