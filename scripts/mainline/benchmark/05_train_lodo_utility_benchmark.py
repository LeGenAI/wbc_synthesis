#!/usr/bin/env python3
"""
Scaffold for the mainline supplementary stage 05.

Purpose:
- run the leakage-safe utility benchmark
- compare real-only and policy-conditioned synthetic pools
- generate the canonical downstream evidence for the paper
"""

import argparse


def main() -> None:
    parser = argparse.ArgumentParser(description="Scaffold for mainline LODO utility benchmark.")
    parser.add_argument("--config", type=str, default=None, help="Path to future benchmark config.")
    parser.parse_args()
    raise SystemExit("Scaffold only: implement the new mainline LODO benchmark here.")


if __name__ == "__main__":
    main()
