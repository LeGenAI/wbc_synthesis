"""
Script 47: Run LODO training on boundary-aware V2 manifests.

This is a thin wrapper around Script 36 so the V2 branch can be evaluated
without changing the existing selective-synth training path.

Outputs:
  results/boundary_v2_lodo/{model}/summary.json
  results/boundary_v2_lodo/{model}/summary.md
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
SCRIPT36 = ROOT / "scripts" / "36_selective_synth_train.py"
MANIFEST_ROOT = ROOT / "results" / "boundary_selective_synth" / "subsets"
OUT_ROOT = ROOT / "results" / "boundary_v2_lodo"
OUT_ROOT.mkdir(parents=True, exist_ok=True)

DOMAINS = ["domain_a_pbc", "domain_b_raabin", "domain_c_mll23", "domain_e_amc"]
DOMAIN_SHORT = {
    "domain_a_pbc": "pbc",
    "domain_b_raabin": "raabin",
    "domain_c_mll23": "mll23",
    "domain_e_amc": "amc",
}
DOMAIN_LABELS = {
    "domain_a_pbc": "PBC (Spain)",
    "domain_b_raabin": "Raabin (Iran)",
    "domain_c_mll23": "MLL23 (Germany)",
    "domain_e_amc": "AMC (Korea)",
}
DEFAULT_TARGETS = ["domain_b_raabin", "domain_e_amc"]


def parse_args():
    parser = argparse.ArgumentParser(description="Run LODO with boundary-aware V2 manifests")
    parser.add_argument("--model", default="efficientnet_b0", choices=["efficientnet_b0", "vgg16"])
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--val_ratio", type=float, default=0.15)
    parser.add_argument("--train_fraction", type=float, default=1.0)
    parser.add_argument("--manifest", action="append", default=[],
                        help="explicit manifest path; may be repeated")
    parser.add_argument("--heldout_domain", action="append", choices=DOMAINS, default=[],
                        help="held-out domains to run; defaults to Raabin and AMC")
    parser.add_argument("--full_finetune", action="store_true")
    parser.add_argument("--include_heldout_synth", action="store_true")
    parser.add_argument("--dry_run", action="store_true")
    return parser.parse_args()


def find_default_manifests() -> list[Path]:
    preferred = ["B2_boundary_ranked.json", "B_AMC_domain_e_amc_boundary.json", "B_RAABIN_domain_b_raabin_boundary.json"]
    found = []
    for name in preferred:
        path = MANIFEST_ROOT / name
        if path.exists():
            found.append(path)
    if found:
        return found
    return sorted(MANIFEST_ROOT.glob("*.json"))


def read_subset_name(manifest_path: Path) -> str:
    data = json.loads(manifest_path.read_text())
    return data["subset_id"]


def run_single(args, heldout_domain: str, manifest_path: Path) -> dict:
    cmd = [
        sys.executable,
        str(SCRIPT36),
        "--heldout_domain", heldout_domain,
        "--subset_manifest", str(manifest_path),
        "--model", args.model,
        "--epochs", str(args.epochs),
        "--batch_size", str(args.batch_size),
        "--lr", str(args.lr),
        "--seed", str(args.seed),
        "--val_ratio", str(args.val_ratio),
        "--train_fraction", str(args.train_fraction),
    ]
    if args.full_finetune:
        cmd.append("--full_finetune")
    if args.include_heldout_synth:
        cmd.append("--include_heldout_synth")
    print(" ".join(cmd))
    if args.dry_run:
        return {
            "heldout_domain": heldout_domain,
            "manifest": str(manifest_path),
            "subset_id": read_subset_name(manifest_path),
            "status": "dry_run",
        }

    subprocess.run(cmd, check=True)
    subset_id = read_subset_name(manifest_path)
    report_path = ROOT / "results" / "selective_synth" / args.model / f"heldout_{DOMAIN_SHORT[heldout_domain]}" / subset_id / "report.json"
    report = json.loads(report_path.read_text())
    return {
        "heldout_domain": heldout_domain,
        "manifest": str(manifest_path),
        "subset_id": subset_id,
        "report_path": str(report_path),
        "macro_f1": report["test"]["macro_f1"],
        "acc": report["test"]["acc"],
        "n_synth_train": report["n_synth_train"],
    }


def write_summary(args, runs: list[dict]) -> None:
    model_dir = OUT_ROOT / args.model
    model_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "model": args.model,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "train_fraction": args.train_fraction,
        "runs": runs,
    }
    (model_dir / "summary.json").write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")

    lines = [
        "# Boundary V2 LODO Summary",
        "",
        f"- model: `{args.model}`",
        f"- epochs: `{args.epochs}`",
        f"- batch_size: `{args.batch_size}`",
        f"- train_fraction: `{args.train_fraction}`",
        "",
        "| Held-out | Subset | Macro-F1 | Acc | N synth train | Manifest |",
        "|---|---|---:|---:|---:|---|",
    ]
    for run in runs:
        if run.get("status") == "dry_run":
            lines.append(
                f"| {DOMAIN_LABELS[run['heldout_domain']]} | {run['subset_id']} | - | - | - | `{Path(run['manifest']).name}` |"
            )
        else:
            lines.append(
                f"| {DOMAIN_LABELS[run['heldout_domain']]} | {run['subset_id']} | "
                f"{run['macro_f1']:.4f} | {run['acc']:.4f} | {run['n_synth_train']} | `{Path(run['manifest']).name}` |"
            )
    (model_dir / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    args = parse_args()
    manifests = [Path(p) for p in args.manifest] if args.manifest else find_default_manifests()
    if not manifests:
        if args.dry_run:
            print(f"[dry_run] no manifests found under {MANIFEST_ROOT}")
            print("[dry_run] run Script 46 after V2 generation reports are available")
            write_summary(args, [])
            return
        raise RuntimeError(f"no manifests found under {MANIFEST_ROOT}")
    heldouts = args.heldout_domain or list(DEFAULT_TARGETS)

    runs = []
    for heldout in heldouts:
        for manifest_path in manifests:
            runs.append(run_single(args, heldout, manifest_path))

    write_summary(args, runs)
    print(f"[done] summary -> {OUT_ROOT / args.model / 'summary.md'}")


if __name__ == "__main__":
    main()
