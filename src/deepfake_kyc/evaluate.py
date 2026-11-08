"""Evaluate a trained model on clean, degraded, or out-of-distribution images.

    # Clean test set
    python -m deepfake_kyc.evaluate --checkpoint outputs/03_effnet_finetune/best.pt
    # Robustness: every degradation at severities 1-5
    python -m deepfake_kyc.evaluate --checkpoint <ckpt> --degradation all
    # Out-of-distribution folder containing real/ and fake/
    python -m deepfake_kyc.evaluate --checkpoint <ckpt> --data-dir data/ood --name ood

Writes a CSV with one row per condition next to the checkpoint (or to --output).
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

from .config import apply_overrides
from .data import build_loader
from .degradations import DEGRADATIONS
from .engine import predict
from .metrics import compute_metrics
from .models import load_checkpoint
from .utils import get_device


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--split", default="test", choices=["train", "val", "test"])
    parser.add_argument("--data-dir", help="Evaluate on this real/fake folder instead of the split")
    parser.add_argument(
        "--degradation", nargs="*", default=[], help=f"Any of {sorted(DEGRADATIONS)} or 'all'"
    )
    parser.add_argument("--severities", nargs="*", type=int, default=[1, 2, 3, 4, 5])
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--name", help="Label for the output file (default: split name)")
    parser.add_argument("--output", help="CSV path (default: <checkpoint dir>/eval_<name>.csv)")
    parser.add_argument("--set", nargs="*", default=[], metavar="KEY=VALUE", help="e.g. data.test_subset=2000")
    return parser.parse_args(argv)


def main(argv=None) -> list[dict]:
    args = parse_args(argv)
    device = get_device()
    model, cfg = load_checkpoint(args.checkpoint, device)
    cfg = apply_overrides(cfg, args.set)

    names = sorted(DEGRADATIONS) if "all" in args.degradation else args.degradation
    conditions = [("clean", 0, None)] + [
        (name, s, DEGRADATIONS[name](s)) for name in names for s in args.severities
    ]

    rows = []
    for name, severity, degradation in conditions:
        loader = build_loader(cfg, args.split, eval_degradation=degradation, root_override=args.data_dir)
        probs, labels, _ = predict(model, loader, device)
        m = compute_metrics(labels, probs, args.threshold)
        rows.append({"condition": name, "severity": severity, **{k: round(v, 5) for k, v in m.items()}})
        print(
            f"{name:<10} sev={severity}  acc={m['accuracy']:.4f}  auc={m['roc_auc']:.4f}  "
            f"FAR={m['far']:.4f}  FRR={m['frr']:.4f}"
        )

    label = args.name or args.split
    output = Path(args.output) if args.output else Path(args.checkpoint).parent / f"eval_{label}.csv"
    with open(output, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    print(f"Saved {output}")
    return rows


if __name__ == "__main__":
    main()
