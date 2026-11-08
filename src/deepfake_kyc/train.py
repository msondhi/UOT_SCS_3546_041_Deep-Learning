"""Train one experiment.

    python -m deepfake_kyc.train --config configs/experiments/03_effnet_finetune.yaml
    python -m deepfake_kyc.train --config <file> --set data.train_subset=10000 train.epochs=2

Writes to ``<output_dir>/<experiment_name>/``: best.pt, config.yaml, history.csv, summary.json.
"""

from __future__ import annotations

import argparse
import csv
import time
from pathlib import Path

import torch
import yaml
from torch import nn
from torch.optim.lr_scheduler import LambdaLR

from .config import load_config
from .data import build_loader
from .engine import predict, train_one_epoch, warmup_cosine
from .metrics import compute_metrics
from .models import build_model, set_backbone_trainable, split_parameters
from .utils import get_device, save_json, seed_everything


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", required=True, help="Experiment YAML file")
    parser.add_argument("--set", nargs="*", default=[], metavar="KEY=VALUE", help="Override config values")
    return parser.parse_args(argv)


def main(argv=None) -> dict:
    args = parse_args(argv)
    cfg = load_config(args.config, args.set)
    seed_everything(cfg["seed"])
    device = get_device()
    out_dir = Path(cfg["output_dir"]) / cfg["experiment_name"]
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "config.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False))
    print(f"Experiment {cfg['experiment_name']} on {device} -> {out_dir}")

    train_loader = build_loader(cfg, "train", train=True)
    val_loader = build_loader(cfg, "val")
    print(f"Train images: {len(train_loader.dataset)}  Val images: {len(val_loader.dataset)}")

    mc, tc = cfg["model"], cfg["train"]
    model = build_model(mc["name"], pretrained=mc["pretrained"], dropout=mc["dropout"]).to(device)
    backbone_params, head_params = split_parameters(model)
    optimizer = torch.optim.AdamW(
        [
            {"params": backbone_params, "lr": tc["lr"] * tc["backbone_lr_mult"]},
            {"params": head_params, "lr": tc["lr"]},
        ],
        weight_decay=tc["weight_decay"],
    )
    steps_per_epoch = len(train_loader)
    scheduler = LambdaLR(
        optimizer, warmup_cosine(tc["warmup_epochs"] * steps_per_epoch, tc["epochs"] * steps_per_epoch)
    )
    criterion = nn.BCEWithLogitsLoss()

    history, best_val_loss, epochs_without_improvement = [], float("inf"), 0
    for epoch in range(1, tc["epochs"] + 1):
        frozen = epoch <= tc["freeze_backbone_epochs"]
        set_backbone_trainable(model, not frozen)
        start = time.time()
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, scheduler, device)
        probs, labels, val_loss = predict(model, val_loader, device, criterion)
        val = compute_metrics(labels, probs)
        row = {
            "epoch": epoch,
            "backbone_frozen": frozen,
            "train_loss": round(train_loss, 5),
            "val_loss": round(val_loss, 5),
            "val_accuracy": round(val["accuracy"], 5),
            "val_f1": round(val["f1"], 5),
            "val_roc_auc": round(val["roc_auc"], 5),
            "seconds": round(time.time() - start, 1),
        }
        history.append(row)
        print("  ".join(f"{k}={v}" for k, v in row.items()))

        if val_loss < best_val_loss:
            best_val_loss, epochs_without_improvement = val_loss, 0
            torch.save({"model_state": model.state_dict(), "config": cfg, "epoch": epoch}, out_dir / "best.pt")
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= tc["early_stopping_patience"]:
                print(f"Early stopping: no validation improvement for {epochs_without_improvement} epochs")
                break

    with open(out_dir / "history.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=history[0].keys())
        writer.writeheader()
        writer.writerows(history)
    best = min(history, key=lambda r: r["val_loss"])
    summary = {"experiment_name": cfg["experiment_name"], "best_epoch": best["epoch"], "best": best}
    save_json(summary, out_dir / "summary.json")
    print(f"Best epoch {best['epoch']}: val_loss={best['val_loss']} val_accuracy={best['val_accuracy']}")
    return summary


if __name__ == "__main__":
    main()
