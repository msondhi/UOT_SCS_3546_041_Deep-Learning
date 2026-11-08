# 05 Experiments

Every experiment is a YAML file in `configs/experiments/`. Each file starts from `configs/base.yaml` and lists only what changes, so it's easy to see exactly what differs between experiments.

## Experiment plan

| ID | Config | Question it answers | Compare with |
|---|---|---|---|
| E1 | `01_baseline_cnn.yaml` | How good is a CNN trained from scratch? | - |
| E2 | `02_effnet_frozen.yaml` | Are pretrained features good enough on their own? | E1 |
| E3 | `03_effnet_finetune.yaml` | Does full fine-tuning help? (**main model**) | E2 |
| E4 | `04_effnet_gradual.yaml` | Does gradual unfreezing help? | E3 |
| E5 | `05_resnet50_finetune.yaml` | Does the architecture matter? | E3 |
| E6 | `06_effnet_robust.yaml` | Does degradation-aware training improve robustness? | E3 |
| E7 | E3 with `--set` subsets | How much data do we need? | E3 |

## How to run

```bash
# Train (E1-E6)
make train CONFIG=configs/experiments/03_effnet_finetune.yaml

# Clean test-set results
make evaluate CKPT=outputs/03_effnet_finetune/best.pt

# Robustness: all 5 degradations x 5 severities (run for E3 and E6)
make robustness CKPT=outputs/03_effnet_finetune/best.pt

# Out-of-distribution generator test
.venv/bin/python -m deepfake_kyc.evaluate --checkpoint outputs/03_effnet_finetune/best.pt --data-dir data/ood --name ood
```

### E7: data-size study
Run E3 three times on smaller training sets, giving each run its own name:

```bash
for n in 1000 10000 50000; do
  .venv/bin/python -m deepfake_kyc.train --config configs/experiments/03_effnet_finetune.yaml \
    --set data.train_subset=$n experiment_name=07_effnet_n$n
done
```

### Quick development runs
Add `--set data.train_subset=5000 data.val_subset=1000 train.epochs=1` to any command to check things work in a few minutes.

## What each run saves

In `outputs/<experiment_name>/`:

| File | Contents |
|---|---|
| `config.yaml` | The exact settings used, so the run can be reproduced |
| `best.pt` | Model weights from the best epoch |
| `history.csv` | Per-epoch train/val loss, accuracy, F1, ROC-AUC, time |
| `summary.json` | Best epoch and its metrics |
| `eval_test.csv` | Test results; one row per condition if run with `--degradation` |
| `eval_ood.csv` | Out-of-distribution results |
| `gradcam_test.png` | Grad-CAM heatmaps |

## Estimated run times (Apple M5, full dataset)

These are estimates. Record real times in the results table.

| Experiment | Per epoch | Total |
|---|---|---|
| E1 SimpleCNN | ~5 min | ~30-50 min |
| E2 frozen EfficientNet | ~4 min | ~20 min |
| E3/E4/E6 EfficientNet fine-tune | ~8-12 min | ~45-60 min |
| E5 ResNet-50 | ~15-20 min | ~1.5 h |
| Robustness eval (26 conditions) | - | ~20-30 min |

Split the runs across team members' laptops and Colab so they happen in parallel.

## Results log (fill in as you go)

### Clean test set
| ID | Model | Strategy | Accuracy | F1 | ROC-AUC | FAR | FRR | Train time |
|---|---|---|---|---|---|---|---|---|
| E1 | SimpleCNN | from scratch | | | | | | |
| E2 | EfficientNet-B0 | frozen | | | | | | |
| E3 | EfficientNet-B0 | fine-tune | | | | | | |
| E4 | EfficientNet-B0 | gradual | | | | | | |
| E5 | ResNet-50 | fine-tune | | | | | | |
| E6 | EfficientNet-B0 | fine-tune + robust aug | | | | | | |

### Robustness (accuracy, E3 vs. E6)
| Degradation | Sev 1 | Sev 3 | Sev 5 |
|---|---|---|---|
| jpeg | / | / | / |
| blur | / | / | / |
| downscale | / | / | / |
| noise | / | / | / |
| screen | / | / | / |

### Generalization
| Model | Clean test (StyleGAN) | OOD generator |
|---|---|---|
| E3 | | |
| E6 | | |

## Rules for fair experiments
- Use the same seed (42) and the same data for every run.
- Change **one thing at a time** between compared experiments.
- Make decisions on **validation** results, and report **test** results.
- Never edit a config after running it. Create a new one instead.
- Note anything unusual (crashes, reruns) in the team notes.

Next: [06 Evaluation](06_evaluation.md)
