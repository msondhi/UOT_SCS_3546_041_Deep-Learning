# Results (local training runs)

Results of the experiments trained locally on an Apple M5 (PyTorch, MPS GPU).
Same layout as `outputs/`, so notebooks can read them: `results/<experiment>/...`.
The trained models (`best.pt`) are **not** in git. All 9 are in the GitHub Release
[`models-v1`](https://github.com/msondhi/UOT_SCS_3546_041_Deep-Learning/releases/tag/models-v1)
(`<experiment>.pt`). Put each one in `outputs/<experiment>/best.pt`, for example:

```bash
mkdir -p outputs/03_effnet_finetune
gh release download models-v1 -R msondhi/UOT_SCS_3546_041_Deep-Learning -p 03_effnet_finetune.pt -O outputs/03_effnet_finetune/best.pt
```

All numbers are on the **validation** set (20,000 images). The test set hasn't been used yet.

| Exp | Model | Training | Best epoch | Val accuracy | Val ROC-AUC | Time |
|---|---|---|---|---|---|---|
| E1 | SimpleCNN | from scratch | 10 / 10 | 97.52% | 0.9974 | 105 min |
| E2 | EfficientNet-B0 | frozen backbone | 5 / 5 | 77.68% | 0.8606 | 34 min |
| E3 | EfficientNet-B0 | full fine-tune (**main model**) | 5 / 5 | 98.89% | 0.9996 | 126 min |
| E4 | EfficientNet-B0 | gradual unfreezing | 4 / 4 * | 98.21% | 0.9989 | 203 min * |
| E5 | ResNet-50 | full fine-tune | 5 / 5 | **99.17%** | **0.9997** | 247 min |
| E6 | EfficientNet-B0 | fine-tune + damaged-photo training | 5 / 5 | 98.50% | 0.9992 | 211 min |

\* E4 was stopped after 4 of 5 epochs. The Mac ran out of memory and the epochs were getting
slower. It was behind E3 at every epoch, so the conclusion wouldn't change. `history.csv` was
rebuilt from the run log, and there's no `summary.json`.

## E7: how much training data is needed (E3 settings, 5 epochs each)

| Training images | Val accuracy | Val ROC-AUC |
|---|---|---|
| 1,000 | 65.78% | 0.7200 |
| 10,000 | 86.73% | 0.9425 |
| 50,000 | 97.23% | 0.9969 |
| 100,000 (E3) | 98.89% | 0.9996 |

Caveat: with a fixed number of epochs, smaller sets also get far fewer update steps
(about 80 for 1k vs. 7,800 for 100k), so the small runs are partly undertrained.
The 1k run's time (716 min) was caused by the Mac stalling, not by the training itself.

## Robustness: E3 vs. E6

`03_effnet_finetune/eval_val_robustness.csv` and `06_effnet_robust/eval_val_robustness.csv`:
2,000 validation images (seed 42), 5 types of damage x 5 strengths, plus clean.
Columns include accuracy, ROC-AUC, FAR (fakes accepted) and FRR (real faces rejected).

- Average accuracy over the 25 damaged conditions: **E3 71.6% vs. E6 89.6%**
- Clean accuracy: 98.9% vs. 98.7%
- Caveat: E6 was trained on the same 5 types of damage it was tested on.

Chart: `figures/robustness_val.png` (made with `scripts/plot_robustness.py`).

## Grad-CAM

`03_effnet_finetune/gradcam_val.png` and `06_effnet_robust/gradcam_val.png`: 8 validation
images (4 real, 4 fake). For fakes, both models focus on the centre of the face
(nose, mouth, chin).

## Files in each experiment folder

- `config.yaml`: the exact settings used
- `history.csv`: loss and validation metrics for every epoch
- `summary.json`: the best epoch (lowest validation loss), which is the one saved as `best.pt`

`logs/` has the console output of each run, with progress bars removed: one line per epoch for
training runs, and one line per damage condition for the robustness runs.
