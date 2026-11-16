# 06 Evaluation

## The confusion matrix
"Positive" means **fake** (the thing we want to catch).

| | Model says **fake** | Model says **real** |
|---|---|---|
| **Actually fake** | TP: caught it | FN: **fraud got through** |
| **Actually real** | FP: **real customer blocked** | TN: correct |

## Metrics
| Metric | Plain-words meaning |
|---|---|
| **Accuracy** | Share of all images classified correctly |
| **Precision** | Of the images flagged as fake, how many really were fake |
| **Recall** | Of all the fakes, how many we caught |
| **F1** | A single score balancing precision and recall |
| **ROC-AUC** | How well the model ranks fakes above reals across *all* thresholds (1.0 = perfect, 0.5 = guessing) |
| **FAR**: False Acceptance Rate | Share of **fake** faces accepted as real = FN / (TP + FN) |
| **FRR**: False Rejection Rate | Share of **real** customers rejected = FP / (FP + TN) |

FAR and FRR are the standard metrics in identity verification and biometrics, and they matter most for KYC:
- **High FAR** means fraudsters get accounts, which leads to fraud losses and mule accounts.
- **High FRR** means genuine customers are turned away, which loses business and treats people unfairly.

## The threshold
The model outputs `P(fake)`. With threshold 0.5, anything at or above 0.5 is flagged.
- **Lower threshold:** catches more fakes (lower FAR) but blocks more real people (higher FRR)
- **Higher threshold:** the opposite

To test a different threshold:
```bash
python -m deepfake_kyc.evaluate --checkpoint <ckpt> --threshold 0.3
```
In the report, discuss what threshold a bank might choose and why. Your payments teammates can help put rough costs on each kind of error.

## Robustness evaluation
`make robustness` runs the test set:
1. **Clean** (no changes)
2. With each degradation (`jpeg`, `blur`, `downscale`, `noise`, `screen`) at severities 1-5

That's 26 conditions, one row each in `eval_test.csv`.

**How to present it:** one line chart per degradation, with severity on the x-axis and accuracy (or FAR) on the y-axis, and two lines: E3 (standard) and E6 (robust-trained). This is the key figure of the project.

## Out-of-distribution evaluation
Run the model on fakes from a generator it never saw (see [03 Data](03_data.md)), and compare with clean test results. A large drop means the model learned StyleGAN-specific artifacts, not general "fakeness".

## Error analysis (don't skip this)
After the numbers, look at the images:
- Open 20 false negatives (fakes that got through) and 20 false positives (real faces flagged).
- Look for patterns: glasses, hats, unusual lighting, age, skin tone, background.
- Run Grad-CAM on them (see [07](07_explainability_and_demo.md)).
- Write down 3-5 observations for the report.

## Reporting checklist
- [ ] Clean results table for E1-E6
- [ ] Robustness line charts (E3 vs. E6)
- [ ] OOD results
- [ ] Data-size curve (E7)
- [ ] Confusion matrix for the main model
- [ ] Threshold / FAR-FRR trade-off discussion
- [ ] Error analysis with examples

Next: [07 Explainability and demo](07_explainability_and_demo.md)
