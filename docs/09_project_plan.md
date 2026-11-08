# 09 Project plan

## Timeline (3 weeks)

### Week 1: Foundations
| Day | Task | Done when |
|---|---|---|
| 1 | Everyone: `make setup`, `make test`, `make smoke` | All three pass on every laptop |
| 1-2 | Download data (`make data`); look through samples | Folder layout matches [03 Data](03_data.md) |
| 2-3 | Quick run of E3 on a 5k subset | Code works on real data |
| 3-5 | Full E1 (baseline) and E2 (frozen) | Results in the results log |
| 5 | Start collecting the OOD test set | `data/ood/real` and `data/ood/fake` filled |

### Week 2: Main experiments
| Day | Task | Done when |
|---|---|---|
| 6-7 | E3 (main model) and E4 (gradual) | Results logged |
| 7-8 | E5 (ResNet-50) and E7 (data-size study) | Results logged |
| 8-9 | E6 (robust training) | Results logged |
| 9-10 | Robustness eval for E3 and E6; OOD eval | `eval_test.csv`, `eval_ood.csv` for both |

### Week 3: Analysis and delivery
| Day | Task | Done when |
|---|---|---|
| 11-12 | Grad-CAM and error analysis | Heatmap figures and 3-5 written observations |
| 12 | Demo app working with the best model | Live demo rehearsed |
| 12-14 | Charts, report, slides | Draft shared with the team |
| 14-15 | Review, rehearse, submit | Submitted |

## Roles
| Person | Owns | Also helps with |
|---|---|---|
| [Name]: law enforcement | Robustness testing, Grad-CAM and error analysis, ethics section | Report editing |
| [Name]: payments | Data pipeline, E1-E5 training, results tables | KYC/onboarding context, cost of errors |
| [Name]: payments | E6 robust training, OOD set, demo app | Threshold/FAR-FRR discussion |

Everyone: review each other's pull requests (see [CONTRIBUTING](../CONTRIBUTING.md)).

## Weekly check-ins
Hold a short meeting at least twice a week covering:
1. What did I finish?
2. What's next?
3. Am I blocked?

Keep notes in a shared doc.

## Report outline
1. **Introduction:** synthetic identities in KYC; why it matters
2. **Related work:** deepfake detection, transfer learning, robustness (3-6 references)
3. **Data:** 140k faces dataset, splits, OOD set
4. **Methods:** baseline CNN, transfer learning strategies, training setup, augmentation, degradations, Grad-CAM
5. **Experiments and results:** clean results (E1-E6), data-size curve, robustness charts, OOD results
6. **Explainability and error analysis:** heatmaps and observations
7. **Discussion:** FAR/FRR trade-off, real-world deployment, payments and law enforcement perspective
8. **Ethics and limitations**
9. **Conclusion and future work:** video deepfakes, liveness, diffusion-generated training data
10. **Contributions:** who did what

## Presentation outline (about 10 minutes)
1. Hook: "Which of these faces is real?" (show 4, all fake)
2. The problem: synthetic identities in KYC
3. Approach in one diagram
4. Results: baseline vs. transfer learning
5. Key figure: robustness before and after robust training
6. Grad-CAM examples
7. Live demo
8. Limitations, ethics, and lessons learned

Next: [10 Codebase](10_codebase.md)
