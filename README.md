# Deepfake KYC: Detecting AI-Generated Faces in Identity Verification

University of Toronto deep learning final project.

Banks check new customers by matching a selfie against an ID ("Know Your Customer", or KYC). Fraudsters now use AI-generated faces of people who don't exist to open accounts for credit fraud and money laundering. This project trains a deep learning model to tell **real** faces from **AI-generated** ones, then tests how well it holds up with real-world photos: blurry, compressed, screenshots, and faces from AI tools it has never seen.

## Quick start

```bash
make setup     # create .venv and install everything (once)
make smoke     # 1-minute check that the code runs, using a tiny fake dataset
make data      # download the real dataset from Kaggle (~4 GB; needs a Kaggle token)
make train CONFIG=configs/experiments/03_effnet_finetune.yaml
make evaluate   CKPT=outputs/03_effnet_finetune/best.pt
make robustness CKPT=outputs/03_effnet_finetune/best.pt
make explain    CKPT=outputs/03_effnet_finetune/best.pt
make demo       CKPT=outputs/03_effnet_finetune/best.pt
```

New to the project? Read the docs in order, starting with [docs/01_overview.md](docs/01_overview.md).

## Documentation

| Doc | What's in it |
|---|---|
| [01 Overview](docs/01_overview.md) | The problem, goals, scope, and what counts as success |
| [02 Setup](docs/02_setup.md) | Installing on a Mac or Google Colab, Kaggle access |
| [03 Data](docs/03_data.md) | The dataset, folder layout, labels, and data rules |
| [04 Methodology](docs/04_methodology.md) | The deep learning techniques, explained in plain words |
| [05 Experiments](docs/05_experiments.md) | Every experiment, the command to run it, and where results go |
| [06 Evaluation](docs/06_evaluation.md) | Metrics (including FAR/FRR) and how robustness is tested |
| [07 Explainability and demo](docs/07_explainability_and_demo.md) | Grad-CAM heatmaps and the Gradio app |
| [08 Ethics and limitations](docs/08_ethics_and_limitations.md) | Risks, bias, and what this model should not be used for |
| [09 Project plan](docs/09_project_plan.md) | 3-week timeline, roles, and report outline |
| [10 Codebase](docs/10_codebase.md) | How the code is organised and how to extend it |
| [CONTRIBUTING](CONTRIBUTING.md) | Git workflow, code style, and tests |
| [Proposal](docs/proposal/Deepfake_KYC_Proposal.docx) | The submitted project proposal |

## Repository layout

```
deepfake-kyc/
├── README.md
├── CONTRIBUTING.md
├── Makefile                  # shortcuts for every common task
├── pyproject.toml            # package definition and dependencies
├── requirements.txt
├── configs/
│   ├── base.yaml             # shared defaults
│   └── experiments/          # one file per experiment (E1-E6)
├── src/deepfake_kyc/         # the Python package
│   ├── config.py             # load YAML configs + overrides
│   ├── data.py               # datasets, transforms, loaders
│   ├── degradations.py       # JPEG, blur, downscale, noise, screen-capture effects
│   ├── models.py             # baseline CNN + pretrained models (timm)
│   ├── engine.py             # training and prediction loops
│   ├── metrics.py            # accuracy, F1, ROC-AUC, FAR, FRR
│   ├── train.py              # CLI: train one experiment
│   ├── evaluate.py           # CLI: clean, robustness, and out-of-distribution tests
│   ├── explain.py            # CLI: Grad-CAM heatmaps
│   └── dummy_data.py         # tiny synthetic dataset for tests
├── app/app.py                # Gradio demo
├── scripts/                  # data download and helper scripts
├── tests/                    # pytest unit and end-to-end tests
├── notebooks/                # exploration and results notebooks
├── docs/                     # project documentation
├── data/                     # datasets (git-ignored)
└── outputs/                  # checkpoints, metrics, figures (git-ignored)
```

## Team

| Name | Role |
|---|---|
| [Name] | Law enforcement context, robustness testing, explainability, ethics |
| [Name] | Data pipeline and model training |
| [Name] | Robust training, demo app, payments context |
