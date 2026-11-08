# 02 Setup

There are two ways to run the project: on your own **Mac** (Apple Silicon) or on **Google Colab** (free, in the browser). Both use the same code.

## Option A: Mac (Apple Silicon, M1-M5)

### 1. Prerequisites
- Python 3.10 or newer. Check with `python3 --version`. If it's older, install a newer one with `brew install python@3.12`.
- About 10 GB of free disk space (dataset plus outputs)

### 2. Install
From the project folder:

```bash
make setup
```

This creates a virtual environment in `.venv/` and installs everything. To do it by hand:

```bash
python3.12 -m venv .venv
.venv/bin/pip install -e ".[dev,demo]"
```

### 3. Check it works
```bash
make test     # unit tests, about 10 seconds
make smoke    # trains for 1 epoch on a tiny fake dataset
```

The smoke run should print `on mps`, which means it's using the Mac's GPU.

### 4. Point VS Code at the environment
Press `Cmd+Shift+P`, choose **Python: Select Interpreter**, and pick `.venv/bin/python`. Otherwise VS Code reports the libraries as "not installed".

## Option B: Google Colab

1. Go to https://colab.research.google.com and create a new notebook.
2. Choose **Runtime → Change runtime type → T4 GPU**.
3. Run:

```python
!git clone <your-repo-url> deepfake-kyc
%cd deepfake-kyc
!pip install -q -e ".[dev]"
```

4. Set up your Kaggle key (see below). On Colab, add it as a secret named `KAGGLE_API_TOKEN` and load it into the environment. Then run `!bash scripts/download_data.sh`.
5. Train with `!python -m deepfake_kyc.train --config configs/experiments/03_effnet_finetune.yaml`.

Colab deletes files when the session ends. Copy `outputs/` to Google Drive before you close it.

## Kaggle access (needed to download the data)

1. Create a free account at https://www.kaggle.com.
2. Go to https://www.kaggle.com/settings/api and create an API token. Kaggle shows you a key to copy. No file is downloaded.
3. Click copy on the Kaggle page, then save the key straight from your clipboard:
   ```bash
   mkdir -p ~/.kaggle && pbpaste > ~/.kaggle/access_token && chmod 600 ~/.kaggle/access_token
   ```
4. Run `make data`.

Never commit your key to git or paste it into a chat. `access_token` and `kaggle.json` are both listed in `.gitignore`.

## Troubleshooting

| Problem | Fix |
|---|---|
| `NotImplementedError ... MPS` | Run with `PYTORCH_ENABLE_MPS_FALLBACK=1 make train ...` |
| Out of memory | Lower the batch size: `--set data.batch_size=16` |
| Training is slow while you're writing code | Use a subset: `--set data.train_subset=10000` |
| `401` or `Unauthorized` from Kaggle | The key didn't save correctly. Copy it again and repeat step 3 |
| `403` from Kaggle | Open the dataset page while signed in and accept any terms it shows |
| `FileNotFoundError` for the data | Check `data.root` in `configs/base.yaml` matches the unzipped folder |
| DataLoader worker errors | Try `--set data.num_workers=0` |

Next: [03 Data](03_data.md)
