# Notebooks

Use notebooks for **exploring and plotting**, and keep training code in `src/deepfake_kyc/`.

The notebooks are made for **Kaggle** (see "Running the notebooks on Kaggle" below). How they share files
through the attached dataset and `outputs/`: [docs/2_notebook_flow.html](../docs/2_notebook_flow.html).

| Notebook | Purpose |
|---|---|
| `01_explore_data.ipynb` | View sample real/fake faces, check image sizes and class balance |
| `02_degradations_gallery.ipynb` | Show one face under every degradation at severities 1-5 (good report figure) |
| `03_results_charts.ipynb` | Load `outputs/*/eval_test.csv` and draw the robustness charts and results tables |
| `04_error_analysis.ipynb` | Browse false positives and false negatives, with Grad-CAM |
| `05_next_architecture_effnet_b0.ipynb` | Train any experiment(s) E1-E6 (choose `EXPERIMENTS` in Step 2; each is saved in its own `outputs/<experiment>/` folder and never overwritten), check the attached data, check each model, opt-in quick or full training, loss charts, and a once-only final test score |

## Running the notebooks on Kaggle

Step 1 of every notebook clones the project code into `/kaggle/working/` and finds the attached dataset under
`/kaggle/input/`. Setup for each notebook:

1. **Open it:** on kaggle.com → **Create → New Notebook → File → Import Notebook**, and paste the notebook's
   GitHub link, e.g. `https://github.com/msondhi/UOT_SCS_3546_041_Deep-Learning/blob/main/notebooks/01_explore_data.ipynb`.
2. **Settings → Internet → On** (Step 1 downloads the project code from GitHub; Kaggle asks you to verify your phone once).
3. **Add Input → Datasets → "140k Real and Fake Faces"** (by xhlulu). The images are attached read-only under
   `/kaggle/input/`; nothing is downloaded or unzipped.
4. **Settings → Accelerator → GPU T4 x2** for notebooks 04 and 05 (and Step 1b of notebook 03).

How the notebooks share results on Kaggle:

| What | Where it lives on Kaggle |
|---|---|
| Images | `/kaggle/input/.../real_vs_fake/real-vs-fake/` (the attached dataset) |
| Project code | cloned into `/kaggle/working/UOT_SCS_3546_041_Deep-Learning/` by Step 1 |
| Trained models and scores | `/kaggle/working/UOT_SCS_3546_041_Deep-Learning/outputs/<experiment>/` |

`/kaggle/working/` is kept only when you click **Save Version → Save & Run All (Commit)**: Kaggle runs the
notebook again from the top and stores its files. So in notebook 05 choose `EXPERIMENTS` and set `RUN_TRAINING = True` before saving. Train all the experiments
you need in that one run: a new version keeps only its own files, and notebooks 03/04 attach one version.
Then, in notebooks 03 and 04, **Add Input → Your Work →** your saved notebook 05: its `outputs/` folder appears
read-only under `/kaggle/input/`, and the notebooks find the models there. New score files from notebook 03 are
written to its own `/kaggle/working/.../outputs/`, because inputs are read-only.

## Using the project code in a notebook
Select the `.venv` kernel in VS Code or Jupyter, then:

```python
from deepfake_kyc.config import load_config
from deepfake_kyc.data import make_dataset
from deepfake_kyc.degradations import DEGRADATIONS

cfg = load_config("../configs/base.yaml")
ds = make_dataset("../" + cfg["data"]["root"] + "/train")
img, label = ds[0]   # PIL image, label 1 = fake
DEGRADATIONS["blur"](5)(img)
```

## Rules
- Clear outputs before committing (**Kernel → Restart & Clear Output**) to keep git diffs small.
- Name notebooks with a number prefix so they sort in order.
