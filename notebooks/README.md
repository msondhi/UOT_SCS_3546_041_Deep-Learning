# Notebooks

Use notebooks for **exploring and plotting**, and keep training code in `src/deepfake_kyc/`.

Suggested notebooks (create these as you go):

| Notebook | Purpose |
|---|---|
| `01_explore_data.ipynb` | View sample real/fake faces, check image sizes and class balance |
| `02_degradations_gallery.ipynb` | Show one face under every degradation at severities 1-5 (good report figure) |
| `03_results_charts.ipynb` | Load `outputs/*/eval_test.csv` and draw the robustness charts and results tables |
| `04_error_analysis.ipynb` | Browse false positives and false negatives, with Grad-CAM |

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
