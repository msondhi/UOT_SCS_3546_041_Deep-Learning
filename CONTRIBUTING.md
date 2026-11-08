# Contributing

## Git workflow

1. **Never commit directly to `main`.** Create a branch for each task:
   ```bash
   git checkout main && git pull
   git checkout -b feature/robust-training      # or fix/..., docs/..., exp/...
   ```
2. Make small, focused commits with clear messages:
   ```
   Add screen-capture degradation
   Fix label mapping for OOD folder
   ```
3. Before pushing, run:
   ```bash
   make lint && make test
   ```
4. Push and open a **pull request** on GitHub. Ask one teammate to review it.
5. Merge after approval, then delete the branch.

### Branch names
| Prefix | For |
|---|---|
| `feature/` | New code |
| `fix/` | Bug fixes |
| `exp/` | New experiment configs |
| `docs/` | Documentation and report |

## What never goes in git
- Datasets (`data/`)
- Model weights (`*.pt`, `outputs/`)
- `kaggle.json`, `access_token`, or any passwords or keys
- Any data from work (payments or law enforcement)

`.gitignore` already covers the first three. The last one is up to you.

## Code style
- Python 3.10+, formatted to `ruff` defaults (line length 120). `make lint` checks it.
- Clear names over clever code. Add a short docstring to each new function.
- Put settings in YAML configs, not hard-coded in Python.
- New functionality gets a test in `tests/`.

## Running experiments
- Create a new config file for each new experiment. Don't edit configs that have already been run.
- Log results in the table in [docs/05_experiments.md](docs/05_experiments.md).
- Share result files (CSV, PNG, `summary.json`) through the team drive. Don't commit `best.pt`.

## Getting help
Stuck for more than 30 minutes? Post in the team chat with:
1. What you ran
2. The full error message
3. What you've tried
