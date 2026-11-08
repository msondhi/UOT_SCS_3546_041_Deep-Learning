#!/usr/bin/env bash
# Download "140k Real and Fake Faces" from Kaggle into data/raw (about 4 GB).
# Needs a Kaggle API token at ~/.kaggle/access_token -- see docs/02_setup.md.
set -euo pipefail

DEST="${1:-data/raw}"
mkdir -p "$DEST"

# Use the project environment when there is one (Colab has none).
if [ -d .venv/bin ]; then
  export PATH="$PWD/.venv/bin:$PATH"
fi

# kaggle imports six but does not install it, so install both.
if ! python -c "import kaggle" >/dev/null 2>&1; then
  echo "Installing the Kaggle command-line tool..."
  python -m pip install --quiet kaggle six
fi

kaggle datasets download -d xhlulu/140k-real-and-fake-faces -p "$DEST" --unzip

echo "Done. Expected image folders:"
echo "  $DEST/real_vs_fake/real-vs-fake/{train,valid,test}/{real,fake}"
