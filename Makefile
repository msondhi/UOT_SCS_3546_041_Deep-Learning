PY      ?= .venv/bin/python
CONFIG  ?= configs/experiments/03_effnet_finetune.yaml
CKPT    ?= outputs/03_effnet_finetune/best.pt

.PHONY: setup data dummy smoke train evaluate robustness explain demo test lint

setup:          ## Create virtual env and install everything
	python3 -m venv .venv
	$(PY) -m pip install --upgrade pip
	$(PY) -m pip install -e ".[dev,demo]"

data:           ## Download the Kaggle dataset into data/raw
	bash scripts/download_data.sh

dummy:          ## Create a tiny fake dataset for quick code checks
	$(PY) scripts/make_dummy_data.py --out data/dummy

smoke: dummy    ## End-to-end check on dummy data (about 1 minute)
	$(PY) -m deepfake_kyc.train --config configs/experiments/01_baseline_cnn.yaml \
		--set data.root=data/dummy data.image_size=64 data.num_workers=0 train.epochs=1 experiment_name=smoke

train:          ## Train one experiment: make train CONFIG=configs/experiments/xx.yaml
	$(PY) -m deepfake_kyc.train --config $(CONFIG)

evaluate:       ## Clean test-set evaluation: make evaluate CKPT=outputs/<exp>/best.pt
	$(PY) -m deepfake_kyc.evaluate --checkpoint $(CKPT)

robustness:     ## Test under every degradation at severities 1-5
	$(PY) -m deepfake_kyc.evaluate --checkpoint $(CKPT) --degradation all

explain:        ## Grad-CAM heatmaps for sample test images
	$(PY) -m deepfake_kyc.explain --checkpoint $(CKPT)

demo:           ## Launch the Gradio demo app
	$(PY) app/app.py --checkpoint $(CKPT)

test:           ## Run unit tests
	$(PY) -m pytest -q

lint:           ## Check code style
	$(PY) -m ruff check src tests app scripts
