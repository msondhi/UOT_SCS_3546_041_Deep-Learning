"""Grad-CAM heatmaps: which parts of the face pushed the model towards "fake".

    python -m deepfake_kyc.explain --checkpoint outputs/03_effnet_finetune/best.pt --num-images 8
"""

from __future__ import annotations

import argparse
import random
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from PIL import Image  # noqa: E402
from pytorch_grad_cam import GradCAM  # noqa: E402
from pytorch_grad_cam.utils.image import show_cam_on_image  # noqa: E402
from pytorch_grad_cam.utils.model_targets import BinaryClassifierOutputTarget  # noqa: E402

from .data import SPLIT_DIRS, build_transform, make_dataset  # noqa: E402
from .models import gradcam_target_layer, load_checkpoint  # noqa: E402
from .utils import get_device  # noqa: E402


class Explainer:
    """Wraps a model for single-image prediction plus Grad-CAM overlay."""

    def __init__(self, model, image_size: int, device):
        self.model, self.image_size, self.device = model, image_size, device
        self.transform = build_transform(image_size)
        self.cam = GradCAM(model=model, target_layers=[gradcam_target_layer(model)])

    def __call__(self, img: Image.Image) -> tuple[float, np.ndarray]:
        """Return (P(fake), RGB heatmap overlay as uint8 array)."""
        img = img.convert("RGB")
        x = self.transform(img).unsqueeze(0).to(self.device)
        with torch.no_grad():
            prob = torch.sigmoid(self.model(x)).item()
        heat = self.cam(input_tensor=x, targets=[BinaryClassifierOutputTarget(1)])[0]
        rgb = np.asarray(img.resize((self.image_size, self.image_size)), dtype=np.float32) / 255.0
        return prob, show_cam_on_image(rgb, heat, use_rgb=True)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--split", default="test", choices=list(SPLIT_DIRS))
    parser.add_argument("--data-dir", help="Use this real/fake folder instead of the split")
    parser.add_argument("--num-images", type=int, default=8, help="Half real, half fake")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", help="PNG path (default: <checkpoint dir>/gradcam_<split>.png)")
    return parser.parse_args(argv)


def main(argv=None) -> Path:
    args = parse_args(argv)
    device = get_device()
    model, cfg = load_checkpoint(args.checkpoint, device)
    root = Path(args.data_dir) if args.data_dir else Path(cfg["data"]["root"]) / SPLIT_DIRS[args.split]
    ds = make_dataset(root)
    fake_idx = ds.class_to_idx["fake"]

    rng = random.Random(args.seed)
    fakes = [s for s in ds.samples if s[1] == fake_idx]
    reals = [s for s in ds.samples if s[1] != fake_idx]
    half = args.num_images // 2
    chosen = rng.sample(reals, min(half, len(reals))) + rng.sample(fakes, min(half, len(fakes)))

    explainer = Explainer(model, cfg["data"]["image_size"], device)
    fig, axes = plt.subplots(len(chosen), 2, figsize=(6, 3 * len(chosen)), squeeze=False)
    for row, (path, target) in zip(axes, chosen, strict=True):
        img = Image.open(path)
        prob, overlay = explainer(img)
        truth = "fake" if target == fake_idx else "real"
        row[0].imshow(img.convert("RGB"))
        row[0].set_title(f"true: {truth}")
        row[1].imshow(overlay)
        row[1].set_title(f"P(fake) = {prob:.2f}")
        for ax in row:
            ax.axis("off")
    fig.tight_layout()

    output = Path(args.output) if args.output else Path(args.checkpoint).parent / f"gradcam_{args.split}.png"
    fig.savefig(output, dpi=120)
    print(f"Saved {output}")
    return output


if __name__ == "__main__":
    main()
