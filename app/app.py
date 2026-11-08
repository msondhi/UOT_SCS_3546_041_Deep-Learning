"""Gradio demo: upload a face photo, get P(fake) and a Grad-CAM heatmap.

    python app/app.py --checkpoint outputs/03_effnet_finetune/best.pt
"""

import argparse

import gradio as gr

from deepfake_kyc.explain import Explainer
from deepfake_kyc.models import load_checkpoint
from deepfake_kyc.utils import get_device


def build_app(checkpoint: str) -> gr.Interface:
    device = get_device()
    model, cfg = load_checkpoint(checkpoint, device)
    explainer = Explainer(model, cfg["data"]["image_size"], device)

    def predict(img):
        prob_fake, overlay = explainer(img)
        return {"AI-generated": prob_fake, "Real": 1 - prob_fake}, overlay

    return gr.Interface(
        fn=predict,
        inputs=gr.Image(type="pil", label="Face photo"),
        outputs=[gr.Label(label="Prediction"), gr.Image(label="What the model looked at (Grad-CAM)")],
        title="Deepfake KYC Face Check",
        description="Research prototype. Flags faces that may be AI-generated, for a human reviewer to check.",
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--share", action="store_true", help="Create a temporary public link")
    args = parser.parse_args()
    build_app(args.checkpoint).launch(share=args.share)
