# 07 Explainability and demo

## Grad-CAM heatmaps

### What it shows
A heatmap over the face. **Red/yellow** areas pushed the model towards "fake" the most, and **blue** areas mattered little.

### How to run
```bash
make explain CKPT=outputs/03_effnet_finetune/best.pt
# or with options:
.venv/bin/python -m deepfake_kyc.explain --checkpoint <ckpt> --num-images 12 --seed 1
.venv/bin/python -m deepfake_kyc.explain --checkpoint <ckpt> --data-dir data/ood   # on OOD images
```
This saves `gradcam_<split>.png` next to the checkpoint: each row is the original image (with its true label) and the heatmap (with `P(fake)`).

### What to look for
| If the heatmap focuses on... | It suggests |
|---|---|
| Eyes, teeth, ears, hair edges, earrings | Real generator artifacts. Good. |
| Skin texture overall | Learned texture differences. Plausible. |
| Background or image borders | **Shortcut learning**: the model is using something other than the face. Investigate. |
| The same spot on every image | The model may rely on a dataset quirk |

Compare heatmaps for:
- correct vs. wrong predictions
- clean vs. degraded images (does attention shift when the image is blurred?)
- E3 vs. E6 (does robust training change what it looks at?)

### Caution
Grad-CAM is a rough guide, not proof. Heatmaps are low-resolution and can look convincing even when they're misleading. Say this in the report.

## Demo app (Gradio)

### Run it
```bash
make demo CKPT=outputs/06_effnet_robust/best.pt
```
Open the link it prints (usually http://127.0.0.1:7860). Upload a face photo and you'll see:
- the prediction: "AI-generated" vs. "Real" with probabilities
- the Grad-CAM heatmap

Add `--share` to get a temporary public link, which is useful for a remote presentation.

### Tips for the live demo
- Have a folder of test images ready (real, fake, blurred, screenshot) rather than relying on internet images.
- Show one where the model is **wrong** and explain why. That shows you understand its limits.
- Have screenshots ready as a backup in case the live demo fails.

Next: [08 Ethics and limitations](08_ethics_and_limitations.md)
