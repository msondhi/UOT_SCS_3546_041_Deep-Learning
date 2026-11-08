# 04 Methodology

This page explains each deep learning technique in the project in plain words, and where it lives in the code.

## 1. The task: binary image classification
Input: a face image. Output: one number, `P(fake)`. The model produces a raw score (a **logit**), and a **sigmoid** function turns it into a probability between 0 and 1.

## 2. Convolutional neural networks (CNNs)
A CNN looks at an image through small sliding filters. Early layers detect edges and colours, middle layers detect textures and shapes, and deep layers detect whole parts such as eyes or hair. That makes CNNs good at spotting the small texture artifacts that AI generators leave behind.

**Baseline (E1):** `SimpleCNN` in `models.py` has four blocks of *convolution → batch normalization → ReLU → max-pooling*, then global average pooling and a single output. It's trained **from scratch**, starting with random weights.

## 3. Transfer learning (the core technique)
Training a big network from scratch needs millions of images. Instead, we start from a model that has already learned to "see" on ImageNet (1.2 million everyday photos) and adapt it to our task.

- **EfficientNet-B0:** small, fast and accurate. Our main model.
- **ResNet-50:** larger and well known. Used for comparison.
- Both are loaded with the `timm` library. Their original 1000-class output layer is replaced with a new 1-output layer (the **classifier head**).

### Three ways to fine-tune
| Strategy | What's trained | Experiment |
|---|---|---|
| Frozen backbone | Only the new head. The pretrained layers stay fixed. | E2 |
| Full fine-tuning | Everything. The backbone learns 10× slower than the head. | E3 |
| Gradual unfreezing | Head only for the first epoch, then everything | E4 |

The backbone gets a lower learning rate (`backbone_lr_mult: 0.1`) so we gently adjust what it already knows instead of overwriting it.

## 4. Training setup
| Component | Choice | Plain-words reason |
|---|---|---|
| Loss | Binary cross-entropy (`BCEWithLogitsLoss`) | The standard loss for yes/no predictions |
| Optimizer | AdamW | Adam with correctly applied weight decay; a reliable default |
| LR schedule | Linear warm-up, then cosine decay | Start gently, learn fast, then settle down |
| Weight decay | 1e-4 | Keeps weights small, which reduces overfitting |
| Dropout | 0.2 before the head | Randomly switches off features so the model can't rely on just one |
| Batch norm | Built into the backbones | Keeps training stable |
| Early stopping | Stop after 2 epochs without validation improvement | Don't keep training once it stops helping |
| Checkpointing | Keep the epoch with the lowest validation loss | We report the best model, not the last |

## 5. Data augmentation
Small random changes to training images so the model learns the face, not accidental details of each photo.

- **Standard** (always on): random crop and resize, horizontal flip, small brightness/contrast/colour changes
- **Degradation-aware** (E6 only): with 50% probability, also apply one random real-world degradation (next section) at a random severity

## 6. Robustness: real-world degradations
Real KYC photos are not clean lab images. `degradations.py` simulates five common problems, each at severity 1 (mild) to 5 (severe):

| Name | Simulates | Parameter (severity 1 → 5) |
|---|---|---|
| `jpeg` | Compression by messaging apps and upload forms | Quality 90 → 10 |
| `blur` | Out-of-focus or shaky camera | Blur radius 0.5 → 4 px |
| `downscale` | Low-resolution camera or upload | Scale 0.75 → 0.15 |
| `noise` | Poor lighting and sensor noise | Noise std 5 → 45 |
| `screen` | Photographing a screen (a common fraud trick) | Moiré strength 0.03 → 0.15 |

We use them twice: to **test** robustness (evaluate on degraded test images) and to **improve** it (augmentation in E6).

## 7. Generalization to unseen generators
A detector trained on StyleGAN faces may learn "StyleGAN's fingerprint" rather than "what fake looks like". Testing on a different generator (see [03 Data](03_data.md#out-of-distribution-ood-test-set-optional-recommended)) shows whether it learned something more general. A big drop here is common and is an important finding.

## 8. Explainability: Grad-CAM
Grad-CAM produces a heatmap of the image regions that most pushed the model towards "fake". It uses the gradients flowing into the last convolutional layer. We use it to:

- check the model looks at the face, not the background or borders (**shortcut learning**)
- explain individual mistakes
- show reviewers *why* an image was flagged

See [07 Explainability and demo](07_explainability_and_demo.md).

## 9. Optional extension: Vision Transformer
Any `timm` model works by changing one line, for example `model.name: vit_small_patch16_224`. Comparing a transformer with CNNs is a good extra if there's time.

Next: [05 Experiments](05_experiments.md)
