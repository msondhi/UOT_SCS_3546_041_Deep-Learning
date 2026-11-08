# 01 Overview

## The problem

When you open a bank account or apply for a card online, the bank asks for a photo of your ID and a selfie. This is called **KYC (Know Your Customer)**. It is how banks confirm you are a real person and who you say you are.

AI image generators (GANs and diffusion models) can now create photos of people **who do not exist** that look completely real. Criminals use them to build **synthetic identities**: fake customers made from a generated face plus stolen or invented details. These identities are used to:

- open accounts, build up credit, then max out cards and disappear ("bust-out" fraud)
- run **money mule** accounts that move the proceeds of crime
- make investigations harder, because there is no real person behind the account

Human reviewers increasingly can't spot these faces. An automated check can act as an extra layer.

## What we are building

A deep learning model that looks at a face photo and outputs **the probability that it is AI-generated**. We then measure:

1. **How accurate it is** on clean test images
2. **How robust it is** when photos are blurry, compressed, low-resolution, or photographed off a screen
3. **How well it generalizes** to faces from an AI generator it never saw during training
4. **What it is looking at**, using Grad-CAM heatmaps

## Goals

| # | Goal | How we'll show it |
|---|---|---|
| G1 | Build a working real-vs-fake classifier | Test accuracy and ROC-AUC |
| G2 | Show the value of transfer learning | Baseline CNN (E1) vs. pretrained models (E2-E5) |
| G3 | Measure real-world robustness | Accuracy at degradation severities 1-5 |
| G4 | Improve robustness | Standard model (E3) vs. robust-trained model (E6) |
| G5 | Explain decisions | Grad-CAM heatmaps on correct and wrong predictions |
| G6 | Demonstrate it | Gradio demo app |

## In scope

- Still face images (selfies), real vs. AI-generated
- Public research datasets only
- A research prototype, with a demo

## Out of scope

- Video deepfakes and face swaps in live video
- Matching the selfie to the ID document (face verification)
- Liveness detection (checking that a live person is in front of the camera)
- Production deployment

## Success criteria

- Fine-tuned model clearly beats the from-scratch baseline
- A robustness table and chart showing where the model breaks
- Measurable robustness improvement from degradation-aware training
- Heatmaps that give insight into what the model relies on
- A working demo, plus a report that covers ethics and real-world use

Next: [02 Setup](02_setup.md)
