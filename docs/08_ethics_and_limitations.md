# 08 Ethics and limitations

## Intended use
A **research prototype** that flags possibly AI-generated faces **for a human reviewer to check**. It should never be the only reason someone is denied an account or treated as a suspect.

## Risks

### Wrongly rejecting real people (FRR)
A false rejection can deny someone access to banking. If errors are higher for some groups (by skin tone, age, gender, or camera quality, which often tracks income), the system is unfair and may be discriminatory.

**What we do:** report FRR clearly. If time allows, check errors on a small, manually reviewed sample across visible attributes, and state that a full fairness audit would need a properly labelled dataset.

### Missing fakes (FAR)
Over-trusting the model creates false security. Fraudsters adapt quickly and will use new generators.

**What we do:** test on an unseen generator and report the drop honestly.

### Dual use
A detector can be used to *train better fakes* (an adversary tests faces against it until one passes).

**What we do:** don't publish a public API. The demo runs locally.

### Privacy
The real faces are real people (FFHQ, from Flickr under open licences). We use them only for research, and show just a few examples in the report and slides.

## Limitations
- **One generator family:** training fakes are all StyleGAN; newer diffusion models may look very different.
- **Simulated degradations:** our blur/JPEG/screen effects approximate real conditions, but aren't real phone uploads.
- **Clean, aligned faces:** FFHQ faces are centred and high quality, unlike real selfies.
- **Not a full KYC system:** there's no liveness check, ID matching, or video.
- **Grad-CAM is approximate:** heatmaps suggest, but don't prove, what the model uses.

## Law enforcement and payments context
- **Payments:** where this check would sit in onboarding, how it combines with other signals (device, behaviour, credit history), and what each error costs.
- **Law enforcement:** synthetic identities as a tool for mule networks and fraud; why a model score alone isn't evidence; the importance of human review and audit trails.

## Data handling
- Only public datasets. No customer, employer, or police data.
- No datasets or model weights in git.

Next: [09 Project plan](09_project_plan.md)
