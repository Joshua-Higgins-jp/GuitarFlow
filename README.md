# GuitarFlow 🎸

**A hands-on Machine Learning and Data Engineering portfolio project**

---

## What is This?

GuitarFlow is a 2-class image classifier portfolio project that categorises electric guitars
and acoustic guitars. The classification problem is intentionally simple and
forever iterable - it can always be improved with more data, better
augmentation, or a more sophisticated architecture.

However, 'perfecting the model' isn't the main objective of this project (nor realistic).
Rather, the engineering around it - meaning the data versioning, experiment tracking, inference 
logging, monitoring, deployment, CI/CD, and the architectural decisions and tradeoffs 
that hold it all together. A simple problem with the bells and whistles of a production
ML system, publicly documented, from end to end.

---

## Why This Project?

My engineering day job is Backend, ML, and CV, and for that matter, NDA. Sadly, this means the 
most interesting engineering decisions I make day to day are not mine to share publicly
(... and the code, of course!)

GuitarFlow exists to fix that. I am building it in my own time, on a problem I chose,
with the full freedom to document every decision, including the ones that didn't
work out. The model classifies guitars. The portfolio demonstrates how to
build, version, deploy, and monitor a real ML system. While the product is imaginary, 
the development process is real.

---

## Architecture

```
[Image APIs]  →  [SQLite metadata DB]  →  [train/val/test CSVs]
                                                    |
                                          [HF Hub — dataset repo]
                                          (versioned split CSVs,
                                           URL + label + split only,
                                           no raw images)
                                                    |
                                           [MLflow — local]
                                           (experiment tracking,
                                            logs dataset version
                                            commit hash per run)
                                                    |
                                          [model weights — .pth]
                                                    |
                                    [HF Spaces — Streamlit UI]
                                    (upload or camera capture)
                                                    |
                                         [inference logging]
                                        /                   \
                              [HF Dataset Hub]          [DataDog]
                              (permanent record,         (live monitoring,
                               versioned flat store)      error alerting,
                                                          confidence metrics)
```

---

## Engineering Decisions and Tradeoffs

| Layer | Tool | Why                                               | Tradeoff                                                                 |
|---|---|---------------------------------------------------|--------------------------------------------------------------------------|
| Language | Python 3.12 | Stable, wide package support                      | Not latest version (meh)                                                 |
| Packaging | uv + pyproject.toml | Fast, modern, reproducible, works with Pycharm    | It's Not pip                                                             |
| Model architecture | ResNet18 (PyTorch) | Lightweight, well understood, fast to train locally on CPU | ViT would be overkill at this data scale                                 |
| Experiment tracking | MLflow (local) | Prior experience, no account needed, runs offline via `mlflow ui` | Not hosted, not shareable without extra setup                            |
| Data versioning | HF Hub dataset repo | Free, Git-native, same ecosystem as deploy        | Public only on free tier — push split CSVs (URL + label), not raw images |
| Deployment | HF Spaces + Streamlit | Free tier, ML-native, supports Streamlit SDK natively | No custom domain on free tier                                            |
| Inference persistence | HF Dataset Hub | Free, versioned, no license concerns (own data)   | Not a queryable database, flat versioned record store                    |
| Monitoring | DataDog free tier | Industry standard tool, good to have on resume    | 1 day log retention (ok for a demo)                                      |
| OOD handling | Trained "no_guitar" catch class | No extra model, fits existing pipeline            | Requires ~250 varied images and training, kinda annoying                 |
| Drift signal | Confidence rolling average | Simple, interpretable, no extra tooling required  | Not rigorous, but good enough for a demo                                 |
| CI/CD | GitHub Actions | Free, integrates with HF Spaces deploy            | —                                                                        |
| Testing | pytest | Standard                                          | —                                                                        |
| Logging | loguru | Clean API, structured output                      | —                                                                        |

---

## Project Stages

_This README is updated as the project progresses._

### Stage 1 — Data Collection ✅
- Manually fetched (curated) ~500 images via unsplash (NOT automated, complying with ToS)
- SQLite metadata store (image path, source, label, url, hash, etc)
- Basic deduplication and quality checks

### Stage 2 — Model Development ✅
- ResNet18 fine-tuned on ~250 images per class
- Eval transform pipeline (224x224, ImageNet normalisation)
- Softmax confidence scores returned per class

### Stage 3 — Deployment 🚧
- [x] Streamlit UI — upload or camera capture
- [x] Inference pipeline — PIL Image in, dict of confidence scores out
- [x] model.infer() decoupled from file path dependency (accepts PIL Image directly)
- [ ] Wire up inference logging to HF Dataset Hub
- [ ] Deploy to HF Spaces

### Stage 4 — Experiment Tracking and Data Versioning 📋
- [ ] MLflow local tracking — log loss, accuracy, precision, recall,
      confusion matrix, hyperparams per training run
- [ ] Export train/val/test splits from SQLite to CSV
- [ ] Push split CSVs to HF Hub dataset repo
- [ ] Log HF dataset commit hash in MLflow run so each experiment
      references the exact data version it was trained on

### Stage 5 — Monitoring 📋
- [ ] DataDog custom metric: guitarflow.confidence.score
- [ ] DataDog monitor: alert when 7-day rolling average confidence
      drops below threshold (indicates potential data drift or model
      degradation)
- [ ] Log per-inference record to HF Dataset Hub:
      timestamp, predicted class, confidence, image hash, input method

### Stage 6 — CI/CD 📋
- [ ] GitHub Actions: run pytest on pull request
- [ ] GitHub Actions: deploy to HF Spaces on merge to main
- [ ] uv.lock committed for fully reproducible builds

---

## Tech Stack

| Layer | Tool |
|---|---|
| Language | Python 3.12 |
| Packaging | uv + pyproject.toml |
| Model | PyTorch, torchvision (ResNet18) |
| UI | Streamlit (hosted on HF Spaces) |
| Logging | loguru |
| Experiment tracking | MLflow (local) |
| Data versioning | HF Hub dataset repo (split CSVs) |
| Inference persistence | HF Hub dataset repo (inference logs) |
| Monitoring and alerting | DataDog free tier |
| CI/CD | GitHub Actions |
| Testing | pytest |

---

## Running Locally

```bash
git clone https://github.com/Joshua-Higgins-jp/GuitarFlow
cd guitarflow
uv sync
PYTHONPATH=. uv run streamlit run src/app/streamlit_app.py
```

MLflow UI (experiment tracking):

```bash
uv run mlflow ui
# open http://localhost:5000
```

---

## Future Architecture — Detection + Classification Pipeline

The production-grade approach for this problem is a two-stager:

```
Raw image
    |
[Stage 1 — Object Detection]
YOLO or similar: "is there a guitar in this image, and where?"
    |
bounding box crop
    |
[Stage 2 — Classification]
ResNet18: "acoustic or electric?"
```

This is the preferred "I've got time to kill on this demo" computer vision architecture - 
a purpose made detector handles localisation, and the classifier just gets the crop of the 
guitar rather than the full image scene. It would also eliminate the need
for a trained "no_guitar" OOD catch class entirely, since the detector
handles that gate. Also, YOLO in commercial use requires licensing,
so I would be setting up the product for additional operating expenses to offset.

Why it is not implemented in the current version:

YOLO pretrained on COCO doesn't have a guitar class (womp womp). 
Preparing this data annotation is quite time consuming, so I decided not to do it
for the MVP. Eventually, I might do that so the inference process goes  
"detect guitar" -> "classify guitar type".

Current mitigation: a trained "no_guitar" catch class handles out-of-distribution
inputs within the existing single-stage classifier.

This upgrade is in scope for a future iteration given sufficient time and
annotated data.

---

## Design Notes and Lessons Learned

_Living section — updated as the project progresses._

- ResNet18 confidently misclassifies dark, stage-lit electric guitars as
  acoustic because pickup hardware is not visible at that lighting and angle.
  The model is working correctly given what it can see. This issue is a
  training data distribution gap, not a model architecture problem. Fix:
  augment training data with concert photography.

- Streamlit Community Cloud was the original planned deployment target.
  However, I prefer to package containers, and free tier doesn't appear to support this. 
  HuggingFace Spaces supports Streamlit as a first-class SDK and
  is the better platform for ML demos (sad for StreamLit tbh, back in the day it was the GOAT).

- uv was chosen over pip/venv for packaging. New PyCharm versions also natively supports it. 
  Pixi also works fine, but overkill for common non-conda packages. 

- Model weights are loaded once at startup using @st.cache_resource to
  avoid reloading on every Streamlit interaction.

---

## License

MIT License