# 💊 MedScan — Prescription Recognition System

![Python](https://img.shields.io/badge/Python-3.12-blue)
![Streamlit](https://img.shields.io/badge/Streamlit-App-red)
![License](https://img.shields.io/badge/License-MIT-green)
![OpenCV](https://img.shields.io/badge/OpenCV-Computer%20Vision-orange)

**A hybrid OCR + AI pipeline that extracts medicines, dosages, and usage instructions from prescription photos — with a confidence-driven local Privacy Engine, a polished Streamlit dashboard, and PDF report export.**

Built by **Malik Usman**

---

## Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Three Analysis Modes](#three-analysis-modes)
- [Any OpenAI-Compatible Provider](#any-openai-compatible-provider--not-locked-to-one-vendor)
- [Privacy Engine](#️-privacy-engine-confidence-driven-fully-local)
- [Pipeline Overview](#pipeline-overview)
- [Setup](#setup)
- [Deployment Notes](#deployment-notes--tesseract-is-a-system-binary-not-a-python-package)
- [Project Structure](#project-structure)
- [Disclaimer](#disclaimer)
- [License](#license)

---

## Overview

MedScan automates the error-prone process of manually reading handwritten
or printed medical prescriptions. It combines classical **Digital Image
Processing** with a choice of local OCR, cloud vision AI, or both (Hybrid),
plus a from-scratch, confidence-fused **Privacy Engine** that keeps patient
data off the wire before any cloud call — all wrapped in a dashboard with
light/dark mode, analytics, an editable results table, and PDF export.

## Features

- 🧪 **OpenCV image processing pipeline** — grayscale, denoising, CLAHE
  contrast enhancement, sharpening, adaptive thresholding
- 🖥️ **Offline OCR** via Tesseract + local fuzzy medicine-name matching —
  no internet, no API key, no data leaves the machine
- ☁️ **Any cloud AI provider** — Groq, OpenAI, OpenRouter, Together AI, or a
  custom OpenAI-compatible endpoint
- 🔀 **Hybrid mode (default)** — OCR extracts first, AI verifies/corrects
  against the actual image, with automatic escalation and graceful fallback
- 🛡️ **Local Privacy Engine** — multi-detector, confidence-fused PII
  redaction before any image is sent to a cloud provider
- ✏️ **Editable results table** — fix any OCR/AI mistake by hand, with a
  Source column showing what was kept/corrected/added
- 📊 **Analytics dashboard** — confidence charts, processing-time breakdown,
  image quality heuristics
- 🌗 **Light/dark mode**, PDF report export, and a one-click Tesseract
  auto-installer

## Three Analysis Modes

| Mode | What happens | Needs internet / API key? |
|---|---|---|
| **Hybrid (default, recommended)** | Tesseract OCR + local fuzzy medicine-name matching extracts a draft first. If a cloud AI provider is configured, that draft is sent — together with the privacy-sanitized image — for the AI to verify/correct/complete. If Tesseract finds nothing at all (or isn't installed), Hybrid escalates straight to full AI extraction instead. If no API is configured, you still get the raw Tesseract result. | Optional — works either way |
| **Tesseract (Offline, research)** | OCR + fuzzy matching only, for comparison | No |
| **Cloud API (research)** | Sends the image straight to the configured provider for from-scratch extraction, for comparison | Yes |

Switch modes anytime from the **Settings** page — no restart needed.

## Any OpenAI-Compatible Provider — not locked to one vendor

Cloud calls go through a single generic HTTP client (`modules/cloud_ai_engine.py`)
that works with any provider exposing an OpenAI-compatible `/chat/completions`
vision endpoint. Pick one in Settings, or point "Custom" at your own:

- Groq
- OpenAI
- OpenRouter
- Together AI
- Custom (any self-hosted or third-party OpenAI-compatible endpoint)

Hybrid mode also lets you send a **different image version to each engine** —
Tesseract typically reads a binarized "Final Processed" image best, while
vision AI models tend to do better on the "Original" or "Enhanced" version.

## 🛡️ Privacy Engine (confidence-driven, fully local)

Whenever Cloud API or Hybrid mode might send an image to an external provider,
a dedicated **Privacy Engine** (`privacy/` package) runs first — entirely
on-device, no network call. Rather than trusting OCR as the single source of
truth (which fails silently on handwritten prescriptions), it fuses evidence
from multiple independent detectors:

```
Image
  │
  ├─ Stage 1: Layout Detector     — pure OpenCV structural analysis (ink-density
  │                                  projection, ruled-underline detection). Needs
  │                                  ZERO legible text — this is what still finds
  │                                  the header block even when OCR reads garbage.
  ├─ Stage 2: OCR Detector        — label/keyword matching ("Name:", "Address:"),
  │                                  supporting evidence, not sole authority.
  ├─ Stage 3: Regex Detector      — phone/email/national-ID/DOB patterns, high
  │                                  precision.
  ├─ Stage 4: Confidence Fusion   — combines all evidence probabilistically into
  │                                  one confidence score per region.
  └─ Stage 5: Redaction Generator — auto-masks anything above threshold; a
                                     guaranteed top-10% fallback fires if every
                                     detector finds nothing, so an image is
                                     never sent fully unredacted.
  │
  ▼
Sanitized image ──▶ Cloud AI
```

User-specified manual top/bottom bands, an Rx-marker-aware header band, and
custom rectangles are also supported as high-confidence "user override"
inputs to the same fusion pipeline. A legacy **Whitelist strategy** (redact
everything except recognized clinical lines) remains available as an
explicit opt-in alternative.

Every run also produces a `PrivacyReport`: which detectors contributed,
per-field fused confidence, redaction coverage %, and safety-check flags
(high coverage, medicine-line overlap, fallback-baseline used) — shown in
the UI and included in the PDF report. **The original, unmodified image
never leaves your machine — only the sanitized copy is transmitted.**

This is a heuristic, best-effort system — not a certified compliance tool.
Always check the sanitized preview before relying on it with real patient
data, and prefer not including identifying details in the photo at all when
possible (crop or cover the header before taking the picture).

`privacy/` module layout:

```
privacy/
  types.py             # DetectedRegion, FusedRegion, PrivacyConfig, PrivacyReport
  layout_detector.py   # Stage 1 — OCR-independent structural analysis
  ocr_detector.py       # Stage 2 — label-based detection (supporting evidence)
  regex_detector.py     # Stage 3 — high-precision pattern detection
  confidence.py         # Stage 4 — probabilistic evidence fusion
  redaction.py          # Stage 5 — masking, safety checks, user overrides
  engine.py             # Orchestrator — single entry point: PrivacyEngine.process()
```

Extensible by design: a future detector (YOLO-based, PaddleOCR layout,
DocLayout, etc.) just needs to return `List[DetectedRegion]` — no changes
needed to the fusion, redaction, or app.py integration code.

## Pipeline Overview

1. **Digital Image Processing** (OpenCV): Grayscale → Median Filter →
   CLAHE contrast enhancement → Unsharp-mask sharpening → Adaptive
   Gaussian thresholding.
2. **Local PII detection + redaction** (Cloud API / Hybrid modes only, before
   any image leaves the device).
3. **Text recognition** via Tesseract OCR + local fuzzy matching, a cloud AI
   provider, or both (Hybrid).
4. **Analytics dashboard**: per-medicine confidence chart, overall confidence
   gauge, processing-time breakdown, local OpenCV image quality heuristics.
5. **Editable results table** — fix any OCR/AI mistake by hand before export;
   a "Source" column shows what OCR found vs. what AI kept/corrected/added.
6. **PDF report export** via ReportLab, including the Privacy Protection summary.

## Setup

```bash
git clone https://github.com/MalikUsmanAli-dev/MedScan.git
cd MedScan
pip install -r requirements.txt
```

### Offline mode (Tesseract) — install the OCR binary

- **Ubuntu / Debian:** `sudo apt install tesseract-ocr`
- **macOS (Homebrew):** `brew install tesseract`
- **Windows:** install from the
  [UB-Mannheim Tesseract build](https://github.com/UB-Mannheim/tesseract/wiki)
  and make sure it's on your `PATH` (or set the path directly in Settings).

The Settings page also has a one-click "Try Automatic Install" button that
attempts this for you via winget / Homebrew / apt / dnf.

### Cloud mode — add a provider + API key

Pick a provider in Settings (or "Custom" for your own endpoint) and paste an
API key — or copy `.env.example` to `.env` and set `CLOUD_API_KEY` to have it
picked up automatically on startup.

### Run

```bash
streamlit run app.py
```

## Deployment Notes — Tesseract is a system binary, not a Python package

`pip install pytesseract` only installs a thin Python wrapper. The actual OCR
engine (`tesseract`) must be installed separately **on whatever machine the
app actually runs on** — it does not travel with the project folder, and
`requirements.txt` alone will not install it.

- **Running locally:** install Tesseract once on that PC and it stays
  available every time you `streamlit run app.py` there.
- **Moving the project to a different PC/server:** install Tesseract again
  on that machine.
- **Streamlit Community Cloud:** this project includes `packages.txt`
  listing `tesseract-ocr` — Streamlit Cloud reads this and runs
  `apt-get install` automatically at deploy time, so it works with **zero
  local installation** on that platform.
- **Docker:** add `RUN apt-get update && apt-get install -y tesseract-ocr`
  before the `pip install -r requirements.txt` step.
- **Heroku:** requires an apt buildpack plus an `Aptfile` containing
  `tesseract-ocr`.

If Cloud API / Hybrid mode is used instead, none of the above applies for the
AI call itself — just make sure the API key/provider is set in that
deployment's Settings or environment.

## Project Structure

```
app.py                         # Streamlit entry point / page router
packages.txt                   # System packages for Streamlit Community Cloud (tesseract-ocr)
requirements.txt                # Python dependencies
.env.example                     # Copy to .env for optional API key / Tesseract path

privacy/                        # Confidence-driven Privacy Engine (see above)
  types.py
  layout_detector.py
  ocr_detector.py
  regex_detector.py
  confidence.py
  redaction.py
  engine.py

modules/
  image_processing.py           # OpenCV DIP pipeline
  ocr_engine.py                 # Offline Tesseract OCR + rule-based extraction
  medicine_dictionary.py        # Offline dictionary used for fuzzy name matching
  api_providers.py               # Provider registry (Groq/OpenAI/OpenRouter/Together/Custom)
  cloud_ai_engine.py             # Generic OpenAI-compatible vision client (extract + verify)
  analysis_router.py             # Routes to tesseract / api / hybrid pipelines
  analysis_types.py               # Shared result data structures
  auto_install.py                  # Best-effort automatic Tesseract installer
  ui_components.py                 # CSS injection, theme system, reusable UI widgets
  pdf_generator.py                 # ReportLab PDF report builder
```

## Disclaimer

Built for academic / portfolio demonstration purposes. Always verify
extracted medicine names, strengths, and dosing with a licensed pharmacist
or physician before any clinical or dispensing decision. The Privacy Engine
is a best-effort local heuristic, not a certified compliance/anonymization
tool.

## License

Released under the [MIT License](LICENSE) — free to use, modify, and
distribute with attribution.

---

https://medscan-prescription-recognition-system.streamlit.app/

<p align="center">Developed by <b>Malik Usman</b></p>
