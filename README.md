# Cosmica - Open Source AI Image Processing Suite for Astrophotography

**Professional astrophotography image processing. GPU-accelerated, free, and open source alternative to PixInsight.**

[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![Tests](https://github.com/majmichu1/cosmica/actions/workflows/ci.yml/badge.svg)](https://github.com/majmichu1/cosmica/actions)
[![Buy Me a Coffee](https://img.shields.io/badge/Buy%20Me%20a%20Coffee-donate-yellow)](https://buymeacoffee.com/majmichu)



<img width="2559" height="1600" alt="obraz" src="https://github.com/user-attachments/assets/a688865b-476d-4139-9cfc-4efbe89435a7" />




## 📥 Download (Alpha Release)
You don't need to install Python to use Cosmica. Download the standalone, ready-to-use versions (The AppImage is a portable **CPU-only** version. For full NVIDIA GPU acceleration, please install via Poetry):

* **Windows:** [Download .exe (v0.1.9-alpha)](https://github.com/majmichu1/Cosmica/releases/latest)
* **Linux:** [Download .AppImage (v0.1.9-alpha)](https://github.com/majmichu1/Cosmica/releases/latest)

*(Note: This is an Alpha release. Bug reports and feedback are welcome in the Issues tab!)*



## 🆚 Why Cosmica?
Cosmica is built as a modern workflow tool from calibration to export, featuring out-of-the-box AI integration.

| | Cosmica | Siril | PixInsight |
|---|---|---|---|
| **Price** | Free (GPL v3) | Free | €230+ |
| **GPU Acceleration** | ✅ Full PyTorch | ⚠️ Partial | ⚠️ Partial |
| **AI Denoise / Sharpen** | ✅ Built-in | ❌ | ✅ Extra cost |
| **Multi-Session Stacking** | ✅ | ❌ | ✅ |
| **Spatially-Varying Deconvolution** | ✅ | ❌ | ✅ |
| **Plate Solve + PCC** | ✅ ASTAP / Astrometry.net | ⚠️ Basic | ✅ |
| **Scripting** | Python console | Python | JavaScript |
| **Star Removal** | ✅ StarNet | ❌ | ✅ |



## ✨ Features

### Acquisition & Pre-Processing
- **Calibration** — Master dark / flat / bias creation with batch light frame calibration
- **Alignment** — Star-based registration (1-pass, 2-pass refinement, triangle matching), FFT phase-correlation, and comet nucleus tracking
- **Stacking** — Sigma-clip, winsorized sigma, linear fit, percentile clip, ESD, and min/max rejection
- **Drizzle Integration** — 2× / 3× scale-up for undersampled data
- **Multi-Session Stacking** — Combine data from multiple nights with per-session adaptive weighting
- **Subframe Selector** — Automatic frame scoring by FWHM, eccentricity, SNR, background, and star count
- **Debayer** — RGGB, BGGR, GRBG, GBRG with VNG and other methods; auto-detection from FITS header

### AI-Powered Tools
Local AI models download automatically on first use. No cloud required.

- **AI Denoise** — Noise2Self U-Net trained on real astro images
- **AI Sharpen** — Neural deconvolution for recovering fine detail
- **Star Removal** — StarNet-based starless image generation

### Detail Enhancement
- **Deconvolution** — Richardson-Lucy with optional total-variation regularization and deringing protection
- **Spatially-Varying Deconvolution** — Per-zone PSF measurement and blending for field curvature / coma correction
- **Wavelet Sharpening** — Multi-scale contrast enhancement
- **MLT (Multiscale Linear Transform)** — Selective noise reduction per frequency band
- **Local Contrast Enhancement** — GPU-accelerated CLAHE
- **Unsharp Mask** — Standard and advanced masking
- **Median Filter** — Impulse noise removal

### Color & Calibration
- **Photometric Color Calibration (PCC)** — Plate solve then match against Gaia DR3
- **SPCC** — Spectrophotometric calibration with filter response curves
- **Background Extraction** — Polynomial surface fitting, ABE (RBF-based), and dynamic sample placement
- **Background Neutralization** — Robust color balancing from background samples
- **Color Calibration** — Statistical and catalog-based correction
- **SCNR** — Green noise reduction for narrowband and OSC images
- **Color Adjustment** — Saturation, hue shift, vibrance
- **Curves** — Per-channel curve editor with histogram overlay
- **Histogram Transform** — Black point, midtone, white point with live preview

### Narrowband & Composition
- **Narrowband Combine** — HOO, SHO, and custom palette mappings
- **LRGB Combine** — Luminance-weighted RGB merging
- **Channel Combine** — Custom channel mapping dialog
- **Continuum Subtraction** — Remove broadband contamination from narrowband filters
- **HDR Composition** — Multi-exposure blending

### Corrections & Utilities
- **Cosmetic Correction** — Hot, cold, and dead pixel repair
- **Banding Reduction** — Horizontal and vertical pattern removal
- **Chromatic Aberration Correction** — Auto-detect and manual shift
- **Vignette Correction** — Model-based flat-field emulation
- **Star Reduction** — Shrink star bloat without star removal
- **Morphology** — Dilate, erode, open, close for star masks
- **TGV Denoise** — Total generalized variation (non-AI, edge-preserving)
- **PSF Measurement** — Interactive FWHM, ellipticity, and angle from detected stars

### Plate Solving & Annotation
- **Plate Solve** — ASTAP and astrometry.net with auto-fallback
- **WCS Overlay** — Catalog star positions drawn on the image
- **DSO Annotation** — Automatic deep-sky object labels from solved coordinates

### Workflow & UI
- **Modern Dark Theme** — GitHub-inspired green accent, designed for long nights
- **4-Panel Layout** — Project tree | Canvas + Histogram | Tools | Log
- **Split Before/After Preview** — Draggable divider with live preview on every tool
- **Blink Comparator** — A/B frame comparison at variable FPS
- **Interactive Histogram** — Log scale, per-channel stats, clip indicators
- **Curve Editor** — Per-channel control points with histogram backdrop
- **Macro Recorder** — Record and playback processing steps
- **Python Console** — Embedded scripting dock with live image access
- **Batch Processing** — Unattended folder processing
- **Smart Processor** — One-click automated workflow
- **Equipment Profiles** — Camera / telescope metadata for plate-scale calculations
- **Undo / Redo** — Full history stack with display-reference matching
- **Presets** — Save and recall tool settings

### File Support
- **Read:** FITS, XISF, TIFF, PNG, JPEG (auto-debayer for OSC)
- **Write:** FITS, XISF, TIFF (8/16-bit), PNG (8/16-bit), JPEG



## 🚀 Getting Started (For Developers)

### Prerequisites
- Python 3.11–3.14
- [Poetry](https://python-poetry.org/) (dependency management)

### Installation

```bash
# Clone the repository
git clone [https://github.com/majmichu1/cosmica.git](https://github.com/majmichu1/cosmica.git)
cd cosmica

# Install dependencies
poetry install

# Run the application
poetry run cosmica
# or
poetry run python -m cosmica
````

### Building Standalone Binary

```bash
poetry install --with build
poetry run pyinstaller build/cosmica.spec
```



## 🧪 Development

### Run Tests

```bash
poetry run pytest   # 729+ tests
```

### Run Linter

```bash
poetry run ruff check .
```

### Run Type Checker

```bash
poetry run mypy cosmica
```



## 🤖 AI Model Training

Cosmica includes self-supervised AI training scripts. You can train your own denoise model on your astro images:

```bash
# Place your FITS files in astro_data/
mkdir -p astro_data
cp /path/to/your/*.fits astro_data/

# Train the denoise model
poetry run python scripts/train_denoise_model.py --input astro_data --epochs 30
```

The model uses **Noise2Self** — a self-supervised approach that learns to denoise from noisy images alone, without needing clean reference images.



## 🤝 Contributing

Contributions are welcome\! Please read our [Contributing Guide](https://www.google.com/search?q=CONTRIBUTING.md) for details on how to get started.



## 📄 License

This project is licensed under the **GNU General Public License v3.0** (GPL-3.0).

The GPL v3 is required because Cosmica uses PyQt6, which is licensed under GPL v3 for open-source use.



## ☕ Support

If you find Cosmica useful and would like to support the project:

[](https://buymeacoffee.com/majmichu)

Every contribution helps keep this project free and open source\!



## 🙏 Acknowledgments

  - [PyQt6](https://www.riverbankcomputing.com/software/pyqt/) — User interface framework
  - [PyTorch](https://pytorch.org/) — GPU-accelerated computation
  - [Astropy](https://www.astropy.org/) — FITS file I/O
  - [Noise2Self](https://github.com/czbioinfo-scripps/Noise2Void) — Self-supervised denoising
  - All the open-source astronomical software that inspired this project



## 📊 Stats

  - **42** core processing modules
  - **729+** tests
  - **7.7M** parameters in the AI denoise model
  - **181k+** training patches from real astro images

