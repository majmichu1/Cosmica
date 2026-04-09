# Cosmica

**Professional astrophotography image processing. GPU-accelerated, free and open source.**

[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![Tests](https://github.com/majmichu1/cosmica/actions/workflows/ci.yml/badge.svg)](https://github.com/majmichu1/cosmica/actions)
[![Buy Me a Coffee](https://img.shields.io/badge/Buy%20Me%20a%20Coffee-donate-yellow)](https://buymeacoffee.com/majmichu)

---

Cosmica is a professional astrophotography image processing application built as a free, open-source alternative to PixInsight. It features GPU-accelerated processing, AI-powered denoising, and a modern dark-themed UI designed for a smooth workflow from calibration to export.

## ✨ Features

### Processing Pipeline
- **Calibration** — Dark/flat/bias frame calibration with master creation
- **Stacking** — Sigma-clip, winsorized sigma, and linear fit rejection
- **Stretch** — Auto-stretch, GHS, histogram transform, curves
- **Background Extraction** — Polynomial surface fitting, ABE
- **Color Tools** — SCNR, color calibration, saturation/hue/vibrance
- **Detail Enhancement** — Deconvolution, wavelet sharpening, CLAHE, unsharp mask
- **Transforms** — Crop, rotate, flip, resize, bin
- **Masks** — Luminance/range masks with live preview

### AI-Powered (Coming Soon)

> **Note:** AI models are currently in training. The first release ships with traditional algorithm fallbacks. AI features will be added in a future update after testing.

- **AI Denoise** — Noise2Self trained on real astro images (model training in progress)
- **AI Sharpen** — Neural deconvolution (architecture ready, training pending)
- **Star Removal** — StarNet-like starless image generation (planned)

### Training Your Own AI Models

Cosmica includes self-supervised AI training scripts. You can train your own denoise model on your astro images:

### GPU Acceleration
Cosmica uses PyTorch for GPU acceleration on **critical processing operations**:
- Stacking (sigma-clip rejection)
- Calibration (bias/dark/flat correction)
- Stretch (MTF, GHS)
- Color tools (SCNR, HSV conversion)
- Curves (LUT application)
- Vignette correction

### UI/UX
- Modern dark theme with GitHub-inspired green accent
- 4-panel layout: Project | Canvas + Histogram | Tools | Log
- Split-view before/after preview with draggable divider
- Interactive histogram display with log scale
- Curve editor with per-channel control points
- Full keyboard shortcuts
- Undo/Redo with 50-step history
- Preset system for all tools

### File Support
- **Read**: FITS, XISF, TIFF, PNG, JPEG
- **Write**: FITS, XISF, TIFF (8/16-bit), PNG (8/16-bit), JPEG

## 🚀 Getting Started

### Prerequisites
- Python 3.11–3.14
- [Poetry](https://python-poetry.org/) (dependency management)

### Installation

```bash
# Clone the repository
git clone https://github.com/majmichu1/cosmica.git
cd cosmica

# Install dependencies
poetry install

# Run the application
poetry run cosmica
# or
poetry run python -m cosmica
```

### Building Standalone Binary

```bash
poetry install --with build
poetry run pyinstaller build/<platform>/cosmica.spec
```

## 🧪 Development

### Run Tests
```bash
poetry run pytest  # 729+ tests
```

### Run Linter
```bash
poetry run ruff check .
```

### Run Type Checker
```bash
poetry run mypy cosmica
```

## 📖 Documentation

- [Getting Started Guide](docs/getting-started.md)
- [Processing Workflow](docs/workflow.md)
- [Tools Reference](docs/tools.md)

## 🤖 AI Model Training

Cosmica includes self-supervised AI models that train on your own astro images:

```bash
# Place your FITS files in astro_data/
mkdir -p astro_data
cp /path/to/your/*.fits astro_data/

# Train the denoise model
poetry run python scripts/train_denoise_model.py --input astro_data --epochs 30
```

The model uses **Noise2Self** — a self-supervised approach that learns to denoise from noisy images alone, without needing clean reference images.

## 🤝 Contributing

Contributions are welcome! Please read our [Contributing Guide](CONTRIBUTING.md) for details on how to get started.

This project follows the [Contributor Covenant Code of Conduct](CODE_OF_CONDUCT.md).

## 📄 License

This project is licensed under the **GNU General Public License v3.0** (GPL-3.0).

The GPL v3 is required because Cosmica uses PyQt6, which is licensed under GPL v3 for open-source use.

## ☕ Support

If you find Cosmica useful and would like to support the project:

[![Buy Me a Coffee](https://img.shields.io/badge/Buy%20Me%20a%20Coffee-donate-yellow?style=for-the-badge)](https://buymeacoffee.com/majmichu)

Every contribution helps keep this project free and open source!

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
