# Cosmica — Getting Started Guide

Welcome to **Cosmica**, a professional astrophotography image processing application.

## Installation

**Note:** Cosmica is currently available as **source code only**. We are working on pre-compiled installers (.exe, .dmg, .AppImage) for a future release.

### From Source (Linux, macOS, Windows)

```bash
# 1. Clone the repository
git clone https://github.com/majmichu1/cosmica.git
cd cosmica

# 2. Install dependencies (requires Poetry)
# If you don't have Poetry: https://python-poetry.org/docs/#installation
poetry install

# 3. Run the application
poetry run cosmica

## First Launch

When you first open Cosmica, you'll see the main interface with four panels:

```
┌─────────────┬──────────────────────┬─────────────┐
│   Project   │     Image Canvas     │    Tools    │
│   Panel     │   + Histogram        │    Panel    │
│   (left)    │     (center)         │   (right)   │
├─────────────┴──────────────────────┴─────────────┤
│               Log Panel (bottom)                  │
└──────────────────────────────────────────────────┘
```

## Quick Start — Your First Image

### Step 1: Open an Image
- **File → Open Image** or press `Ctrl+I`
- Or **drag and drop** a FITS/XISF file onto the canvas

### Step 2: Stretch the Image
1. Go to the **Tools** panel (right)
2. Click the **Stretch** tab
3. Adjust the **Midtone** slider (default: 0.25)
4. Click **Apply**

### Step 3: Save Your Work
- **File → Save Image As...** or `Ctrl+Shift+S`
- Choose format: FITS (astronomy), TIFF (print), PNG (web)

## Basic Workflow

The recommended processing order for a typical astrophotography image:

```
1. Pre-Process
   ├── Calibration (Dark/Flat/Bias frames)
   ├── Cosmetic Correction (Hot/cold pixels)
   └── Subframe Selector (Frame scoring)

2. Stacking
   └── Sigma-Clip Rejection + Average Integration

3. Stretch
   ├── Auto-Stretch (midtone balance)
   └── Histogram Transform (manual control)

4. Background
   ├── Background Extraction (remove gradients)
   └── ABE (Automatic Background Extraction)

5. Color
   ├── Color Calibration (white balance)
   └── SCNR (remove green cast)

6. Detail
   ├── Deconvolution (sharpening)
   ├── Noise Reduction
   └── Local Contrast (CLAHE)

7. Export
   └── File → Export (TIFF 16-bit / PNG / JPEG)
```

## Keyboard Shortcuts

| Shortcut | Action |
|----------|--------|
| `Ctrl+N` | New Project |
| `Ctrl+O` | Open Project |
| `Ctrl+S` | Save Project |
| `Ctrl+I` | Open Image |
| `Ctrl+Shift+S` | Save/Export Image |
| `Ctrl+,` | Preferences |
| `Ctrl+Z` | Undo |
| `Ctrl+Shift+Z` | Redo |
| `Ctrl+0` | Fit to Window |
| `Ctrl+1` | Zoom 100% |
| `Ctrl+P` | Pixel Math |
| `Ctrl+B` | Batch Processing |
| `Ctrl+Shift+P` | Smart Processor |
| `Ctrl+Q` | Quit |

## Features

Cosmica is currently **100% free and open-source**. All features are available to everyone.

### Processing Pipeline
- **Calibration**: Dark/Flat/Bias frame management.
- **Stacking**: Sigma-clip, winsorized sigma, and linear fit rejection.
- **Stretch**: Auto-stretch, GHS, and manual curves.
- **Tools**: Background extraction, SCNR, deconvolution, and more.

### AI Features (Coming Soon)
- **AI Denoise**: Self-supervised learning for superior noise reduction.
- **AI Sharpen**: Neural deconvolution for sharpening without ringing artifacts.

## Need Help?

- **Help → About Cosmica** — version info
- **GitHub Issues** — [Report bugs](https://github.com/majmichu1/cosmica/issues)
