# Cosmica TODO

Last updated: 2026-04-12

## In Progress / Next Up

- [ ] **Blink Comparator** — flip between two images rapidly for alignment/quality checking
  - Two image slots (A/B), FPS slider, toggle key (B or spacebar)
  - Canvas-level switching (no processing), async load from file or current image

- [ ] **Resume AI model training (epochs 5–30)**
  ```bash
  poetry run python -m cosmica.ai.training.train_n2s_v2 \
      --resume cosmica/ai/models/checkpoint_epoch_4.pt
  ```
  - ~26 epochs × ~56min ≈ 24h on RTX 5060
  - Content-filtered re-extraction (`prepare_data_v2.py`) recommended before final production model

## Feature Gaps vs PixInsight / Siril

### High Priority

- [ ] **Live stacking** — real-time stack update as frames arrive (OSC/ASIAIR)
- [ ] **Sequence pipeline** — one-click calibrate → align → stack full session folder
- [ ] **FITS header editor** — view/edit FITS header key-value pairs in a dialog
- [ ] **Blink comparator** (see above)

### Medium Priority

- [ ] **StarNet subprocess** — binary must be installed; wrapper already in code
- [ ] **Dynamic background extraction (interactive)** — canvas sample mode implemented; verify end-to-end on real image
- [ ] **WCS overlay** — plate solve + draw catalog stars on canvas; implemented but untested
- [ ] **Script recorder / macro playback** — macro system exists; add playback from file
- [ ] **AI model retrain** — content-filtered patches → fix ringing artifacts around stars
- [ ] **AI sharpen model** — train a separate sharpening model (blind deconvolution)

### Lower Priority

- [ ] **Annotation layer** — draw circles/arrows/text on image for presentation output
- [ ] **Noise estimation panel** — per-channel background σ, SNR dB (statistics dialog updated but not panel)
- [ ] **PSF-driven deconvolution auto-fill** — "Measure" button wired; make sure FWHM flows into deconvolution params
- [ ] **Continuum subtraction** — Ha-continuum narrowband workflow; implemented but needs UI polish
- [ ] **Multi-channel export** — export separate R/G/B/Ha/OIII as individual FITS
- [ ] **Image annotation from WCS** — label DSOs, double stars from catalog on canvas

## Completed (Phase A–D summary)

- [x] Mask system + star detection refactor
- [x] Curves, histogram transform, deconvolution, color tools, banding
- [x] Denoise (NLM + wavelet), star reduction, color calibration, narrowband, GHS, pixel math
- [x] Wavelets GPU (à trous via torch.conv2d), HDR Mertens, plate solving, drizzle, mosaic, CLAHE, morphological ops
- [x] MLT (multi-scale linear transform) with per-band boost + noise thresholds
- [x] Dynamic background sample placement (canvas click → BackgroundParams.manual_points)
- [x] WCS overlay (canvas draw after PCC)
- [x] LRGB combine (CIE Lab channel replacement)
- [x] Python REPL console (QDockWidget)
- [x] AI denoise J-invariant inference + star protection blend
- [x] PSF measurement in statistics dialog
- [x] Continuum subtraction
- [x] Training resume support (`--resume` flag)
- [x] 271 tests passing
