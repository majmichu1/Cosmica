# Cosmica TODO

Last updated: 2026-04-13

## In Progress

- [ ] **AI model training** — epochs 16–30 running, ~6h remaining
  - Resume if interrupted: `poetry run python -m cosmica.ai.training.train_n2s_v2 --resume cosmica/ai/models/checkpoint_epoch_N.pt`

## Next Up (this session)

- [ ] **Multi-session / multi-setup stacking** — highest priority feature
  - Different cameras, telescopes, pixel scales, exposure lengths across sessions
  - Per-session calibration (own darks/flats), normalize by background + signal scale
  - Weighted integration across all sessions
  - APP-style: drag multiple session folders, configure each, stack all to one result

- [ ] **DSO annotation layer** — quick win after WCS solve
  - Embedded Messier + brightest NGC/IC catalog (~1000 objects)
  - Project RA/Dec → pixel via WCS, draw labels on canvas
  - Toggle overlay button in Color tab (near WCS overlay)

- [ ] **FITS header editor** — simple dialog
  - Table of keyword / value / comment
  - Editable values, save back to FITS
  - Open from File menu or Project panel right-click

- [ ] **Smart FITS frame type detection** — read IMAGETYP/EXPTIME/OBJECT headers on import
  - Auto-tag as LIGHT/DARK/FLAT/BIAS
  - Manual override always available — user can correct any wrong tag
  - Warn instead of silently mis-tag

## AI Tools Roadmap (after denoising model is done)

- [ ] **AI gradient removal** — SETIAstro-style, train separate model after denoise
- [ ] **HaNBX-style AI narrowband enhancement** — Ha/OIII boosting
- [ ] **AI star reduction** — better than current morphological approach
- [ ] **AI deconvolution** — blind, no PSF measurement needed (BlurXTerminator-level)

## Other Planned

- [ ] **Gradient-corrected mosaic** — equalize background at panel seams (APP/PI level)
- [ ] **RGB / SHO Combine dialog** — dedicated mono+filter workflow (load R+G+B or Ha+OIII+SII)
- [ ] **Live stacking** — real-time stack for ASIAIR/OSC users
- [ ] **Comet registration** — align on nucleus instead of stars
- [ ] **SER/AVI video support** — planetary/lunar lucky imaging
- [ ] **Photometry** — magnitude measurement, variable star lightcurves
- [ ] **Sequence pipeline** — one-click folder → calibrate → align → stack

## Completed

- [x] Mask system, star detection refactor, cosmetic, curves, HT, deconvolution, color tools, banding
- [x] Denoise (NLM + Wavelet), star reduction, color calibration, narrowband, GHS, pixel math, channels
- [x] Wavelets GPU (à trous), HDR Mertens, plate solving, drizzle, mosaic, CLAHE, morphology
- [x] MLT, dynamic background samples, WCS overlay, LRGB combine, Python console
- [x] AI denoise J-invariant inference + star protection + tiled inference
- [x] PSF measurement, continuum subtraction, blink comparator
- [x] TGVDenoise (GPU Chambolle-Pock), SPCC with filter database
- [x] SubframeSelector with thumbnails + folder load
- [x] Auto-master calibration from raw frame folders or pre-made masters
- [x] Training resume support, content-filtered data extraction
- [x] 740 tests passing
