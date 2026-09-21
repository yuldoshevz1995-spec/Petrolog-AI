# User guide

## Inputs
* **PPL and XPL images of the same field** (JPEG/PNG/TIFF, any size). Images
  larger than 900 px on the longer side are reduced for segmentation.
* A **composite file** with both views side by side can be split with
  `kadrlarni_ajratish(path)`; assign nicols to the two panels with the score
  2S − V (the higher score is XPL).
* Optional **pixel size** in µm/px (`mkm_px`); otherwise `detect_scale` reads it
  from the printed scale bar.

## Main functions and options
| Function | Purpose | Options (default) |
|---|---|---|
| `detect_scale(path)` | find scale box, read its label | env `TESSERACT` (tesseract) |
| `segment_juft(ppl, xpl, k, mkm_px, max_tomon, bekor)` | joint segmentation | `k`=7, `max_tomon`=900 px, `bekor`=list of (x, y, w, h) boxes to exclude |
| `segment_kmeans(path, k, ...)` | PPL-only segmentation | `k`=6 |
| `modal_tarkib(result, eng_kam_ulush)` | merge phases by group | minor phases below 1 % pooled |
| `olchovlar(ppl, xpl, segment)` | fabric measurements and terms | — |

## Outputs
`segment_juft` returns a dict with `fazalar` (one entry per phase):
`ulush_foiz` area %, `talqin` group label, `ishonch` confidence (0–1, rule
specificity, not a probability), `optik_belgi` evidence, `izoh` note,
`median_diametr`/`max_diametr` (µm if a scale is known, else px),
`donlar_ajraldi` (False when the phase is a network or unresolved groundmass),
plus overlay images `qoplama` (PPL) and `qoplama_xpl`.

`olchovlar` returns `yonalganlik` (coherence 0–1 and direction), `tomirlar`
(vein share), `don_shakli` (convexity, elongation), and the terms
`struktura`, `tekstura`, `morfologiya`, each with the value (`sabab`) that
produced it.

## Expected behaviour and limits
Pre-processing (splitting, nicol ordering, scale reading) is reliable on
archive photographs. Mineral-group assignment from colour is **not** reliable
in altered, fine-grained volcanic rocks (see Section 4 of the paper): treat
group labels with confidence < 0.5 as suggestions for the petrographer.
Scale boxes drawn in styles other than a white box or a dark-framed box are
not detected.
