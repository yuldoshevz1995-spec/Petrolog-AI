# Petrolog-AI — paper version (Computers & Geosciences, 2026)

Code, evaluation scripts and per-pair results for

> Yuldoshev, Z., 2026. *Automated thin-section analysis of felsic volcanic rocks
> from the Qorason paleovolcano, Uzbekistan.* Submitted to Computers & Geosciences.

This folder contains **only the part of Petrolog-AI evaluated in the paper**: a
transparent, rule-based pipeline for archive thin-section photographs
(plane-polarised, PPL, and cross-polarised, XPL, light). The web application,
user accounts and the literature-derived reference base of the full system are
not included.

## What the pipeline does

| Step | Module | Paper section |
|---|---|---|
| Split composite images (two panels in one file) | `petrolog/imaging/split.py` | 3.1 |
| Read the printed scale bar (box detection + Tesseract OCR, majority vote) | `petrolog/imaging/scale_bar.py` | 3.1 |
| Joint PPL–XPL CIELAB k-means segmentation + mode filter | `petrolog/imaging/segment.py` | 3.2 |
| Phase description (median colour, area-weighted grain size) | `petrolog/imaging/segment.py` | 3.3 |
| Ordered optical rules → mineral group + confidence + evidence | `petrolog/imaging/segment.py` (`_talqin_juft`) | 3.4 |
| Fabric: structure tensor, veins, grain shape → structure/texture/morphology | `petrolog/fabric.py` | 3.5 |
| Reference labels from petrographic descriptions | `petrolog/labels.py` | 4.1 |

## Installation

Requires Python ≥ 3.10 and [Tesseract OCR](https://github.com/tesseract-ocr/tesseract) ≥ 4.1
(needed only for reading scale bars; without it sizes are reported in pixels).

```bash
git clone https://github.com/yuldoshevz1995-spec/Petrolog-AI.git
cd Petrolog-AI/paper-cg2026
pip install -r requirements.txt
pytest -q tests            # 4 smoke tests on synthetic images
```

If Tesseract is not on the `PATH`, set `TESSERACT=/path/to/tesseract`.
Runs on a standard office computer (single CPU core, no GPU); a pair takes
about 2–4 s for archive images.

## Quick start: analyse one image pair

```python
import cv2
from petrolog.imaging import detect_scale, segment_juft, modal_tarkib
from petrolog.fabric import olchovlar

scale = detect_scale("ppl.jpg")                      # {'topildi': True, 'mkm_px': 5.26, ...}
seg = segment_juft("ppl.jpg", "xpl.jpg", k=7,
                   mkm_px=scale.get("mkm_px"),
                   bekor=[scale["quti"]] if scale.get("quti") else None)
for phase in modal_tarkib(seg):
    print(phase["talqin"], phase["ulush_foiz"], phase["ishonch"], phase["optik_belgi"])
fabric = olchovlar(cv2.imread("ppl.jpg"), cv2.imread("xpl.jpg"), seg)
print(fabric["struktura"], fabric["tekstura"], fabric["morfologiya"])
```

See `docs/USER_GUIDE.md` for all inputs, outputs and options.

## Reproducing the results of the paper

The tables of the paper can be reproduced **without the images** from the
per-pair results in `results/`:

```bash
python scripts/compute_metrics.py
```

This chooses the sample-level thresholds on the development partition
(maximum Cohen's κ), applies them unchanged to the Qorason test partition and
prints Table 2 (pre-processing, time), Table 3 (mineral groups), the fabric
agreement of Section 4.4 and the image-free baseline.

With access to the images, the per-pair results are regenerated with

```bash
python scripts/evaluate_pairs.py --pairs data/pairs_dev.csv  --image-root IMAGES --out results/per_pair_dev.csv
python scripts/evaluate_pairs.py --pairs data/pairs_test.csv --image-root IMAGES --out results/per_pair_test.csv
python scripts/make_figure3.py --ppl "IMAGES/new_sections/KN-15/KN-15-10(1).jpg" \
                               --xpl "IMAGES/new_sections/KN-15/KN-15+10(1).jpg" --out figures
```

Outputs are deterministic (fixed k-means seed); processing times depend on the machine.

## Contents

```
petrolog/            pipeline (imaging, fabric, reference labels)
scripts/             evaluate_pairs.py, compute_metrics.py, make_figure3.py
data/pairs_*.csv     image pairs of each partition (relative paths, nicol assignment)
data/sample_labels_*.csv  reference labels derived from the descriptions (0/1 flags)
results/per_pair_*.csv    pipeline output for every pair (inputs to compute_metrics.py)
results/thresholds_dev.json, results/test_metrics.json
docs/USER_GUIDE.md   inputs, outputs, options, glossary
tests/               smoke tests
```

## Data availability

Thin-section photographs and written petrographic descriptions belong to the
collections of the Institute of Geology and Geophysics (Tashkent) and are not
redistributed here; they are available from the author on reasonable request,
subject to institutional permission. The derived per-pair measurements and the
0/1 reference labels needed to reproduce every number in the paper are included.

## Identifier glossary

Identifiers and output labels are in Uzbek, the working language of the
application; all comments and documentation are in English. The most frequent
terms: `namuna` sample, `juft` pair, `faza` phase, `ulush_foiz` area %,
`talqin` interpretation (group label), `ishonch` confidence, `optik_belgi`
optical evidence, `izoh` note, `masshtab` scale, `mkm_px` µm per pixel,
`quti` box, `topildi` found, `don` grain, `median_diametr` median diameter,
`donlar_ajraldi` grains resolved, `yonalganlik` orientation (coherence),
`tomir` vein, `opak`/`ma'danli` opaque, `struktura` structure, `tekstura`
texture, `morfologiya` morphology, `kadr` panel, `bekor` excluded region.
English group names are listed in `petrolog/imaging/segment.py` (`GROUP_EN`).

## Licence and citation

MIT licence (see `LICENSE`). If you use this code, please cite the paper above
(see `CITATION.cff`).
