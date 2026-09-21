"""Reproduce Fig. 3 and Table 4 of the paper (sample QN-15, objective 10x).

Usage:
    python scripts/make_figure3.py --ppl KN-15-10(1).jpg --xpl KN-15+10(1).jpg --out figures/
Writes Figure_3.png (600 dpi) and QN15_phases.csv.
"""
import argparse, csv, sys
from pathlib import Path

import cv2
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from petrolog.imaging import segment as S      # noqa: E402
from petrolog.labels import program_group      # noqa: E402

COLOURS = {"QF": (0.93, 0.93, 0.90), "MAF": (0.10, 0.62, 0.35), "HIB": (0.90, 0.55, 0.10),
           "OPQ": (0.05, 0.05, 0.05), "ISO": (0.55, 0.35, 0.75), "OTHER": (0.6, 0.6, 0.6)}
NAMES = {"QF": "Quartz / feldspar", "MAF": "Coloured mafic", "HIB": "High birefringence",
         "OPQ": "Opaque", "ISO": "Isotropic / extinct", "OTHER": "Other"}


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--ppl", required=True); ap.add_argument("--xpl", required=True)
    ap.add_argument("--out", default="figures"); ap.add_argument("--k", type=int, default=7)
    a = ap.parse_args()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)

    ppl, _ = S._oqish(a.ppl); xpl, _ = S._oqish(a.xpl)
    if xpl.shape != ppl.shape:
        xpl = cv2.resize(xpl, (ppl.shape[1], ppl.shape[0]), interpolation=cv2.INTER_AREA)
    X = np.hstack([S._belgilar(ppl), S._belgilar(xpl)])            # Eq. (1)
    lab = S._silliqlash(S._kmeans(X, a.k).reshape(ppl.shape[:2]), a.k)  # Eq. (2)

    rows, gmap = [], np.zeros((*lab.shape, 3))
    for i in range(a.k):
        m = lab == i
        if not m.any():
            continue
        sp, sx = S._faza_statistikasi(ppl, m), S._faza_statistikasi(xpl, m)
        t = S._talqin_juft(sp["V"], sp["S"], sp["H"], sx["V"], sx["S"])
        g = program_group(t["nom"]); gmap[m] = COLOURS[g]
        rows.append({"phase": i + 1, "area_pct": round(float(m.mean()) * 100, 1),
                     "Vp": round(sp["V"], 2), "Sp": round(sp["S"], 2), "Hp": round(sp["H"]),
                     "Vx": round(sx["V"], 2), "Sx": round(sx["S"], 2),
                     "group": NAMES[g], "confidence": t["ishonch"]})
    rows.sort(key=lambda r: -r["area_pct"])
    with open(out / "QN15_phases.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)

    rgb = lambda im: cv2.cvtColor(im, cv2.COLOR_BGR2RGB) / 255.0
    pal = np.array(S.PALITRA[:a.k]) / 255.0
    edge = np.abs(cv2.Laplacian(lab.astype(np.float32), cv2.CV_32F)) > 0.5
    over = 0.55 * rgb(xpl) + 0.45 * pal[lab]; over[edge] = 1.0
    fig, ax = plt.subplots(2, 2, figsize=(7.2, 6.3))
    for axis, im, title in zip(ax.flat, [rgb(ppl), rgb(xpl), over, gmap],
                               ["(a) Plane-polarised light", "(b) Cross-polarised light",
                                f"(c) k-means phases (k = {a.k}, mode filter)",
                                "(d) Mineral groups from optical rules"]):
        axis.imshow(im); axis.set_title(title, fontsize=8, loc="left"); axis.axis("off")
    present = [g for g in NAMES if any(r["group"] == NAMES[g] for r in rows)]
    ax[1, 1].legend(handles=[Patch(facecolor=COLOURS[g], edgecolor="0.4",
                    label=f"{NAMES[g]} {sum(r['area_pct'] for r in rows if r['group']==NAMES[g]):.0f} %")
                    for g in present], fontsize=6, loc="lower left", framealpha=0.9)
    plt.tight_layout(); fig.savefig(out / "Figure_3.png", dpi=600)
    for r in rows:
        print(r)


if __name__ == "__main__":
    main()
