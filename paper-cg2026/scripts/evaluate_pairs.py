"""Run the pipeline on every PPL/XPL pair of a partition and write per-pair results.

Usage:
    python scripts/evaluate_pairs.py --pairs data/pairs_test.csv \
        --image-root /path/to/images --out results/per_pair_test.csv

The pair list gives image paths relative to --image-root and the nicol
assignment made by the pre-processing step. The run can be interrupted and
resumed: pairs already present in the output file are skipped.
"""
import argparse, csv, sys, time
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from petrolog.imaging import detect_scale, modal_tarkib, segment_juft  # noqa: E402
from petrolog.fabric import olchovlar                                  # noqa: E402
from petrolog.labels import GURUHLAR, program_group                    # noqa: E402


def analyse_pair(ppl: Path, xpl: Path) -> dict:
    t0 = time.perf_counter()
    # 1. scale: PPL first, then XPL
    mas = detect_scale(ppl)
    if not mas.get("topildi"):
        mas = detect_scale(xpl)
    mkm_px = mas.get("mkm_px") if mas.get("topildi") else None
    bekor = [mas["quti"]] if mas.get("quti") else None
    # 2. joint segmentation and rule-based groups
    seg = segment_juft(ppl, xpl, k=7, mkm_px=mkm_px, bekor=bekor)
    seg["modal"] = modal_tarkib(seg)
    # 3. fabric
    o = olchovlar(cv2.imread(str(ppl)), cv2.imread(str(xpl)), seg)
    sec = time.perf_counter() - t0

    share = {g: 0.0 for g in GURUHLAR + ["OTHER"]}
    review = fine = 0.0
    for f in seg["fazalar"]:
        share[program_group(f["talqin"])] += f["ulush_foiz"]
        if f["ishonch"] < 0.5:
            review += f["ulush_foiz"]
        if "juda mayda" in f.get("olcham_sababi", ""):
            fine += f["ulush_foiz"]
    sized = [f for f in seg["fazalar"] if f.get("median_diametr")]
    sizes = [f["median_diametr"] for f in sized]
    main = max(sized, key=lambda f: f["ulush_foiz"]) if sized else {}
    return {
        "nisbat": round(max(sizes) / max(min(sizes), 1e-6), 2) if sizes else "",
        "asos_d": main.get("median_diametr", ""),
        "asos_ajraldi": int(bool(main.get("donlar_ajraldi"))) if main else "",
        "birlik": main.get("birlik", ""),
        "masshtab": int(bool(mas.get("topildi"))), "masshtab_matn": mas.get("matn", ""),
        "mkm_px": mkm_px or "", "masshtab_ishonch": mas.get("ishonch", ""),
        **{f"u_{g}": round(v, 2) for g, v in share.items()},
        "tekshiruv_ulushi": round(review, 1), "mayda_ulushi": round(fine, 1),
        "struktura": o["struktura"]["atama"], "tekstura": o["tekstura"]["atama"],
        "morfologiya": o["morfologiya"]["atama"],
        "yonalganlik": o["yonalganlik"]["koeffitsient"],
        "tomir_ulushi": o["tomirlar"]["tomir_ulushi"],
        "opak_ulush": round(o["opak_ulush"] * 100, 2),
        "sekund": round(sec, 2),
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--pairs", required=True)
    ap.add_argument("--image-root", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    root, out = Path(a.image_root), Path(a.out)
    done = set()
    if out.exists():
        done = {(r["namuna"], r["juft"]) for r in csv.DictReader(open(out, encoding="utf-8-sig"))}
    rows = list(csv.DictReader(open(a.pairs, encoding="utf-8")))
    for i, r in enumerate(rows, 1):
        if (r["sample"], r["pair"]) in done:
            continue
        res = analyse_pair(root / r["ppl"], root / r["xpl"])
        rec = {"namuna": r["sample"], "juft": r["pair"],
               "ppl": Path(r["ppl"]).name, "xpl": Path(r["xpl"]).name,
               "haqiqiy_xpl": r["xpl_label_known_correct"], **res}
        new = not out.exists()
        with open(out, "a", newline="", encoding="utf-8-sig") as f:
            w = csv.DictWriter(f, fieldnames=list(rec)); 
            if new:
                w.writeheader()
            w.writerow(rec)
        print(f"[{i}/{len(rows)}] {r['sample']} {r['pair']} {res['sekund']} s", flush=True)


if __name__ == "__main__":
    main()
