"""Reproduce the evaluation tables of the paper (Tables 2 and 3, Section 4.4).

Usage:
    python scripts/compute_metrics.py            # uses results/ and data/

Sample-level thresholds are chosen on the development partition by
maximising Cohen's kappa over a fixed grid and then applied unchanged to the
test partition (Section 4.1).
"""
import csv, json, statistics as st, sys
from collections import defaultdict
from itertools import product
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from petrolog.labels import GURUHLAR as G  # noqa: E402

GROUP_GRID = (1, 2, 5, 10, 15, 20, 30, 40)            # % modal share
FABRIC_RULES = {
    # feature: (decision function, grid of thresholds)
    "porfir": (lambda v, t: v["ratio"] >= t[0], [(x,) for x in (2, 3, 4, 5, 6, 8, 10, 15)]),
    "mayda_asos": (lambda v, t: (v["main_d"] < t[0]) or v["unresolved"] >= t[1],
                   list(product((20, 30, 50, 75, 100, 150), (0.5, 0.75, 1.01)))),
    "yonalgan": (lambda v, t: v["coh"] >= t[0], [(x,) for x in (0.1, 0.15, 0.2, 0.25, 0.3, 0.4)]),
    "tomir": (lambda v, t: v["vein"] >= t[0] and v["opaque"] >= t[1],
              list(product((0.05, 0.1, 0.15, 0.3), (0.2, 0.5, 1.0, 2.0)))),
}


def f(x, d=float("nan")):
    try:
        return float(x)
    except (TypeError, ValueError):
        return d


def prf(pairs):
    """Precision, recall, F1, accuracy and Cohen's kappa for (reference, prediction) pairs."""
    tp = sum(g and d for g, d in pairs); fp = sum((not g) and d for g, d in pairs)
    fn = sum(g and not d for g, d in pairs); tn = sum((not g) and (not d) for g, d in pairs)
    n = len(pairs)
    p = tp / (tp + fp) if tp + fp else float("nan")
    r = tp / (tp + fn) if tp + fn else float("nan")
    f1 = 2 * p * r / (p + r) if p == p and r == r and p + r else float("nan")
    acc = (tp + tn) / n if n else float("nan")
    pe = ((tp + fp) * (tp + fn) + (fn + tn) * (fp + tn)) / n ** 2 if n else 0
    kappa = (acc - pe) / (1 - pe) if n and pe < 1 else float("nan")
    r2 = lambda x: round(x, 2)
    return dict(n=n, tp=tp, fp=fp, fn=fn, tn=tn, P=r2(p), R=r2(r), F1=r2(f1), acc=r2(acc), kappa=r2(kappa))


def samples(part):
    rows = list(csv.DictReader(open(ROOT / f"results/per_pair_{part}.csv", encoding="utf-8-sig")))
    lab = {r["sample"]: r for r in csv.DictReader(open(ROOT / f"data/sample_labels_{part}.csv", encoding="utf-8"))}
    by = defaultdict(list)
    for r in rows:
        by[r["namuna"]].append(r)
    out = {}
    for k, rr in by.items():
        med = lambda c: st.median([f(r[c]) for r in rr if f(r[c]) == f(r[c])] or [float("nan")])
        um = [f(r["asos_d"]) for r in rr if r["birlik"] == "mkm" and f(r["asos_d"]) == f(r["asos_d"])]
        L = lab.get(k, {})
        out[k] = {
            "series": "KN" if k.startswith("QN") else k[:2], "pairs": rr,
            "share": {g: st.mean(f(r[f"u_{g}"]) for r in rr) for g in G},
            "ratio": med("nisbat"), "coh": med("yonalganlik"), "vein": med("tomir_ulushi"),
            "opaque": med("opak_ulush"), "main_d": st.median(um) if um else float("nan"),
            "unresolved": sum(r["asos_ajraldi"] == "0" for r in rr) / len(rr),
            "has_list": L.get("has_mineral_list") == "1", "has_desc": L.get("has_description") == "1",
            "ref_g": {g: int(L.get(f"ref_{g}", 0)) for g in G},
            "ref_f": {b: int(L.get(f"ref_{b}", 0)) for b in FABRIC_RULES},
        }
    return out


def choose(dev):
    lst = [v for v in dev.values() if v["has_list"]]; dsc = [v for v in dev.values() if v["has_desc"]]
    k = lambda m: m["kappa"] if m["kappa"] == m["kappa"] else -9
    th = {"groups": {}, "fabric": {}}
    for g in G:
        th["groups"][g] = max(GROUP_GRID, key=lambda t: k(prf([(v["ref_g"][g], int(v["share"][g] >= t)) for v in lst])))
    for b, (fn, grid) in FABRIC_RULES.items():
        th["fabric"][b] = list(max(grid, key=lambda t: k(prf([(v["ref_f"][b], int(fn(v, t))) for v in dsc]))))
    return th


def report(ns, th, title):
    lst = [v for v in ns.values() if v["has_list"]]; dsc = [v for v in ns.values() if v["has_desc"]]
    pairs = [r for v in ns.values() for r in v["pairs"]]
    print(f"\n== {title}: {len(ns)} samples, {len(pairs)} pairs, {len(lst)} with mineral list, {len(dsc)} with description")
    sc = [int(r["masshtab"]) for r in pairs]
    print(f"scale recovered {sum(sc)}/{len(sc)} ({sum(sc)/len(sc):.1%}); median time {st.median(f(r['sekund']) for r in pairs):.1f} s; "
          f"median area to expert {st.median(f(r['tekshiruv_ulushi']) for r in pairs):.0f} %")
    kx = [int(r["haqiqiy_xpl"]) for r in pairs if r["haqiqiy_xpl"] not in ("", None)]
    if kx:
        print(f"nicol order correct {sum(kx)}/{len(kx)}")
    for g in G:
        m = prf([(v["ref_g"][g], int(v["share"][g] >= th["groups"][g])) for v in lst])
        print(f"  {g:4s} pos {m['tp']+m['fn']:3d}/{m['n']}  P {m['P']}  R {m['R']}  F1 {m['F1']}  kappa {m['kappa']}")
    m = prf([(v["ref_g"][g], int(v["share"][g] >= th["groups"][g])) for v in lst for g in G])
    print(f"  micro P {m['P']} R {m['R']} F1 {m['F1']} acc {m['acc']} kappa {m['kappa']}")
    for b, (fn, _) in FABRIC_RULES.items():
        m = prf([(v["ref_f"][b], int(fn(v, th["fabric"][b]))) for v in dsc])
        print(f"  {b:10s} pos {m['tp']+m['fn']}/{m['n']}  acc {m['acc']}  kappa {m['kappa']}")


def baseline(dev, test):
    dl = [v for v in dev.values() if v["has_list"]]
    pick = {g: int(sum(v["ref_g"][g] for v in dl) / len(dl) > 0.5) for g in G}
    tl = [v for v in test.values() if v["has_list"]]
    m = prf([(v["ref_g"][g], pick[g]) for v in tl for g in G])
    print(f"\nimage-free baseline {pick}: micro P {m['P']} R {m['R']} F1 {m['F1']} acc {m['acc']}")


if __name__ == "__main__":
    dev, test = samples("dev"), samples("test")
    th = choose(dev)
    print("thresholds chosen on development set:", th)
    report(dev, th, "DEVELOPMENT (tuned, optimistic)")
    for s in ("KH", "KN"):
        report({k: v for k, v in test.items() if v["series"] == s}, th, f"TEST {s}")
    report(test, th, "TEST KH+KN")
    baseline(dev, test)
