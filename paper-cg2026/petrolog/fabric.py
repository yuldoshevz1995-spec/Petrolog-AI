"""Fabric measurements: orientation, veins, grain shape and the derived
structure, texture and morphology terms (Section 3.5 of the paper).

Every term is derived from a measured value, and the value is returned with
the term so that the report can be traced back to it. Output terms are in
Uzbek (the application language); English equivalents are given in comments.
"""

from __future__ import annotations

import cv2
import numpy as np


def yonalganlik(kul: np.ndarray) -> dict:
    """Degree of preferred orientation from the structure tensor.

    Sobel gradients -> products smoothed with a 15 x 15 Gaussian -> local
    orientation and strength -> strength-weighted mean of doubled angles.
    Returns coherence (0 = random, 1 = perfectly aligned) and the mean
    direction in degrees (Eq. 3 of the paper).
    """
    kul = cv2.GaussianBlur(kul.astype(np.float32), (5, 5), 0)
    gx = cv2.Sobel(kul, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(kul, cv2.CV_32F, 0, 1, ksize=3)

    jxx = cv2.GaussianBlur(gx * gx, (15, 15), 0)
    jyy = cv2.GaussianBlur(gy * gy, (15, 15), 0)
    jxy = cv2.GaussianBlur(gx * gy, (15, 15), 0)

    # local orientation angle and its strength
    burchak = 0.5 * np.arctan2(2 * jxy, jxx - jyy)
    kuch = np.sqrt((jxx - jyy) ** 2 + 4 * jxy ** 2)

    ogirlik = kuch / (kuch.sum() + 1e-9)
    # orientations are 180-degree periodic: average doubled angles
    c = float((ogirlik * np.cos(2 * burchak)).sum())
    s = float((ogirlik * np.sin(2 * burchak)).sum())
    koeff = float(np.hypot(c, s))
    asosiy_burchak = float(np.degrees(0.5 * np.arctan2(s, c)) % 180)
    return {"koeffitsient": round(koeff, 3), "burchak": round(asosiy_burchak, 1)}


def tomirlar(maska: np.ndarray) -> dict:
    """Vein-like components of a binary (opaque) mask.

    A component of at least 40 px with length/width >= 3.5 and bounding-box
    fill < 0.55 is counted as vein-like. Returns the vein share of the mask
    area, the largest aspect ratio and the number of components.
    """
    son, teg, stat, _ = cv2.connectedComponentsWithStats(maska.astype(np.uint8), 8)
    if son <= 1:
        return {"tomir_ulushi": 0.0, "eng_uzun_nisbat": 0.0, "bolaklar": 0}

    jami_yuza = float(maska.sum()) or 1.0
    tomir_yuza = 0.0
    eng_nisbat = 0.0
    for i in range(1, son):
        x, y, w, h, yuza = stat[i]
        if yuza < 40:
            continue
        uzun, qisqa = max(w, h), max(min(w, h), 1)
        nisbat = uzun / qisqa
        # elongated and sparsely filled -> vein-like
        if nisbat >= 3.5 and yuza / (w * h) < 0.55:
            tomir_yuza += yuza
        eng_nisbat = max(eng_nisbat, nisbat)
    return {"tomir_ulushi": round(tomir_yuza / jami_yuza, 3),
            "eng_uzun_nisbat": round(float(eng_nisbat), 1),
            "bolaklar": int(son - 1)}


def don_shakli(maska: np.ndarray, eng_kam: int = 60) -> dict:
    """Grain-shape statistics: convexity, elongation and polygon vertices.

    Convexity = area / convex-hull area (high for euhedral grains, low for
    anhedral, interstitial grains). Medians over grains >= eng_kam px.
    """
    konturlar, _ = cv2.findContours(maska.astype(np.uint8), cv2.RETR_EXTERNAL,
                                    cv2.CHAIN_APPROX_SIMPLE)
    toldirilganlik, choziqlik, burchaklar = [], [], []
    for c in konturlar:
        yuza = cv2.contourArea(c)
        if yuza < eng_kam:
            continue
        qavariq = cv2.convexHull(c)
        qavariq_yuza = cv2.contourArea(qavariq)
        if qavariq_yuza > 0:
            toldirilganlik.append(yuza / qavariq_yuza)
        if len(c) >= 5:
            (_, _), (a, b), _ = cv2.minAreaRect(c)
            if min(a, b) > 0:
                choziqlik.append(max(a, b) / min(a, b))
        # number of polygon vertices: straight faces -> few vertices
        taxmin = cv2.approxPolyDP(c, 0.02 * cv2.arcLength(c, True), True)
        burchaklar.append(len(taxmin))

    if not toldirilganlik:
        return {"toldirilganlik": None, "choziqlik": None, "donalar": 0,
                "burchaklar": None}
    return {
        "toldirilganlik": round(float(np.median(toldirilganlik)), 3),
        "choziqlik": round(float(np.median(choziqlik)), 2) if choziqlik else None,
        "burchaklar": round(float(np.median(burchaklar)), 1) if burchaklar else None,
        "donalar": len(toldirilganlik),
    }


# --- Converting measurements into petrographic terms ---------------------

def struktura_atamasi(olchamlar: list[float], mkm_px: float | None) -> dict:
    """Structure term from the grain sizes of the phases.

    ratio (coarsest / finest phase) >= 3 -> porphyritic ("porfir");
    1.8-3 -> seriate / heteroblastic ("geteroblastik");
    otherwise, with a known scale and median < 50 µm -> fine-grained
    groundmass ("mayda donador"); otherwise equigranular ("tengdonador").
    """
    olchamlar = [o for o in olchamlar if o]
    if not olchamlar:
        return {"atama": "aniqlanmadi", "sabab": "don o'lchami o'lchanmadi"}

    eng_katta, eng_kichik = max(olchamlar), min(olchamlar)
    nisbat = eng_katta / max(eng_kichik, 1e-6)
    ortacha = float(np.median(olchamlar))

    birlik = "mkm" if mkm_px else "piksel"
    if nisbat >= 3.0:
        return {"atama": "porfir (notekis donador)",
                "sabab": f"yirik va mayda donalar farqi {nisbat:.1f} baravar",
                "birlik": birlik}
    if nisbat >= 1.8:
        return {"atama": "geteroblastik (har xil o'lchamli)",
                "sabab": f"don o'lchamlari farqi {nisbat:.1f} baravar",
                "birlik": birlik}
    if mkm_px and ortacha < 50:
        return {"atama": "mayda donador (afanit asos massa)",
                "sabab": f"o'rtacha don {ortacha:.0f} mkm", "birlik": birlik}
    return {"atama": "tengdonador (bir xil o'lchamli)",
            "sabab": f"don o'lchamlari farqi atigi {nisbat:.1f} baravar",
            "birlik": birlik}


def tekstura_atamasi(yonalish: dict, tomir: dict, opak_ulush: float) -> dict:
    """Texture term from orientation, veins and opaque-mineral share.

    coherence >= 0.30 -> oriented ("slanesli"); vein share >= 0.15 -> veined
    ("tomirli"); opaque 1-15 % without veins -> disseminated ("sochma");
    opaque >= 35 % -> massive ore; none of these -> massive ("massiv").
    """
    belgilar = []
    if yonalish["koeffitsient"] >= 0.30:
        belgilar.append(("slanesli (yo'nalgan)",
                         f"yo'nalganlik {yonalish['koeffitsient']:.2f}, "
                         f"asosiy yo'nalish {yonalish['burchak']:.0f}°"))
    if tomir["tomir_ulushi"] >= 0.15:
        belgilar.append(("tomirli",
                         f"cho'zilgan sohalar ulushi {tomir['tomir_ulushi']:.0%}"))
    if 0.01 <= opak_ulush < 0.15 and tomir["tomir_ulushi"] < 0.15:
        belgilar.append(("sochma (vkraplennik)",
                         f"ma'danli donalar tarqoq, ulushi {opak_ulush:.1%}"))
    if opak_ulush >= 0.35:
        belgilar.append(("massiv (zich ma'danli)",
                         f"ma'danli minerallar ulushi {opak_ulush:.0%}"))
    if not belgilar:
        belgilar.append(("massiv (bir jinsli)",
                         f"yo'nalganlik past ({yonalish['koeffitsient']:.2f}), "
                         "tomir va sochma belgilari sezilmadi"))
    return {"atama": " + ".join(b[0] for b in belgilar),
            "sabab": "; ".join(b[1] for b in belgilar)}


def morfologiya_atamasi(shakl: dict) -> dict:
    """Morphology term from convexity and elongation.

    convexity >= 0.93 and elongation < 1.8 -> euhedral ("idiomorf");
    convexity >= 0.85 -> subhedral ("gipidiomorf"); else anhedral
    ("ksenomorf"); elongation >= 2.2 adds "elongated" ("cho'zilgan").
    """
    t = shakl.get("toldirilganlik")
    if t is None:
        return {"atama": "aniqlanmadi", "sabab": "o'lchov uchun don topilmadi"}
    choziq = shakl.get("choziqlik") or 1.0
    if t >= 0.93 and choziq < 1.8:
        atama = "idiomorf (o'z shakliga ega, tekis qirrali)"
    elif t >= 0.85:
        atama = "gipidiomorf (qisman shakllangan)"
    else:
        atama = "ksenomorf (shaklsiz, bo'shliqni to'ldirgan)"
    if choziq >= 2.2:
        atama += ", cho'zilgan"
    return {"atama": atama,
            "sabab": f"to'ldirilganlik {t:.2f}, cho'ziqlik {choziq:.1f}, "
                     f"{shakl['donalar']} dona o'lchandi"}


# --- Fabric measurements of one image pair (from petrolog.petrography.tavsif)

def _opak_maska(ppl: np.ndarray, xpl: np.ndarray | None = None) -> np.ndarray:
    """Opaque-mineral mask: pixels darker than 0.20 in PPL and, if an XPL
    image is given, also in XPL. Requiring darkness in both nicols prevents
    brown, fine-grained groundmass (dark in PPL, bright in XPL) from being
    counted as opaque."""
    kul = cv2.cvtColor(ppl, cv2.COLOR_BGR2GRAY)
    maska = kul < 0.20 * 255
    if xpl is not None:
        kx = cv2.cvtColor(xpl, cv2.COLOR_BGR2GRAY)
        if kx.shape != kul.shape:
            kx = cv2.resize(kx, (kul.shape[1], kul.shape[0]), interpolation=cv2.INTER_AREA)
        maska &= kx < 0.20 * 255
    return maska.astype(np.uint8)


def _yirik_donalar_maskasi(kul: np.ndarray) -> np.ndarray:
    """Bright grains (Otsu threshold), used for shape when opaque grains are few."""
    silliq = cv2.GaussianBlur(kul, (5, 5), 0)
    _, niqob = cv2.threshold(silliq, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return (niqob > 0).astype(np.uint8)


def olchovlar(ppl: np.ndarray, xpl: np.ndarray | None, segment: dict) -> dict:
    """All fabric measurements and terms for one PPL (+ XPL) image pair.

    segment: output of segment_juft / segment_kmeans (phase grain sizes are
    taken from it). Orientation is measured on the PPL grey-scale image.
    """
    kul = cv2.cvtColor(ppl, cv2.COLOR_BGR2GRAY)
    opak = _opak_maska(ppl, xpl)
    opak_ulush = float(opak.mean())

    yon = yonalganlik(kul)
    tom = tomirlar(opak)
    shakl = don_shakli(opak if opak.sum() > 500 else _yirik_donalar_maskasi(kul))

    olchamlar = [f.get("median_diametr") for f in segment.get("fazalar", [])]
    struk = struktura_atamasi(olchamlar, segment.get("mkm_px"))
    tekst = tekstura_atamasi(yon, tom, opak_ulush)
    morf = morfologiya_atamasi(shakl)

    return {
        "opak_ulush": round(opak_ulush, 4),
        "yonalganlik": yon,
        "tomirlar": tom,
        "don_shakli": shakl,
        "struktura": struk,
        "tekstura": tekst,
        "morfologiya": morf,
    }
