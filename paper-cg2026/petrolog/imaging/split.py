"""Split composite images that hold two microscope frames side by side.

Archive photomicrographs are often stored as one file with two panels
(e.g. (a) plane-polarised and (b) cross-polarised light) separated by a
white or black seam. If such a file were analysed as a single image, the
two illumination modes would be mixed and all results would be wrong, so
the file is split first.

Method: homogeneity is measured along columns (or rows). The seam between
panels has an almost uniform colour over its full length and is detected
by that property.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

# The seam is searched for in this central part of the image (fractions)
QIDIRUV_ORALIGI = (0.30, 0.70)
# A panel narrower than this fraction is not accepted as a panel
ENG_KICHIK_KADR = 0.25


def _yol_nomzodlari(kul: np.ndarray, oq: int = 1) -> list[tuple[int, int]]:
    """Return runs of uniform columns (oq=1) or rows (oq=0) that may be seams."""
    if oq == 1:
        ogish = kul.std(axis=0)
        ortacha = kul.mean(axis=0)
        uzunlik = kul.shape[1]
    else:
        ogish = kul.std(axis=1)
        ortacha = kul.mean(axis=1)
        uzunlik = kul.shape[0]

    bosh = int(uzunlik * QIDIRUV_ORALIGI[0])
    oxir = int(uzunlik * QIDIRUV_ORALIGI[1])

    # Uniform: very low standard deviation; seam: very light or very dark
    bir_jinsli = (ogish < 12) & ((ortacha > 215) | (ortacha < 40))

    guruhlar = []
    boshlangan = None
    for i in range(bosh, oxir):
        if bir_jinsli[i]:
            if boshlangan is None:
                boshlangan = i
        elif boshlangan is not None:
            guruhlar.append((boshlangan, i - 1))
            boshlangan = None
    if boshlangan is not None:
        guruhlar.append((boshlangan, oxir - 1))
    return guruhlar


def kadrlarni_ajratish(yol: str | Path) -> dict:
    """Read an image file and split it into panels (see kadrlarga_bolish)."""
    img = cv2.imread(str(yol), cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError(f"image could not be read: {yol}")
    return kadrlarga_bolish(img)


def kadrlarga_bolish(img: np.ndarray) -> dict:
    """Split an in-memory BGR image into panels.

    Returns {"kadrlar": [panel, ...], "ajratildi": bool,
             "yonalish": "vertikal" | "gorizontal" | "vertikal (yo'lsiz)" | "",
             "yol_kengligi": seam width in px}.
    If no seam is found, a single-panel list is returned.
    """
    h, w = img.shape[:2]
    kul = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Wide images: look for a vertical seam; tall images: a horizontal one
    yonalishlar = [(1, "vertikal")] if w >= h else [(0, "gorizontal")]
    if 0.75 < w / h < 1.35:          # nearly square: try both directions
        yonalishlar = [(1, "vertikal"), (0, "gorizontal")]

    for oq, nom in yonalishlar:
        uzunlik = w if oq == 1 else h
        for bosh, oxir in sorted(_yol_nomzodlari(kul, oq),
                                 key=lambda g: -(g[1] - g[0])):
            markaz = (bosh + oxir) / 2 / uzunlik
            birinchi = bosh / uzunlik
            ikkinchi = 1 - oxir / uzunlik
            if birinchi < ENG_KICHIK_KADR or ikkinchi < ENG_KICHIK_KADR:
                continue
            if not (0.35 < markaz < 0.65):
                continue
            if oq == 1:
                kadrlar = [img[:, :bosh], img[:, oxir + 1:]]
            else:
                kadrlar = [img[:bosh, :], img[oxir + 1:, :]]
            return {"kadrlar": kadrlar, "ajratildi": True, "yonalish": nom,
                    "yol_kengligi": oxir - bosh + 1}

    # No seam found. In a very wide image the panels may touch: split at the
    # midline and accept the split only if the two halves differ strongly
    # (one light, the other darker and more saturated, i.e. different nicols).
    yarim_nisbat = (w / 2) / h
    if w / h >= 1.7 and 0.65 <= yarim_nisbat <= 1.7:
        chap, ong = img[:, : w // 2], img[:, w // 2:]
        if _keskin_farqmi(chap, ong):
            return {"kadrlar": [chap, ong], "ajratildi": True,
                    "yonalish": "vertikal (yo'lsiz)", "yol_kengligi": 0}
    return {"kadrlar": [img], "ajratildi": False, "yonalish": "", "yol_kengligi": 0}


def _keskin_farqmi(a: np.ndarray, b: np.ndarray) -> bool:
    """True if two halves look like the same field under different nicols."""
    def belgi(x):
        hsv = cv2.cvtColor(x, cv2.COLOR_BGR2HSV)
        return float(hsv[:, :, 2].mean()) / 255, float(hsv[:, :, 1].mean()) / 255

    v1, s1 = belgi(a)
    v2, s2 = belgi(b)
    # Brightness and saturation differences are combined: an XPL view is
    # usually both darker and more colourful than the PPL view.
    return abs(v1 - v2) + 1.5 * abs(s1 - s2) > 0.18


def fayllarga_yozish(natija: dict, asos: Path) -> list[Path]:
    """Write split panels to <base>_1.png, <base>_2.png and return the paths."""
    yollar = []
    for i, kadr in enumerate(natija["kadrlar"], 1):
        yol = asos.with_name(f"{asos.stem}_{i}.png")
        cv2.imwrite(str(yol), kadr)
        yollar.append(yol)
    return yollar
