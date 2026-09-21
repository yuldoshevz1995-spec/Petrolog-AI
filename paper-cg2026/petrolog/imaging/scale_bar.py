"""Find the scale bar in a photomicrograph and compute the pixel size.

Archive images carry a scale label (e.g. "1 mm", "0,5 mm", "0,3 mm") inside a
white box, usually in the lower-left corner. The width of the box is taken as
the length of the scale bar.
"""

from __future__ import annotations

import os
import re
import subprocess
import tempfile
from collections import Counter
from pathlib import Path

import cv2
import numpy as np

# OCR may read digits as letters ("1 mm" -> "T mm", "I mm", "l mm"), so such
# letters are accepted in number positions and converted back to digits.
_RAQAM_BELGI = "0-9OoQDIilLtT|!sSzZbBgG"
# Path to the tesseract executable (override with the TESSERACT variable).
TESSERACT = os.environ.get("TESSERACT", "tesseract")

_MATN_RE = re.compile(
    rf"([{_RAQAM_BELGI}][{_RAQAM_BELGI}.,]*)\s*(mm|nm|mkm|um|µm|мм|мкм)",
    re.IGNORECASE,
)

# Letter -> digit substitutions (the confusions tesseract makes most often)
_ALMASH = {
    "o": "0", "q": "0", "d": "0",
    "i": "1", "l": "1", "t": "1", "|": "1", "!": "1",
    "s": "5", "z": "2", "b": "8", "g": "6",
}

# Plausible pixel size range for optical microscopy (micrometres per pixel)
MKM_PX_ORALIQ = (0.05, 60.0)


def _raqamga(xom: str) -> str:
    return "".join(_ALMASH.get(ch.lower(), ch) for ch in xom)


def parse_scale_text(matn: str) -> tuple[float, str] | None:
    """Parse an OCR string into a length in micrometres.

    "0,5 mm" -> (500.0, "0.5 mm"). If OCR drops the decimal separator
    ("03 mm"), a number starting with 0 is read as a decimal ("0.3").
    """
    m = _MATN_RE.search(matn)
    if not m:
        return None
    xom = _raqamga(m.group(1)).replace(",", ".").strip(".")
    if not xom or not any(ch.isdigit() for ch in xom):
        return None
    if "." not in xom and xom.startswith("0") and len(xom) > 1:
        xom = xom[0] + "." + xom[1:]
    try:
        qiymat = float(xom)
    except ValueError:
        return None
    if qiymat <= 0:
        return None
    birlik = m.group(2).lower()
    # "nm" is an OCR misreading of "mm"; nanometres do not occur here
    if birlik in ("mm", "nm", "мм"):
        mkm = qiymat * 1000
        birlik_nom = "mm"
    else:
        mkm = qiymat
        birlik_nom = "mkm"
    son = f"{qiymat:g}"
    return mkm, f"{son} {birlik_nom}"


# Standard scale-bar lengths used in petrography (micrometres). Readings that
# are not in this set (e.g. "11 mm" from an OCR error) are rejected.
STANDART_MKM = {10, 20, 25, 50, 100, 200, 250, 300, 500, 1000, 2000, 5000}


def _ocr(rasm: np.ndarray, psm: str, oq_royxat: bool) -> str:
    """Run tesseract on one image crop and return the raw text."""
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
        yol = f.name
    cv2.imwrite(yol, rasm)
    buyruq = [TESSERACT, yol, "-", "--psm", psm]
    if oq_royxat:
        buyruq += ["-c", "tessedit_char_whitelist=0123456789.,mkuµ "]
    try:
        chiqish = subprocess.run(buyruq, capture_output=True, text=True, timeout=30).stdout
    except FileNotFoundError:
        # tesseract not installed: the label cannot be read, but this is not
        # an error; sizes are then reported in pixels.
        chiqish = ""
    except subprocess.TimeoutExpired:
        chiqish = ""
    finally:
        Path(yol).unlink(missing_ok=True)
    return chiqish


def _tayyorlash(kesim: np.ndarray) -> np.ndarray:
    """Enlarge a small label crop and binarise it for OCR."""
    h, w = kesim.shape[:2]
    katta = cv2.resize(kesim, (w * 8, h * 8), interpolation=cv2.INTER_CUBIC)
    katta = cv2.copyMakeBorder(katta, 40, 40, 40, 40, cv2.BORDER_CONSTANT, value=255)
    _, katta = cv2.threshold(katta, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return katta


def _ovoz_berish(kesim: np.ndarray, quti_eni: int) -> tuple[float, int] | None:
    """Read one candidate box several times and take a majority vote.

    Three crops x three page-segmentation modes x with/without a character
    whitelist are read; only standard lengths that give a plausible pixel
    size are counted. Returns (micrometres, number of votes) or None.
    """
    ovoz: Counter = Counter()
    h = kesim.shape[0]
    variantlar = [kesim, kesim[: int(h * 0.8)], kesim[int(h * 0.2):]]
    for v in variantlar:
        if v.shape[0] < 6:
            continue
        tayyor = _tayyorlash(v)
        for psm in ("7", "6", "13"):
            for oq in (True, False):
                natija = parse_scale_text(_ocr(tayyor, psm, oq))
                if not natija:
                    continue
                mkm = natija[0]
                if mkm not in STANDART_MKM:
                    continue
                if not (MKM_PX_ORALIQ[0] <= mkm / quti_eni <= MKM_PX_ORALIQ[1]):
                    continue
                ovoz[mkm] += 1
        if ovoz and ovoz.most_common(1)[0][1] >= 3:
            break   # reliable result; the remaining crops need not be read
    if not ovoz:
        return None
    return ovoz.most_common(1)[0]


def _matn(mkm: float) -> str:
    return f"{mkm / 1000:g} mm" if mkm >= 1000 else f"{mkm:g} mkm"


def _nomzodlar(soha: np.ndarray) -> list[tuple[int, int, int, int, int]]:
    """Find rectangles in a grey-scale region that may be scale boxes.

    Two searches are combined:
      1. white rectangles at several thresholds (so that the box does not
         merge with neighbouring white grains in a bright image);
      2. dark frames (a white box with a black border cannot be separated
         from white minerals by its interior, but its frame can).
    Returns (area, x, y, w, h) tuples, largest first, without duplicates.
    """
    nomzodlar = []

    def qosh_nomzod(x, y, cw, ch):
        if cw < 25 or ch < 7 or ch > 70:
            return
        if not (1.8 < cw / ch < 14):
            return
        nomzodlar.append((cw * ch, x, y, cw, ch))

    for bosaga in (200, 225, 245):
        _, bin_ = cv2.threshold(soha, bosaga, 255, cv2.THRESH_BINARY)
        bin_ = cv2.morphologyEx(bin_, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
        konturlar, _ = cv2.findContours(bin_, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for c in konturlar:
            qosh_nomzod(*cv2.boundingRect(c))

    # dark frame: a rectangle with a hollow (low-fill) outline
    _, qora = cv2.threshold(soha, 90, 255, cv2.THRESH_BINARY_INV)
    qora = cv2.morphologyEx(qora, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
    konturlar, _ = cv2.findContours(qora, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for c in konturlar:
        x, y, cw, ch = cv2.boundingRect(c)
        if cw * ch == 0:
            continue
        if cv2.contourArea(c) / (cw * ch) > 0.45:
            continue     # a filled blob, not a frame
        # The bar length is the width of the white interior, not of the frame
        ichki = soha[y:y + ch, x:x + cw]
        _, oq = cv2.threshold(ichki, 170, 255, cv2.THRESH_BINARY)
        ich_kont, _ = cv2.findContours(oq, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if ich_kont:
            eng = max(ich_kont, key=cv2.contourArea)
            ix, iy, iw, ih = cv2.boundingRect(eng)
            qosh_nomzod(x + ix, y + iy, iw, ih)
        else:
            qosh_nomzod(x + 2, y + 2, max(cw - 4, 1), max(ch - 4, 1))

    nomzodlar.sort(reverse=True)
    # drop near-duplicates (similar position and width)
    tanlangan = []
    for n in nomzodlar:
        if any(abs(n[1] - t[1]) < 6 and abs(n[3] - t[3]) < 6 for t in tanlangan):
            continue
        tanlangan.append(n)
    return tanlangan


def detect_scale(path: str | Path) -> dict:
    """Detect the scale bar of an image file.

    Returns {"topildi" (found), "matn" (label), "masshtab_mkm" (bar length,
    µm), "quti_px" (bar length, px), "mkm_px" (µm per pixel), "quti" (box
    x, y, w, h in the full image), "ishonch" ("yuqori" = two or more
    concordant readings, "past" = a single reading)}.
    """
    img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        return {"topildi": False, "sabab": "image could not be read"}
    h, w = img.shape

    # Search order: lower-left, whole lower strip, upper-left
    sohalar = [
        (img[int(h * 0.75):, : int(w * 0.55)], 0, int(h * 0.75)),
        (img[int(h * 0.70):, :], 0, int(h * 0.70)),
        (img[: int(h * 0.25), : int(w * 0.55)], 0, 0),
    ]

    korilgan: set[tuple[int, int, int, int]] = set()
    for soha, siljish_x, siljish_y in sohalar:
        for _, x, y, cw, ch in _nomzodlar(soha)[:4]:
            kalit = (x + siljish_x, y + siljish_y, cw, ch)
            if kalit in korilgan:
                continue
            korilgan.add(kalit)

            javob = _ovoz_berish(soha[y:y + ch, x:x + cw], cw)
            if not javob:
                continue
            mkm, ovozlar = javob
            return {
                "topildi": True,
                "matn": _matn(mkm),
                "masshtab_mkm": mkm,
                "quti_px": cw,
                "mkm_px": round(mkm / cw, 4),
                # box position in the full image; excluded from segmentation
                "quti": [x + siljish_x, y + siljish_y, cw, ch],
                "ishonch": "yuqori" if ovozlar >= 2 else "past",
            }

    return {"topildi": False, "sabab": "no scale box found or label not readable"}
