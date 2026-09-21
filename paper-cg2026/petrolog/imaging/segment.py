"""Phase segmentation of thin-section images (k-means baseline).

Pixels are grouped into k clusters by colour; for every cluster the area
fraction, grain-size statistics and median colour are measured, and a mineral
group is proposed by ordered optical rules.

Important: k-means does NOT identify minerals. It only separates regions of
similar colour. Each proposed group therefore carries a confidence value, the
optical evidence used and a note on what the petrographer should check.

When a PPL and an XPL image of the same field are given, clustering uses the
colours of both images (six CIELAB channels per pixel).

Output labels (group names, notes) are in Uzbek, the language of the
application; English equivalents of the group labels are given in GROUP_EN.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

# Colours used to draw phases (Okabe-Ito palette, colour-blind safe)
PALITRA = [
    (0, 114, 178), (213, 94, 0), (0, 158, 115), (230, 159, 0),
    (204, 121, 167), (86, 180, 233), (120, 120, 120), (0, 0, 0),
]

MIN_DON_PIKSEL = 30   # connected components smaller than this (px) are treated as noise


def _oqish(yol: str | Path, max_tomon: int = 900) -> tuple[np.ndarray, float]:
    """Read an image and limit its longer side. Returns (BGR image, scale factor)."""
    img = cv2.imread(str(yol), cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError(f"image could not be read: {yol}")
    h, w = img.shape[:2]
    k = max_tomon / max(h, w)
    if k < 1:
        img = cv2.resize(img, (int(w * k), int(h * k)), interpolation=cv2.INTER_AREA)
        return img, k
    return img, 1.0


def _belgilar(bgr: np.ndarray) -> np.ndarray:
    """Per-pixel features: CIELAB after a 5 x 5 Gaussian blur."""
    lab = cv2.cvtColor(cv2.GaussianBlur(bgr, (5, 5), 0), cv2.COLOR_BGR2LAB)
    return lab.reshape(-1, 3).astype(np.float32)


def _kmeans(X: np.ndarray, k: int, urugh: int = 42) -> np.ndarray:
    shart = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.5)
    cv2.setRNGSeed(urugh)
    _, teg, _ = cv2.kmeans(X, k, None, shart, 5, cv2.KMEANS_PP_CENTERS)
    return teg.flatten()


def _silliqlash(teglar: np.ndarray, k: int, oyna: int | None = None) -> np.ndarray:
    """Remove isolated-pixel noise from a label map with a mode filter.

    k-means treats every pixel independently, so the raw map is speckled.
    Each pixel is replaced by the most frequent label in a square window
    whose side grows with image size: max(5, 2*floor(min(H, W)/90) + 1).
    """
    if oyna is None:
        # window scales with image size
        oyna = max(5, int(min(teglar.shape) / 90) * 2 + 1)
    ovoz = np.zeros((k, *teglar.shape), np.float32)
    yadro = (oyna, oyna)
    for i in range(k):
        ovoz[i] = cv2.blur((teglar == i).astype(np.float32), yadro)
    return ovoz.argmax(axis=0).astype(np.int32)


def _bekor_maska(shakl: tuple[int, int], bekor, kichraytirish: float) -> np.ndarray | None:
    """Mask of regions excluded from all statistics (scale box, labels).

    The white scale box would otherwise appear as a spurious "mineral" and
    distort area fractions. Boxes are given in full-resolution pixels and
    enlarged by 2 px on each side.
    """
    if not bekor:
        return None
    maska = np.zeros(shakl, bool)
    h, w = shakl
    for x, y, bw, bh in bekor:
        x0 = max(int(x * kichraytirish) - 2, 0)
        y0 = max(int(y * kichraytirish) - 2, 0)
        x1 = min(int((x + bw) * kichraytirish) + 2, w)
        y1 = min(int((y + bh) * kichraytirish) + 2, h)
        maska[y0:y1, x0:x1] = True
    return maska if maska.any() else None


def _don_olchami(maska: np.ndarray, mkm_px: float | None) -> dict:
    """Grain-size statistics of one phase from its connected components.

    The reported size is the area-weighted median equivalent diameter: half
    of the phase area lies in grains larger than this value, so that a few
    large phenocrysts are not hidden among thousands of small fragments.
    The size is flagged as unreliable ("donlar_ajraldi" = False) when the
    phase forms a connected network (largest component > 50 % of the area,
    or inscribed-circle / equivalent diameter < 0.55) or when the median is
    below 15 px (unresolved groundmass).
    """
    m8 = maska.astype(np.uint8)
    son, teg, stat, _ = cv2.connectedComponentsWithStats(m8, 8)
    barcha_yuza = stat[1:, cv2.CC_STAT_AREA]
    tanlov = barcha_yuza >= MIN_DON_PIKSEL
    yuzalar = barcha_yuza[tanlov].astype(np.float64)

    # Inscribed-circle radius separates elongated, connected regions from round grains
    ichki = np.zeros(son, np.float32)
    if yuzalar.size:
        masofa = cv2.distanceTransform(m8, cv2.DIST_L2, 3)
        np.maximum.at(ichki, teg.ravel(), masofa.ravel())
    ichki_d = (ichki[1:][tanlov] * 2).astype(np.float64)

    if yuzalar.size == 0:
        return {"don_soni": 0, "median_diametr": None, "max_diametr": None,
                "birlik": "-", "donlar_ajraldi": False,
                "olcham_sababi": "O'lchash uchun yetarli katta bo'lak topilmadi."}

    diametr_px = np.sqrt(4 * yuzalar / np.pi)
    tartib = np.argsort(diametr_px)
    d_sort, y_sort = diametr_px[tartib], yuzalar[tartib]
    ichki_sort = ichki_d[tartib]
    yigindi = np.cumsum(y_sort)
    orta = int(np.searchsorted(yigindi, yigindi[-1] / 2))
    median_px = float(d_sort[orta])
    # shape index: close to 1 for round grains, much smaller for networks
    yumaloqlik = float(ichki_sort[orta] / median_px) if median_px else 0.0

    # If the phase is connected (one component holds over half the area),
    # connected components cannot represent individual grains.
    eng_katta_ulush = float(y_sort[-1] / y_sort.sum())
    sabab = ""
    if eng_katta_ulush > 0.5 or yumaloqlik < 0.55:
        sabab = ("Faza tutash to'r hosil qilgan - alohida donlarga ajratish uchun "
                 "watershed yoki U-Net kerak. Ko'rsatilgan o'lcham donniki emas, "
                 "tutash sohaniki.")
    elif median_px < 15:
        sabab = ("Donlar tasvir ruxsatiga nisbatan juda mayda - bu asos massa "
                 "bo'lishi mumkin. Kattaroq kattalashtirishda suratga oling.")
    ajraldi = not sabab

    olcham = (lambda v: round(v * mkm_px, 1)) if mkm_px else (lambda v: round(v, 1))
    return {
        "don_soni": int(yuzalar.size),
        "median_diametr": olcham(median_px),
        "max_diametr": olcham(float(d_sort[-1])),
        "birlik": "mkm" if mkm_px else "piksel",
        "donlar_ajraldi": ajraldi,
        "shakl_korsatkichi": round(yumaloqlik, 2),
        "olcham_sababi": sabab,
    }


def _talqin_ppl(V: float, S: float, H: float) -> dict:
    """Group proposal from plane-polarised light only (fallback rule).

    Returns name (mineral group), confidence (0-1), optical evidence and note.
    Confidence reflects how specific the rule is: darkness is nearly
    unambiguous (high), absence of colour fits many minerals (low).
    """
    if V < 0.20:
        return {"nom": "ma'danli mineral yoki bo'shliq", "ishonch": 0.75,
                "optik_belgi": "bir nikolda qorong'i, yorug'lik o'tkazmaydi",
                "izoh": "Qaytgan yorug'likda tekshiring: anshlif kerak."}
    if S < 0.13:
        return {"nom": "rangsiz mineral: kvars, dala shpati, karbonat",
                "ishonch": 0.35, "optik_belgi": "rangsiz, o'z rangi yo'q",
                "izoh": "Ikki nikolsiz bularni ajratib bo'lmaydi."}
    if 10 <= H <= 40:
        return {"nom": "jigarrang-sariq: biotit, rutil, gidroksidlar",
                "ishonch": 0.45, "optik_belgi": f"o'z rangi bor, jigarrang-sariq ton",
                "izoh": "Pleoxroizmni tekshiring: biotitda kuchli."}
    if 40 < H <= 90:
        return {"nom": "yashil: xlorit, amfibol, epidot", "ishonch": 0.45,
                "optik_belgi": "o'z rangi bor, yashil ton",
                "izoh": "Pleoxroizm va interferentsiya rangi kerak."}
    if 90 < H <= 270:
        return {"nom": "ko'k-binafsha ton", "ishonch": 0.30,
                "optik_belgi": "ko'k-binafsha ton",
                "izoh": "Kam uchraydi: glaukofan yoki bo'yoq ta'siri."}
    return {"nom": "qizil-pushti ton", "ishonch": 0.30,
            "optik_belgi": "qizil-pushti ton",
            "izoh": "Gematit, granat yoki bo'yalgan smola bo'lishi mumkin."}


def _talqin_juft(Vp: float, Sp: float, Hp: float, Vx: float, Sx: float) -> dict:
    """Group proposal from a PPL + XPL pair (ordered rules, first match wins).

    The rules stop at the level of mineral groups; species-level names need
    extinction angle, twinning and relief, which the petrographer checks at
    the microscope. See Section 3.4 of the paper for the rule list.
    """
    if Vp < 0.20 and Vx < 0.20:
        return {"nom": "ma'danli mineral", "ishonch": 0.85,
                "optik_belgi": "ikkala nikolda ham qorong'i",
                "izoh": "Anshlifda qaytarish qobiliyati bo'yicha aniqlanadi."}
    if Vx < 0.30 and Vp > 0.45:
        return {"nom": "izotrop yoki so'nish holatidagi mineral", "ishonch": 0.55,
                "optik_belgi": "bir nikolda yorug', ikki nikolda qorong'i",
                "izoh": "Stolni buring: so'nsa anizotrop, doim qora bo'lsa "
                        "granat, shpinel yoki vulqon shishasi."}
    if Sx > 0.30 and Vx > 0.40:
        return {"nom": "yuqori interferentsiyali: karbonat, muskovit, epidot",
                "ishonch": 0.60,
                "optik_belgi": "ikki nikolda II-III tartib yorqin ranglar",
                "izoh": "Karbonatda relyef keskin o'zgaradi - shu bilan farqlang."}
    if Sp >= 0.15 and 10 <= Hp <= 90:
        return {"nom": "rangli mineral: biotit, xlorit, amfibol", "ishonch": 0.55,
                "optik_belgi": "bir nikolda o'z rangi bor",
                "izoh": "Pleoxroizmni tekshiring - biotit va amfibolda kuchli."}
    if Sp < 0.15 and Vx >= 0.55:
        return {"nom": "rangsiz: kvars, dala shpati yoki yuqori tartib oq",
                "ishonch": 0.40,
                "optik_belgi": "rangsiz, ikki nikolda yorug' oq",
                "izoh": "Karbonatda relyef keskin o'zgaradi, kvarsda yo'q."}
    if Sp < 0.15:
        return {"nom": "kvars, dala shpati (past interferentsiya)", "ishonch": 0.50,
                "optik_belgi": "rangsiz, ikki nikolda I tartib kulrang-oq",
                "izoh": "Plagioklazda polisintetik dvoyniklashuv bo'ladi - "
                        "kvarsda bo'lmaydi."}
    asos = _talqin_ppl(Vp, Sp, Hp)
    asos["izoh"] += " Ikki nikoldagi ko'rinishi bilan tasdiqlang."
    return asos


def _faza_statistikasi(bgr: np.ndarray, maska: np.ndarray) -> dict:
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    px = hsv[maska]
    o_rang = bgr[maska].mean(axis=0)
    return {
        "H": float(np.median(px[:, 0])) * 2.0,      # 0-360
        "S": float(np.median(px[:, 1])) / 255.0,
        "V": float(np.median(px[:, 2])) / 255.0,
        "rang_hex": "#%02x%02x%02x" % (int(o_rang[2]), int(o_rang[1]), int(o_rang[0])),
    }


def _qoplama(bgr: np.ndarray, teglar: np.ndarray, k: int, shaffoflik: float = 0.45,
             bekor_maska: np.ndarray | None = None) -> np.ndarray:
    rang_qatlam = np.zeros_like(bgr)
    for i in range(k):
        r, g, b = PALITRA[i % len(PALITRA)]
        rang_qatlam[teglar == i] = (b, g, r)
    qoplama = cv2.addWeighted(bgr, 1 - shaffoflik, rang_qatlam, shaffoflik, 0)
    # draw phase boundaries
    chegara = cv2.Laplacian(teglar.astype(np.float32), cv2.CV_32F)
    qoplama[np.abs(chegara) > 0.5] = (255, 255, 255)
    if bekor_maska is not None:
        qoplama[bekor_maska] = bgr[bekor_maska]   # excluded boxes stay unpainted
    return qoplama


def segment_kmeans(yol: str | Path, k: int = 6, mkm_px: float | None = None,
                   max_tomon: int = 900, bekor=None) -> dict:
    """Segment a single image into k phases.

    If mkm_px (micrometres per pixel) is given, grain sizes are in µm,
    otherwise in pixels. bekor: list of (x, y, w, h) boxes to exclude.
    """
    bgr, kichraytirish = _oqish(yol, max_tomon)
    if mkm_px:
        mkm_px = mkm_px / kichraytirish   # pixels are larger after downsampling
    teglar = _silliqlash(_kmeans(_belgilar(bgr), k).reshape(bgr.shape[:2]), k)
    bekor_maska = _bekor_maska(bgr.shape[:2], bekor, kichraytirish)

    fazalar = []
    jami = teglar.size - (int(bekor_maska.sum()) if bekor_maska is not None else 0)
    jami = max(jami, 1)
    for i in range(k):
        maska = teglar == i
        if bekor_maska is not None:
            maska = maska & ~bekor_maska
        if not maska.any():
            continue
        ulush = float(maska.sum()) / jami * 100
        st = _faza_statistikasi(bgr, maska)
        tal = _talqin_ppl(st["V"], st["S"], st["H"])
        fazalar.append({
            "indeks": i,
            "ulush_foiz": round(ulush, 1),
            "rang_hex": st["rang_hex"],
            "belgi_rang": "#%02x%02x%02x" % PALITRA[i % len(PALITRA)],
            "talqin": tal["nom"],
            "ishonch": tal["ishonch"],
            "optik_belgi": tal["optik_belgi"],
            "izoh": tal["izoh"],
            **_don_olchami(maska, mkm_px),
        })
    fazalar.sort(key=lambda f: -f["ulush_foiz"])

    return {
        "fazalar": fazalar,
        "qoplama": _qoplama(bgr, teglar, k, bekor_maska=bekor_maska),
        "olcham": bgr.shape[:2],
        "mkm_px": round(mkm_px, 4) if mkm_px else None,
        "rejim": "bitta tasvir",
    }


def segment_juft(ppl_yol: str | Path, xpl_yol: str | Path, k: int = 7,
                 mkm_px: float | None = None, max_tomon: int = 900, bekor=None) -> dict:
    """Segment a PPL/XPL pair of the same field jointly.

    The XPL image is resized to the PPL size if they differ. Features are
    the six CIELAB channels of both images (Eq. 1 in the paper).
    """
    ppl, kichraytirish = _oqish(ppl_yol, max_tomon)
    xpl, _ = _oqish(xpl_yol, max_tomon)
    if xpl.shape[:2] != ppl.shape[:2]:
        xpl = cv2.resize(xpl, (ppl.shape[1], ppl.shape[0]), interpolation=cv2.INTER_AREA)
    if mkm_px:
        mkm_px = mkm_px / kichraytirish

    X = np.hstack([_belgilar(ppl), _belgilar(xpl)])
    teglar = _silliqlash(_kmeans(X, k).reshape(ppl.shape[:2]), k)
    bekor_maska = _bekor_maska(ppl.shape[:2], bekor, kichraytirish)

    fazalar = []
    jami = teglar.size - (int(bekor_maska.sum()) if bekor_maska is not None else 0)
    jami = max(jami, 1)
    for i in range(k):
        maska = teglar == i
        if bekor_maska is not None:
            maska = maska & ~bekor_maska
        if not maska.any():
            continue
        ulush = float(maska.sum()) / jami * 100
        sp = _faza_statistikasi(ppl, maska)
        sx = _faza_statistikasi(xpl, maska)
        tal = _talqin_juft(sp["V"], sp["S"], sp["H"], sx["V"], sx["S"])
        fazalar.append({
            "indeks": i,
            "ulush_foiz": round(ulush, 1),
            "rang_hex": sp["rang_hex"],
            "xpl_rang_hex": sx["rang_hex"],
            "belgi_rang": "#%02x%02x%02x" % PALITRA[i % len(PALITRA)],
            "talqin": tal["nom"],
            "ishonch": tal["ishonch"],
            "optik_belgi": tal["optik_belgi"],
            "izoh": tal["izoh"],
            **_don_olchami(maska, mkm_px),
        })
    fazalar.sort(key=lambda f: -f["ulush_foiz"])

    return {
        "fazalar": fazalar,
        "qoplama": _qoplama(ppl, teglar, k, bekor_maska=bekor_maska),
        "qoplama_xpl": _qoplama(xpl, teglar, k, bekor_maska=bekor_maska),
        "olcham": ppl.shape[:2],
        "mkm_px": round(mkm_px, 4) if mkm_px else None,
        "rejim": "PPL + XPL juftligi",
    }


def modal_tarkib(natija: dict, eng_kam_ulush: float = 1.0) -> list[dict]:
    """Modal table for reporting.

    Phases with the same group label are merged (k-means often splits one
    mineral by brightness or extinction position); phases below
    eng_kam_ulush per cent are pooled as "minor phases".
    """
    guruh: dict[str, dict] = {}
    for f in natija["fazalar"]:
        kalit = f["talqin"]
        if kalit in guruh:
            g = guruh[kalit]
            g["ulush_foiz"] = round(g["ulush_foiz"] + f["ulush_foiz"], 1)
            g["don_soni"] += f.get("don_soni", 0)
            g["fazalar_soni"] += 1
            # confidence of the merged entry: maximum of its phases
            g["ishonch"] = round(max(g.get("ishonch", 0), f.get("ishonch", 0)), 2)
            if f.get("median_diametr") and g.get("median_diametr"):
                g["median_diametr"] = round(
                    (g["median_diametr"] + f["median_diametr"]) / 2, 1)
        else:
            g = dict(f)
            g["fazalar_soni"] = 1
            guruh[kalit] = g

    natijalar = sorted(guruh.values(), key=lambda f: -f["ulush_foiz"])
    katta = [f for f in natijalar if f["ulush_foiz"] >= eng_kam_ulush]
    qolgan = sum(f["ulush_foiz"] for f in natijalar if f["ulush_foiz"] < eng_kam_ulush)
    if qolgan > 0:
        katta.append({"talqin": "boshqa (mayda fazalar)", "ulush_foiz": round(qolgan, 1),
                      "belgi_rang": "#999999", "izoh": "", "don_soni": 0,
                      "median_diametr": None, "birlik": "-", "fazalar_soni": 0,
                      "donlar_ajraldi": False, "ishonch": 0.0,
                      "optik_belgi": ""})
    for f in katta:
        if not f.get("donlar_ajraldi") and f.get("olcham_sababi"):
            f["izoh"] = (f["izoh"] + " " + f["olcham_sababi"]).strip()
    return katta


# English equivalents of the group labels produced by the rules
GROUP_EN = {
    "ma'danli mineral": "opaque (ore) mineral",
    "ma'danli mineral yoki bo'shliq": "opaque mineral or void",
    "izotrop yoki so'nish holatidagi mineral": "isotropic or at extinction",
    "yuqori interferentsiyali: karbonat, muskovit, epidot": "high birefringence: carbonate, muscovite, epidote",
    "rangli mineral: biotit, xlorit, amfibol": "coloured mafic: biotite, chlorite, amphibole",
    "rangsiz: kvars, dala shpati yoki yuqori tartib oq": "colourless, bright in XPL: quartz, feldspar",
    "kvars, dala shpati (past interferentsiya)": "quartz, feldspar (first-order grey)",
    "rangsiz mineral: kvars, dala shpati, karbonat": "colourless: quartz, feldspar, carbonate",
    "jigarrang-sariq: biotit, rutil, gidroksidlar": "brown-yellow: biotite, rutile, hydroxides",
    "yashil: xlorit, amfibol, epidot": "green: chlorite, amphibole, epidote",
    "ko'k-binafsha ton": "blue-violet tone",
    "qizil-pushti ton": "red-pink tone",
}
