"""Reference labels from petrographic descriptions (Section 4.1 of the paper).

The descriptions are written in Uzbek. Mineral names are mapped to the five
groups used by the rules, and fabric features are detected by keyword.
English glosses are given in comments. Matching is case-insensitive
substring matching on the normalised text.
"""

GURUHLAR = ["QF", "MAF", "HIB", "OPQ", "ISO"]
# QF quartz-feldspar, MAF coloured mafic, HIB high birefringence,
# OPQ opaque, ISO isotropic

LUGAT = {
    # quartz, plagioclase, albite, feldspar, sanidine, oligoclase, orthoclase,
    # microcline, andesine
    "QF": ["kvars", "plagioklaz", "albit", "dala shpat", "dalashpat", "sanidin",
           "oligoklaz", "ortoklaz", "mikroklin", "andezin"],
    # chlorite, biotite, amphibole, hornblende, actinolite, pyroxene, augite
    "MAF": ["xlorit", "biotit", "amfibol", "gornblend", "aktinolit", "piroksen", "avgit"],
    # carbonate, calcite, dolomite, sericite, muscovite, epidote, titanite (sphene)
    "HIB": ["karbonat", "kalsit", "dolomit", "seritsit", "muskovit", "epidot",
            "titanit", "sfen"],
    # sulphide, ore, ore mineral, pyrite, magnetite, hematite, galena,
    # chalcopyrite, sphalerite, opaque
    "OPQ": ["sulfid", "rudali", "ma'danli mineral", "pirit", "magnetit", "gematit",
            "galenit", "xalkopirit", "sfalerit", "opak"],
    # glass, garnet, fluorite
    "ISO": ["shisha", "granat", "flyuorit"],
}

BELGILAR = {
    "porfir": ["porfir"],                                   # porphyritic
    "mayda_asos": ["mikrolit", "afanit", "kriptofelsit",    # fine-grained groundmass:
                   "kriptokristall", "mayda donador",       # microlitic, aphanitic,
                   "mikrofelsit", "felzit", "shisha"],      # cryptocrystalline, felsitic, glassy
    "yonalgan": ["flyuid", "flyud", "slanes", "oqim"],      # flow / schistose / oriented
    "tomir": ["tomir"],                                     # vein
}


def _norm(t: str) -> str:
    return t.lower().replace("‘", "'").replace("’", "'").replace("`", "'")


def mineral_groups(text: str) -> dict:
    """Presence (0/1) of each mineral group in a description."""
    t = _norm(text)
    return {g: int(any(k in t for k in kal)) for g, kal in LUGAT.items()}


def fabric_features(text: str) -> dict:
    """Presence (0/1) of each fabric feature in a description."""
    t = _norm(text)
    return {b: int(any(k in t for k in kal)) for b, kal in BELGILAR.items()}


def program_group(label: str) -> str:
    """Map a rule output label (Uzbek) to one of the five groups."""
    t = _norm(label)
    if t.startswith("ma'danli"):
        return "OPQ"
    if t.startswith("izotrop"):
        return "ISO"
    if t.startswith("yuqori interf"):
        return "HIB"
    if t.startswith(("rangli", "jigarrang", "yashil")):
        return "MAF"
    if t.startswith(("rangsiz", "kvars")):
        return "QF"
    return "OTHER"
