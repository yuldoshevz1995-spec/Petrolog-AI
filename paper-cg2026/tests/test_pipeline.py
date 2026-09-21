"""Smoke tests on synthetic images (no thin-section data required)."""
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from petrolog.imaging import segment_juft, modal_tarkib, kadrlarga_bolish, parse_scale_text  # noqa: E402
from petrolog.fabric import olchovlar, _opak_maska  # noqa: E402
from petrolog.labels import mineral_groups, program_group  # noqa: E402


def _synthetic(tmp_path):
    rng = np.random.default_rng(0)
    ppl = np.full((300, 400, 3), 225, np.uint8)
    xpl = np.full((300, 400, 3), 60, np.uint8)
    for _ in range(40):                                     # bright "quartz" grains
        c = tuple(int(v) for v in rng.integers(20, 380, 2)); r = int(rng.integers(8, 20))
        cv2.circle(ppl, c, r, (235, 235, 235), -1); cv2.circle(xpl, c, r, (200, 200, 200), -1)
    cv2.rectangle(ppl, (50, 50), (90, 80), (10, 10, 10), -1)  # opaque grain
    cv2.rectangle(xpl, (50, 50), (90, 80), (10, 10, 10), -1)
    p, x = tmp_path / "p.png", tmp_path / "x.png"
    cv2.imwrite(str(p), ppl); cv2.imwrite(str(x), xpl)
    return p, x


def test_segmentation_and_fabric(tmp_path):
    p, x = _synthetic(tmp_path)
    seg = segment_juft(p, x, k=4)
    assert abs(sum(f["ulush_foiz"] for f in seg["fazalar"]) - 100) < 1
    assert any(program_group(f["talqin"]) == "OPQ" for f in seg["fazalar"])
    seg["modal"] = modal_tarkib(seg)
    o = olchovlar(cv2.imread(str(p)), cv2.imread(str(x)), seg)
    assert 0 <= o["yonalganlik"]["koeffitsient"] <= 1


def test_opaque_needs_both_nicols():
    ppl = np.zeros((10, 10, 3), np.uint8); xpl = np.full((10, 10, 3), 200, np.uint8)
    assert _opak_maska(ppl, xpl).sum() == 0          # dark only in PPL -> not opaque
    assert _opak_maska(ppl, np.zeros_like(xpl)).sum() == 100


def test_split_touching_panels():
    left = np.full((200, 200, 3), 230, np.uint8); right = np.zeros((200, 200, 3), np.uint8)
    right[:, :, 2] = 180                              # darker, saturated half
    r = kadrlarga_bolish(np.hstack([left, right]))
    assert r["ajratildi"] and len(r["kadrlar"]) == 2


def test_scale_text_and_labels():
    assert parse_scale_text("0,5 mm")[0] == 500
    assert parse_scale_text("T mm")[0] == 1000
    assert mineral_groups("kvars va seritsit")["HIB"] == 1
