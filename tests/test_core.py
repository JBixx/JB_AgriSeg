"""Tests unitaires des briques critiques (sans modèle ni données réelles).

Lancement :  python -m pytest -q tests/
"""
import sys
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.conversion import (mask_to_polygons, normalize_class_name, polygons_to_yolo_lines,  # noqa: E402
                            read_binary_mask, safe_dirname, yolo_lines_to_polygons)
from src.postprocess import Instance, merge_instances, otsu_threshold, summarize  # noqa: E402
from src.tiling import make_tiles  # noqa: E402


def test_read_mask_palette_and_rgb(tmp_path):
    # masque palette : indice 1 = lésion
    a = np.zeros((20, 20), np.uint8)
    a[5:10, 5:10] = 1
    im = Image.fromarray(a, mode="P")
    im.putpalette([0, 0, 0, 128, 0, 0] + [0] * 762)
    im.save(tmp_path / "p.png")
    assert read_binary_mask(tmp_path / "p.png").sum() == 25
    # masque RGB anti-aliasé : 128 = lésion, 30 = bruit de contour
    b = np.zeros((20, 20, 3), np.uint8)
    b[5:10, 5:10, 0] = 128
    b[10, 5:10, 0] = 30
    Image.fromarray(b).save(tmp_path / "r.png")
    assert read_binary_mask(tmp_path / "r.png", threshold=64).sum() == 25


def test_mask_polygon_roundtrip():
    m = np.zeros((100, 200), bool)
    m[10:40, 20:60] = True      # instance 1
    m[60:90, 120:180] = True    # instance 2
    m[0:2, 0:2] = True          # bruit (< aire minimale)
    polys = mask_to_polygons(m, min_area=30)
    assert len(polys) == 2
    lines = polygons_to_yolo_lines(polys, 200, 100)
    assert all(ln.startswith("0 ") for ln in lines)
    vals = np.array([float(v) for ln in lines for v in ln.split()[1:]])
    assert vals.min() >= 0 and vals.max() <= 1
    back = yolo_lines_to_polygons(lines, 200, 100)
    assert np.allclose(back[0][1], polys[0], atol=0.01)


def test_tiles_cover_image_with_overlap():
    tiles = make_tiles(1500, 1000, 640, 0.25)
    cov = np.zeros((1000, 1500), bool)
    for t in tiles:
        assert t.w <= 640 and t.h <= 640
        cov[t.y0:t.y1, t.x0:t.x1] = True
    assert cov.all()
    assert make_tiles(300, 200, 640, 0.25)[0].w == 300  # petite image = 1 tuile


def test_merge_duplicates_across_tiles():
    from shapely.geometry import box

    a = Instance(box(0, 0, 100, 100), 0, "lesion", 0.8)
    b = Instance(box(50, 0, 150, 100), 0, "lesion", 0.9)   # même foyer vu par la tuile voisine
    c = Instance(box(300, 300, 320, 320), 0, "lesion", 0.5)
    out = merge_instances([a, b, c])
    assert len(out) == 2
    assert out[0].id == 1 and abs(out[0].polygon.area - 15000) < 1e-6 and out[0].conf == 0.9
    s = summarize(out, 1000, 1000, 0.01)
    assert s["nb_foyers"] == 2 and abs(s["pct_image_infectee"] - 1.54) < 1e-6


def test_class_name_normalization():
    ca, cl = {"Paddy": "Rice"}, {"Radish - downey mildew": "Radish - downy mildew"}
    assert normalize_class_name("Apple - apple black rot ") == "Apple - black rot"
    assert normalize_class_name("Coffee - coffee black rot Google") == "Coffee - black rot"
    assert normalize_class_name("Cabbage - cabbage black rot Bing  - Copy") == "Cabbage - black rot"
    assert normalize_class_name("Paddy - blast", ca) == "Rice - blast"
    assert normalize_class_name("Radish - downey mildew", ca, cl) == "Radish - downy mildew"
    assert safe_dirname("Wheat - bacterial leaf streak (black chaff)") == "Wheat__bacterial_leaf_streak_black_chaff"


def test_otsu_bimodal():
    x = np.concatenate([np.random.RandomState(0).normal(0, 0.05, 1000),
                        np.random.RandomState(1).normal(1, 0.05, 1000)])
    assert 0.2 < otsu_threshold(x) < 0.8
