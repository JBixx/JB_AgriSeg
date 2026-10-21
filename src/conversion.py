"""Conversions de formats : masques PNG -> polygones YOLO-seg, COCO -> YOLO-seg,
normalisation des noms de classes."""
from __future__ import annotations

import re
from pathlib import Path

import cv2
import numpy as np
from PIL import Image


# ----------------------------------------------------------------------------- masques
def read_binary_mask(path: str | Path, threshold: int = 64) -> np.ndarray:
    """Lit un masque PNG et renvoie un masque booléen (True = lésion).

    Gère les trois encodages rencontrés :
      * mode 'P' (palette) ou valeurs {0,1} : tout indice > 0 est une lésion ;
      * mode 'L' / 'RGB' 0-255 : pixel lésion si le canal max >= `threshold`
        (le seuil absorbe l'anti-aliasing des contours : valeurs 30, 53, 63...).
    """
    m = Image.open(path)
    a = np.asarray(m)
    if m.mode in ("P", "1") or a.max() <= 1:
        return a > 0 if a.ndim == 2 else a.max(axis=2) > 0
    if a.ndim == 3:
        a = a[..., :3].max(axis=2)
    return a >= threshold


def mask_to_polygons(mask: np.ndarray, min_area: float = 30, epsilon: float = 1.0) -> list[np.ndarray]:
    """Découpe un masque binaire en instances (composantes connexes) et renvoie
    un polygone (N,2) en pixels par instance. Les trous internes sont ignorés
    (le format YOLO-seg ne gère qu'un contour externe par instance)."""
    m = mask.astype(np.uint8)
    contours, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    polys = []
    for c in contours:
        if cv2.contourArea(c) < min_area:
            continue
        if epsilon > 0:
            c = cv2.approxPolyDP(c, epsilon, True)
        if len(c) >= 3:
            polys.append(c.reshape(-1, 2).astype(np.float32))
    return polys


def polygons_to_yolo_lines(polys: list[np.ndarray], w: int, h: int, cls_id: int = 0) -> list[str]:
    """Polygones pixels -> lignes YOLO-seg normalisées : `cls x1 y1 x2 y2 ...`."""
    lines = []
    for p in polys:
        q = p.copy()
        q[:, 0] = np.clip(q[:, 0] / w, 0, 1)
        q[:, 1] = np.clip(q[:, 1] / h, 0, 1)
        lines.append(f"{cls_id} " + " ".join(f"{v:.6f}" for v in q.reshape(-1)))
    return lines


def yolo_lines_to_polygons(lines: list[str], w: int, h: int) -> list[tuple[int, np.ndarray]]:
    """Inverse de polygons_to_yolo_lines (pour la vérification visuelle)."""
    out = []
    for ln in lines:
        v = ln.split()
        if len(v) < 7:
            continue
        pts = np.array(v[1:], dtype=np.float32).reshape(-1, 2) * [w, h]
        out.append((int(v[0]), pts))
    return out


def coco_to_yolo(coco: dict, cls_map: dict[int, int]) -> dict[str, list[str]]:
    """Convertit un dict COCO (segmentation polygonale) en {nom_image: lignes YOLO}.
    Les annotations RLE sont décodées via pycocotools si disponible."""
    imgs = {i["id"]: i for i in coco["images"]}
    out: dict[str, list[str]] = {i["file_name"]: [] for i in coco["images"]}
    for a in coco["annotations"]:
        im = imgs[a["image_id"]]
        w, h = im["width"], im["height"]
        cid = cls_map.get(a["category_id"])
        seg = a.get("segmentation")
        if cid is None or not seg:
            continue
        if isinstance(seg, dict):  # RLE
            try:
                from pycocotools import mask as mu
            except ImportError as e:
                raise ImportError("Annotations RLE : installez pycocotools") from e
            polys = mask_to_polygons(mu.decode(seg) > 0)
        else:
            polys = [np.array(s, dtype=np.float32).reshape(-1, 2) for s in seg if len(s) >= 6]
        out[im["file_name"]] += polygons_to_yolo_lines(polys, w, h, cid)
    return out


# ----------------------------------------------------------------------------- classes
_SRC_SUFFIX = re.compile(r"\s+(google|bing|baidu)$", re.I)


def normalize_class_name(raw: str, crop_aliases: dict | None = None, class_aliases: dict | None = None) -> str:
    """Harmonise un nom de classe « Culture - maladie ».

    Exemples :
      'Apple - apple black rot '            -> 'Apple - black rot'
      'Coffee - coffee black rot Google'    -> 'Coffee - black rot'
      'Cabbage - cabbage black rot Bing  - Copy' -> 'Cabbage - black rot'
      'Paddy - blast' (alias)               -> 'Rice - blast'
    """
    c = re.sub(r"\s+", " ", str(raw)).strip()
    c = re.sub(r"\s*-\s*copy$", "", c, flags=re.I)
    c = _SRC_SUFFIX.sub("", c)
    crop, _, dis = c.partition(" - ")
    crop, dis = crop.strip().title(), dis.strip().lower()
    dis = _SRC_SUFFIX.sub("", dis)
    if dis.startswith(crop.lower() + " "):
        dis = dis[len(crop) + 1:]
    dis = re.sub(r"\s*\(\s*\)$", "", dis).strip()  # « () » vides en fin de nom
    crop = (crop_aliases or {}).get(crop, crop)
    name = f"{crop} - {dis}" if dis else crop
    return (class_aliases or {}).get(name, name)


def safe_dirname(name: str) -> str:
    """Nom de classe -> nom de dossier sûr (« Tomato - late blight » -> « Tomato__late_blight »)."""
    s = name.replace(" - ", "__")
    s = re.sub(r"[^\w]+", "_", s).strip("_")
    return s
