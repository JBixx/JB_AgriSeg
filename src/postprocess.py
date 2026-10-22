"""Post-traitement des prédictions : instances -> polygones fusionnés, surfaces,
indice de végétation, exports GeoJSON / CSV / résumé."""
from __future__ import annotations

import csv
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
from shapely.geometry import MultiPolygon, Polygon, mapping
from shapely.ops import unary_union
from shapely.validation import make_valid


@dataclass
class Instance:
    """Un foyer de maladie détecté (coordonnées en pixels de l'image complète)."""
    polygon: Polygon | MultiPolygon
    cls_id: int
    cls_name: str
    conf: float
    id: int = 0
    extra: dict[str, Any] = field(default_factory=dict)


def _clean(geom) -> Polygon | MultiPolygon | None:
    """Rend une géométrie valide et ne garde que la partie surfacique."""
    if geom.is_empty:
        return None
    if not geom.is_valid:
        geom = make_valid(geom)
    if geom.geom_type == "GeometryCollection":
        parts = [g for g in geom.geoms if g.geom_type in ("Polygon", "MultiPolygon")]
        geom = unary_union(parts) if parts else None
    if geom is None or geom.is_empty or geom.area <= 0:
        return None
    return geom


def result_to_instances(result, offset: tuple[int, int] = (0, 0), min_area: float = 0) -> list[Instance]:
    """Convertit un `ultralytics.engine.results.Results` en instances.
    `offset` = coin haut-gauche de la tuile dans l'image complète."""
    out: list[Instance] = []
    if result.masks is None or len(result.masks) == 0:
        return out
    names = result.names
    confs = result.boxes.conf.cpu().numpy()
    clss = result.boxes.cls.cpu().numpy().astype(int)
    for pts, c, k in zip(result.masks.xy, confs, clss):
        if len(pts) < 3:
            continue
        poly = _clean(Polygon(pts + np.array(offset, dtype=np.float32)))
        if poly is None or poly.area < min_area:
            continue
        out.append(Instance(poly, int(k), names[int(k)], float(c)))
    return out


def merge_instances(instances: list[Instance], min_overlap: float = 0.05) -> list[Instance]:
    """Fusionne les instances de même classe qui se recouvrent (doublons créés par
    le recouvrement des tuiles). Deux polygones sont fusionnés si leur
    intersection dépasse `min_overlap` × l'aire du plus petit. Union-find."""
    n = len(instances)
    if n < 2:
        return _renumber(instances)
    parent = list(range(n))

    def find(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    from shapely.strtree import STRtree

    geoms = [i.polygon for i in instances]
    tree = STRtree(geoms)
    for i, g in enumerate(geoms):
        for j in tree.query(g):
            j = int(j)
            if j <= i or instances[i].cls_id != instances[j].cls_id:
                continue
            inter = g.intersection(geoms[j]).area
            if inter > min_overlap * min(g.area, geoms[j].area):
                parent[find(i)] = find(j)
    groups: dict[int, list[int]] = {}
    for i in range(n):
        groups.setdefault(find(i), []).append(i)
    merged = []
    for idx in groups.values():
        geom = _clean(unary_union([geoms[i] for i in idx]))
        if geom is None:
            continue
        best = max(idx, key=lambda i: instances[i].conf)
        merged.append(Instance(geom, instances[best].cls_id, instances[best].cls_name,
                               max(instances[i].conf for i in idx)))
    return _renumber(merged)


def _renumber(instances: list[Instance]) -> list[Instance]:
    """Numérote les foyers du plus grand au plus petit (id 1 = plus grand foyer)."""
    instances = sorted(instances, key=lambda i: -i.polygon.area)
    for k, inst in enumerate(instances, 1):
        inst.id = k
    return instances


# ----------------------------------------------------------------------------- végétation
def otsu_threshold(x: np.ndarray, bins: int = 256) -> float:
    """Seuil d'Otsu (numpy pur) sur un tableau 1D de valeurs."""
    hist, edges = np.histogram(x, bins=bins)
    hist = hist.astype(np.float64)
    mids = (edges[:-1] + edges[1:]) / 2
    w0 = np.cumsum(hist)
    w1 = w0[-1] - w0
    m0 = np.cumsum(hist * mids) / np.maximum(w0, 1)
    m1 = (np.sum(hist * mids) - np.cumsum(hist * mids)) / np.maximum(w1, 1)
    var = w0 * w1 * (m0 - m1) ** 2
    # Seuil = bord supérieur du bin k (classe 0 = bins 0..k). Si plusieurs k
    # donnent la variance maximale (vallée vide entre deux modes), on prend le
    # centre du plateau pour couper au milieu de la vallée.
    best = np.flatnonzero(var >= var.max() * (1 - 1e-9))
    k = best[len(best) // 2]
    return float(edges[k + 1])


def vegetation_mask(rgb: np.ndarray) -> np.ndarray:
    """Masque de végétation NON SUPERVISÉ : Excess Green (2G-R-B) + seuil d'Otsu.
    Sert de dénominateur pour exprimer la surface malade en % de la végétation."""
    a = rgb[..., :3].astype(np.float32)
    s = a.sum(axis=2) + 1e-6
    r, g, b = a[..., 0] / s, a[..., 1] / s, a[..., 2] / s
    exg = 2 * g - r - b
    sample = exg[:: max(1, exg.shape[0] // 512), :: max(1, exg.shape[1] // 512)].ravel()
    return exg > otsu_threshold(sample)


# ----------------------------------------------------------------------------- géoréférencement
class GeoRef:
    """Transformation pixel -> coordonnées terrain (GeoTIFF via rasterio)."""

    def __init__(self, transform=None, crs=None):
        self.transform, self.crs = transform, crs

    @property
    def ok(self) -> bool:
        return self.transform is not None and self.crs is not None

    @property
    def projected(self) -> bool:
        return self.ok and self.crs.is_projected

    def pixel_area_m2(self) -> float | None:
        """Surface d'un pixel en m² si le CRS est projeté en mètres."""
        if not self.projected:
            return None
        t = self.transform
        unit = (self.crs.linear_units or "").lower()
        f = 1.0 if unit in ("metre", "meter", "m", "") else 0.3048 if "foot" in unit else 1.0
        return abs(t.a * t.e - t.b * t.d) * f * f

    def to_wgs84(self, geom):
        """Géométrie pixels -> géométrie WGS84 (EPSG:4326, conforme RFC 7946)."""
        from rasterio.warp import transform_geom
        from shapely.affinity import affine_transform
        from shapely.geometry import shape

        t = self.transform
        g = affine_transform(geom, [t.a, t.b, t.d, t.e, t.c, t.f])
        return shape(transform_geom(self.crs, "EPSG:4326", mapping(g)))


# ----------------------------------------------------------------------------- statistiques & exports
def instance_table(instances: list[Instance], px_area_m2: float, geo: GeoRef | None = None) -> list[dict]:
    """Une ligne par foyer : id, classe, confiance, surfaces, centroïde."""
    rows = []
    for i in instances:
        c = i.polygon.centroid
        row = {
            "id": i.id,
            "classe": i.cls_name,
            "confiance": round(i.conf, 4),
            "surface_px": round(i.polygon.area, 1),
            "surface_m2": round(i.polygon.area * px_area_m2, 6),
            "centroide_x_px": round(c.x, 1),
            "centroide_y_px": round(c.y, 1),
        }
        if geo is not None and geo.ok:
            gc = geo.to_wgs84(c)
            row["centroide_lon"], row["centroide_lat"] = round(gc.x, 8), round(gc.y, 8)
        rows.append(row)
    return rows


def summarize(instances: list[Instance], width: int, height: int, px_area_m2: float,
              veg_fraction: float | None = None, diagnosis: list[dict] | None = None) -> dict:
    """Résumé global : nombre de foyers, surface infectée (en % image et % végétation)."""
    union = unary_union([i.polygon for i in instances]) if instances else None
    infected_px = float(union.area) if union is not None else 0.0
    total_px = float(width * height)
    s = {
        "nb_foyers": len(instances),
        "foyers_par_classe": {},
        "surface_infectee_px": round(infected_px, 1),
        "surface_infectee_m2": round(infected_px * px_area_m2, 4),
        "surface_image_m2": round(total_px * px_area_m2, 4),
        "pct_image_infectee": round(100 * infected_px / total_px, 3),
        "taille_image_px": [width, height],
        "surface_pixel_m2": px_area_m2,
    }
    for i in instances:
        s["foyers_par_classe"][i.cls_name] = s["foyers_par_classe"].get(i.cls_name, 0) + 1
    if veg_fraction is not None and veg_fraction > 0:
        s["pct_vegetation"] = round(100 * veg_fraction, 2)
        s["pct_vegetation_infectee"] = round(min(100.0, 100 * infected_px / (veg_fraction * total_px)), 3)
    if diagnosis:
        s["diagnostic"] = diagnosis
    return s


def write_csv(rows: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    cols = list(rows[0].keys()) if rows else ["id", "classe", "confiance", "surface_px", "surface_m2",
                                               "centroide_x_px", "centroide_y_px"]
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        w.writerows(rows)


def write_geojson(instances: list[Instance], rows: list[dict], path: Path, geo: GeoRef | None = None,
                  source: str = "") -> None:
    """GeoJSON des foyers. Si l'entrée est géoréférencée : WGS84 ; sinon
    coordonnées en pixels (origine en haut à gauche, y vers le bas)."""
    feats = []
    for inst, row in zip(instances, rows):
        g = geo.to_wgs84(inst.polygon) if (geo is not None and geo.ok) else inst.polygon
        feats.append({"type": "Feature", "geometry": mapping(g), "properties": row})
    fc = {
        "type": "FeatureCollection",
        "name": Path(source).stem if source else "foyers",
        "features": feats,
        "properties": {"coordonnees": "WGS84 (EPSG:4326)" if (geo is not None and geo.ok)
                       else "pixels image (non géoréférencé)", "source": source},
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(fc, ensure_ascii=False), encoding="utf-8")
