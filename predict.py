"""
predict.py — Inférence prête à l'emploi : image, dossier ou grande orthomosaïque.

Pour chaque image, produit dans outputs/predictions/<nom>/ :
  1. <nom>_carte.png      carte des zones infectées colorées, numérotées, avec légende
  2. <nom>_foyers.geojson polygones des foyers (WGS84 si GeoTIFF géoréférencé, sinon pixels)
  3. <nom>_foyers.csv     un foyer par ligne : id, classe, confiance, surface px et m², centroïde
  4. <nom>_resume.json    nombre de foyers, % de surface infectée, diagnostic (si modèle cls)

Les grandes images (côté > predict.tile_trigger) sont découpées en tuiles avec
recouvrement, prédites tuile par tuile puis recollées (fusion des doublons).

Usage :
    python predict.py --source image.jpg
    python predict.py --source dossier/ --gsd 1.5
    python predict.py --source ortho.tif --tile-size 640 --overlap 0.25
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps

from src.postprocess import (GeoRef, instance_table, merge_instances, result_to_instances, summarize,
                             vegetation_mask, write_csv, write_geojson)
from src.tiling import make_tiles
from src.utils import get_device, get_logger, load_config, resolve
from src.visualize import render_map

log = get_logger()
IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
Image.MAX_IMAGE_PIXELS = None  # orthomosaïques : pas de limite de taille PIL


# ----------------------------------------------------------------------------- chargement
def load_image(path: Path) -> tuple[np.ndarray, GeoRef]:
    """Charge une image en RGB uint8. Les GeoTIFF sont lus via rasterio pour
    récupérer la transformation affine et le CRS (géoréférencement)."""
    if path.suffix.lower() in (".tif", ".tiff"):
        try:
            import rasterio

            with rasterio.open(path) as src:
                bands = [1, 2, 3] if src.count >= 3 else [1, 1, 1]
                a = np.transpose(src.read(bands), (1, 2, 0))
                geo = GeoRef(src.transform, src.crs) if src.crs is not None else GeoRef()
            if a.dtype != np.uint8:  # étirement 2-98 % pour les images 16 bits / float
                lo, hi = np.nanpercentile(a, [2, 98])
                a = np.clip((a - lo) / max(hi - lo, 1e-6) * 255, 0, 255).astype(np.uint8)
            return np.ascontiguousarray(a), geo
        except ImportError:
            log.warning("rasterio absent : GeoTIFF lu sans géoréférencement")
    with Image.open(path) as im:
        return np.asarray(ImageOps.exif_transpose(im).convert("RGB")), GeoRef()


def load_class_names(weights: Path) -> dict[str, str]:
    """Noms lisibles des classes de diagnostic (models/cls_classes.json)."""
    p = weights.with_name("cls_classes.json")
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else {}


# ----------------------------------------------------------------------------- moteur
class Analyzer:
    """Charge les modèles une fois et analyse autant d'images que nécessaire
    (réutilisé par app.py)."""

    def __init__(self, cfg: dict, seg_weights: str | Path | None = None, cls_weights: str | Path | None = None,
                 device: str | None = None):
        from ultralytics import YOLO

        self.cfg, self.p = cfg, cfg["predict"]
        self.device = device or get_device(cfg["project"]["device"])
        sw = resolve(seg_weights or self.p["seg_weights"])
        if not sw.exists():
            raise FileNotFoundError(f"Modèle de segmentation introuvable : {sw} (lancez train.py --task seg)")
        self.seg = YOLO(str(sw), task="segment")
        cw = resolve(cls_weights or self.p["cls_weights"])
        self.cls = YOLO(str(cw), task="classify") if cw.exists() else None
        self.cls_names = load_class_names(cw) if self.cls else {}
        log.info(f"Modèles chargés (device={self.device}) : seg={sw.name}"
                 f"{', cls=' + cw.name if self.cls else ' (pas de modèle de diagnostic)'}")

    def _seg_predict(self, rgb: np.ndarray, imgsz: int, offset=(0, 0)):
        bgr = np.ascontiguousarray(rgb[..., ::-1])  # Ultralytics attend du BGR pour les tableaux numpy
        res = self.seg.predict(bgr, conf=self.p["conf"], iou=self.p["iou"], imgsz=imgsz, device=self.device,
                               retina_masks=True, verbose=False)[0]
        return result_to_instances(res, offset, self.p["min_area_px"])

    def diagnose(self, rgb: np.ndarray, k: int | None = None) -> list[dict]:
        """Top-k des maladies probables pour l'image entière (modèle de classification)."""
        if self.cls is None:
            return []
        k = k or self.cfg["cls"]["top_k"]
        res = self.cls.predict(np.ascontiguousarray(rgb[..., ::-1]), imgsz=self.cfg["cls"]["imgsz"],
                               device=self.device, verbose=False)[0]
        probs = res.probs.data.cpu().numpy()
        out = []
        for i in np.argsort(-probs)[:k]:
            d = self.cls.names[int(i)]
            out.append({"classe": self.cls_names.get(d, d.replace("__", " - ").replace("_", " ")),
                        "probabilite": round(float(probs[i]), 4)})
        return out

    def analyze(self, rgb: np.ndarray, geo: GeoRef | None = None, gsd_cm: float | None = None,
                tile_size: int | None = None, overlap: float | None = None, diagnose: bool | None = None) -> dict:
        """Analyse complète d'une image RGB -> instances, tableau, résumé."""
        geo = geo or GeoRef()
        h, w = rgb.shape[:2]
        ts = tile_size or self.p["tile_size"]
        ov = self.p["tile_overlap"] if overlap is None else overlap
        tiled = max(h, w) > self.p["tile_trigger"]
        if tiled:
            tiles = make_tiles(w, h, ts, ov)
            log.info(f"Image {w}x{h} -> {len(tiles)} tuiles de {ts}px (recouvrement {ov:.0%})")
            inst = []
            for k, t in enumerate(tiles, 1):
                inst += self._seg_predict(rgb[t.y0:t.y1, t.x0:t.x1], ts, (t.x0, t.y0))
                if k % 50 == 0:
                    log.info(f"  {k}/{len(tiles)} tuiles")
        else:
            inst = self._seg_predict(rgb, self.cfg["seg"]["imgsz"])
        inst = merge_instances(inst)

        px_m2 = geo.pixel_area_m2()
        if px_m2 is None:
            gsd = self.p["gsd_cm_per_px"] if gsd_cm is None else gsd_cm
            px_m2 = (gsd / 100.0) ** 2
        for i in inst:
            i.extra["surface_m2"] = i.polygon.area * px_m2
        veg = float(vegetation_mask(rgb).mean()) if self.p.get("vegetation_index") else None
        do_diag = (not tiled) if diagnose is None else diagnose
        diag = self.diagnose(rgb) if do_diag else []
        rows = instance_table(inst, px_m2, geo)
        summary = summarize(inst, w, h, px_m2, veg, diag)
        summary["georeference"] = bool(geo.ok)
        summary["tuilage"] = tiled
        return {"instances": inst, "rows": rows, "summary": summary, "geo": geo}

    def export(self, rgb: np.ndarray, result: dict, out_dir: Path, stem: str, source: str = "") -> dict[str, Path]:
        """Écrit carte PNG, GeoJSON, CSV et résumé JSON ; renvoie les chemins."""
        out_dir.mkdir(parents=True, exist_ok=True)
        paths = {"carte": out_dir / f"{stem}_carte.png", "geojson": out_dir / f"{stem}_foyers.geojson",
                 "csv": out_dir / f"{stem}_foyers.csv", "resume": out_dir / f"{stem}_resume.json"}
        render_map(rgb, result["instances"], result["summary"], paths["carte"], self.p["max_map_side"],
                   title=f"Carte des foyers — {stem}")
        write_geojson(result["instances"], result["rows"], paths["geojson"], result["geo"], source)
        write_csv(result["rows"], paths["csv"])
        paths["resume"].write_text(json.dumps(result["summary"], indent=2, ensure_ascii=False), encoding="utf-8")
        return paths


def main() -> None:
    ap = argparse.ArgumentParser(description="Détection et détourage des foyers de maladie")
    ap.add_argument("--source", required=True, help="image, dossier d'images ou GeoTIFF")
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--weights", help="poids de segmentation (défaut : models/best.pt)")
    ap.add_argument("--cls-weights", help="poids de diagnostic (défaut : models/cls_best.pt, optionnel)")
    ap.add_argument("--gsd", type=float, help="résolution au sol en cm/pixel (défaut : config)")
    ap.add_argument("--conf", type=float, help="seuil de confiance")
    ap.add_argument("--tile-size", type=int)
    ap.add_argument("--overlap", type=float)
    ap.add_argument("--force-tiles", action="store_true", help="tuile même les petites images")
    ap.add_argument("--out", help="dossier de sortie (défaut : outputs/predictions)")
    ap.add_argument("--device")
    a = ap.parse_args()

    cfg = load_config(a.config)
    if a.conf is not None:
        cfg["predict"]["conf"] = a.conf
    if a.force_tiles:
        cfg["predict"]["tile_trigger"] = 0
    src = Path(a.source)
    if not src.exists():
        sys.exit(f"[ERREUR] Source introuvable : {src}")
    files = [src] if src.is_file() else sorted(p for p in src.rglob("*") if p.suffix.lower() in IMG_EXT)
    if not files:
        sys.exit(f"[ERREUR] Aucune image dans {src}")
    out_root = resolve(a.out or cfg["predict"]["out_dir"])
    try:
        an = Analyzer(cfg, a.weights, a.cls_weights, a.device)
    except FileNotFoundError as e:
        sys.exit(f"[ERREUR] {e}")

    recap = []
    for f in files:
        try:
            rgb, geo = load_image(f)
            res = an.analyze(rgb, geo, a.gsd, a.tile_size, a.overlap)
            paths = an.export(rgb, res, out_root / f.stem, f.stem, str(f))
        except Exception as e:  # noqa: BLE001 — on continue sur les autres images
            log.error(f"{f.name} : échec ({e})")
            continue
        s = res["summary"]
        diag = f" | diagnostic : {s['diagnostic'][0]['classe']} ({s['diagnostic'][0]['probabilite']:.0%})" \
            if s.get("diagnostic") else ""
        log.info(f"{f.name} : {s['nb_foyers']} foyers, {s['pct_image_infectee']:.2f} % de l'image infectée, "
                 f"{s['surface_infectee_m2']:.4g} m²{diag} -> {paths['carte'].parent}")
        recap.append({"image": f.name, "nb_foyers": s["nb_foyers"], "pct_image_infectee": s["pct_image_infectee"],
                      "surface_infectee_m2": s["surface_infectee_m2"],
                      "pct_vegetation_infectee": s.get("pct_vegetation_infectee"),
                      "diagnostic": s["diagnostic"][0]["classe"] if s.get("diagnostic") else ""})
    if len(files) > 1 and recap:
        write_csv(recap, out_root / "recapitulatif.csv")
        log.info(f"Récapitulatif : {out_root / 'recapitulatif.csv'}")


if __name__ == "__main__":
    main()
