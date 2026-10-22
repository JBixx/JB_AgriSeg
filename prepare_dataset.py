"""
prepare_dataset.py — Convertit les données brutes en jeux prêts pour Ultralytics.

  * --task seg : masques PNG (ou YOLO / COCO) -> YOLO segmentation unifié
                 data/processed/seg/{images,labels}/{train,val,test} + data.yaml
                 + 10 images de vérification dans outputs/verification/
  * --task cls : manifest du zip multi-sources -> dossiers par classe
                 data/processed/cls/{train,val,test}/<classe>/
  * --task all : les deux

Les données brutes (Dataset/) ne sont JAMAIS modifiées.

Usage :
    python prepare_dataset.py --task seg
    python prepare_dataset.py --task cls
"""
from __future__ import annotations

import argparse
import io
import json
import random
import shutil
import sys
import zipfile
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from PIL import Image

from src.conversion import (coco_to_yolo, mask_to_polygons, normalize_class_name, polygons_to_yolo_lines,
                            read_binary_mask, safe_dirname, yolo_lines_to_polygons)
from src.utils import ROOT, get_logger, load_config, resolve, set_seed
from src.visualize import save_annotation_preview

log = get_logger()
IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
SPLITS = ("train", "val", "test")


# =============================================================================
#  SEGMENTATION
# =============================================================================
def _find_mask(mask_dir: Path, stem: str) -> Path | None:
    for ext in (".png", ".PNG", ".bmp", ".tif", ".jpg"):
        p = mask_dir / f"{stem}{ext}"
        if p.exists():
            return p
    return None


def collect_seg_items(cfg: dict) -> list[dict]:
    """Inventaire des paires (image, annotation) de toutes les sources de segmentation."""
    items = []
    for src in cfg["raw"]["seg_sources"]:
        kind = src.get("type", "masks")
        img_dir = resolve(src["images"])
        if not img_dir.is_dir():
            raise FileNotFoundError(f"Dossier d'images introuvable : {img_dir}")
        if kind == "masks":
            mdir = resolve(src["masks"])
            for p in sorted(img_dir.iterdir()):
                if p.suffix.lower() not in IMG_EXT:
                    continue
                m = _find_mask(mdir, p.stem)
                if m is None:
                    log.warning(f"Image sans masque ignorée : {p.name}")
                    continue
                items.append({"img": p, "mask": m, "kind": "masks", "group": f"{src['name']}/{p.stem}",
                              "source": src["name"], "split": src.get("split")})
        elif kind == "yolo":
            ldir = resolve(src["labels"])
            for p in sorted(img_dir.iterdir()):
                if p.suffix.lower() in IMG_EXT:
                    items.append({"img": p, "label": ldir / f"{p.stem}.txt", "kind": "yolo",
                                  "group": f"{src['name']}/{p.stem}", "source": src["name"], "split": src.get("split")})
        elif kind == "coco":
            coco = json.loads(resolve(src["annotations"]).read_text(encoding="utf-8"))
            names = cfg["prepare"]["seg"]["class_names"]
            cmap = {c["id"]: names.index(src.get("class_map", {}).get(c["name"], c["name"]))
                    for c in coco["categories"]
                    if src.get("class_map", {}).get(c["name"], c["name"]) in names}
            lines = coco_to_yolo(coco, cmap)
            for fn, ls in lines.items():
                items.append({"img": img_dir / fn, "lines": ls, "kind": "coco",
                              "group": f"{src['name']}/{Path(fn).stem}", "source": src["name"], "split": src.get("split")})
        else:
            raise ValueError(f"Type de source inconnu : {kind}")
    return items


def _load_pair(it: dict, pcfg: dict) -> tuple[np.ndarray, np.ndarray | None, list[str] | None]:
    """Charge l'image (RGB, pixels bruts sans rotation EXIF) et son masque ou ses lignes YOLO."""
    with Image.open(it["img"]) as im:
        rgb = np.asarray(im.convert("RGB"))
    if it["kind"] == "masks":
        mask = read_binary_mask(it["mask"], pcfg["mask_threshold"])
        if mask.shape != rgb.shape[:2]:  # masque à une autre résolution -> recalage
            mask = np.asarray(Image.fromarray(mask.astype(np.uint8) * 255).resize(
                (rgb.shape[1], rgb.shape[0]), Image.NEAREST)) > 0
        return rgb, mask, None
    if it["kind"] == "yolo":
        lab = it["label"]
        return rgb, None, lab.read_text().splitlines() if lab.exists() else []
    return rgb, None, it["lines"]


def _resize(rgb: np.ndarray, mask: np.ndarray | None, max_side: int):
    h, w = rgb.shape[:2]
    s = max_side / max(h, w)
    if s >= 1:
        return rgb, mask
    size = (round(w * s), round(h * s))
    rgb = np.asarray(Image.fromarray(rgb).resize(size, Image.BILINEAR))
    if mask is not None:
        mask = np.asarray(Image.fromarray(mask.astype(np.uint8)).resize(size, Image.NEAREST)) > 0
    return rgb, mask


def _write_sample(rgb, mask, lines, name: str, split: str, out: Path, pcfg: dict) -> int:
    """Écrit image + label YOLO ; renvoie le nombre d'instances."""
    rgb, mask = _resize(rgb, mask, pcfg["max_side"])
    h, w = rgb.shape[:2]
    if mask is not None:
        polys = mask_to_polygons(mask, pcfg["min_instance_area_px"], pcfg["polygon_epsilon_px"])
        lines = polygons_to_yolo_lines(polys, w, h, 0)
    Image.fromarray(rgb).save(out / "images" / split / f"{name}.jpg", quality=95)
    (out / "labels" / split / f"{name}.txt").write_text("\n".join(lines) + ("\n" if lines else ""))
    return len(lines)


def _stratified_split(items: list[dict], ratios, seed: int, pcfg: dict) -> dict[str, str]:
    """Split reproductible par groupe, stratifié sur la part de surface malade
    (4 quartiles) pour que val/test couvrent petits et grands foyers."""
    from sklearn.model_selection import train_test_split

    groups = sorted({it["group"] for it in items if not it.get("split")})
    if not groups:
        return {}
    frac = {}
    for it in items:
        if it["group"] in groups and it["kind"] == "masks":
            frac[it["group"]] = float(read_binary_mask(it["mask"], pcfg["mask_threshold"]).mean())
    strat = None
    if len(frac) == len(groups):
        v = np.array([frac[g] for g in groups])
        strat = np.digitize(v, np.quantile(v, [0.25, 0.5, 0.75]))
    tr, va, te = ratios
    g_tr, g_rest, s_tr, s_rest = train_test_split(groups, strat if strat is not None else groups,
                                                  train_size=tr, random_state=seed,
                                                  stratify=strat)
    g_va, g_te = train_test_split(g_rest, train_size=va / (va + te), random_state=seed,
                                  stratify=s_rest if strat is not None else None)
    return {**{g: "train" for g in g_tr}, **{g: "val" for g in g_va}, **{g: "test" for g in g_te}}


def prepare_seg(cfg: dict) -> None:
    pcfg = cfg["prepare"]["seg"]
    seed = cfg["project"]["seed"]
    out = resolve(pcfg["out_dir"])
    if out.exists():
        log.info(f"Suppression de l'ancien dossier {out}")
        shutil.rmtree(out)
    for s in SPLITS:
        (out / "images" / s).mkdir(parents=True)
        (out / "labels" / s).mkdir(parents=True)

    items = collect_seg_items(cfg)
    if not items:
        sys.exit("[ERREUR] Aucune image de segmentation trouvée — vérifiez raw.seg_sources dans config.yaml")
    log.info(f"{len(items)} images de segmentation trouvées")
    assign = _stratified_split(items, pcfg["split"], seed, pcfg)

    stats = {s: Counter() for s in SPLITS}
    bad = []
    for k, it in enumerate(items):
        split = it.get("split") or assign[it["group"]]
        name = it["group"].replace("/", "__")
        try:
            rgb, mask, lines = _load_pair(it, pcfg)
            n = _write_sample(rgb, mask, lines, name, split, out, pcfg)
            # Rotations 90/180/270° hors-ligne (train uniquement, option vue aérienne)
            if split == "train" and pcfg.get("offline_rot90") and mask is not None:
                for r in (1, 2, 3):
                    _write_sample(np.rot90(rgb, r).copy(), np.rot90(mask, r).copy(), None,
                                  f"{name}_rot{90 * r}", split, out, pcfg)
                    stats[split]["images"] += 1
                    stats[split]["instances"] += n
        except Exception as e:  # noqa: BLE001
            bad.append(f"{it['img']} : {e}")
            continue
        stats[split]["images"] += 1
        stats[split]["instances"] += n
        stats[split]["sans_instance"] += int(n == 0)
        if (k + 1) % 100 == 0:
            log.info(f"  {k + 1}/{len(items)} images converties")

    # Variantes augmentées hors-ligne : train seulement, groupées par image d'origine
    aug = cfg["raw"].get("seg_augmented")
    if pcfg.get("use_augmented_in_train") and aug:
        train_stems = {it["group"].split("/", 1)[1] for it in items
                       if (it.get("split") or assign[it["group"]]) == "train"}
        adir, mdir = resolve(aug["images"]), resolve(aug["masks"])
        for p in sorted(adir.iterdir()):
            if p.stem.rsplit("_", 1)[0] not in train_stems:
                continue  # variante d'une image val/test -> exclue (fuite)
            m = _find_mask(mdir, p.stem)
            if m is None:
                continue
            it = {"img": p, "mask": m, "kind": "masks"}
            rgb, mask, _ = _load_pair(it, pcfg)
            n = _write_sample(rgb, mask, None, f"aug__{p.stem}", "train", out, pcfg)
            stats["train"]["images"] += 1
            stats["train"]["instances"] += n

    names = pcfg["class_names"]
    data_yaml = {"path": str(out.resolve()).replace("\\", "/"), "train": "images/train", "val": "images/val",
                 "test": "images/test", "nc": len(names), "names": {i: n for i, n in enumerate(names)}}
    (out / "data.yaml").write_text(yaml.safe_dump(data_yaml, sort_keys=False, allow_unicode=True), encoding="utf-8")

    _seg_previews(cfg, out)
    _write_seg_report(cfg, stats, bad, out)
    log.info(f"[OK] Jeu YOLO-seg prêt : {out / 'data.yaml'}")


def _seg_previews(cfg: dict, out: Path) -> None:
    """10 images annotées (relues depuis les .txt produits) pour contrôle visuel."""
    n = cfg["prepare"]["seg"]["n_preview"]
    vdir = resolve(cfg["project"]["outputs_dir"]) / "verification"
    if vdir.exists():
        shutil.rmtree(vdir)
    imgs = sorted((out / "images" / "train").glob("*.jpg")) + sorted((out / "images" / "val").glob("*.jpg"))
    imgs = [p for p in imgs if "_rot" not in p.stem and not p.stem.startswith("aug__")]
    random.Random(cfg["project"]["seed"]).shuffle(imgs)
    for p in imgs[:n]:
        lab = out / "labels" / p.parent.name / f"{p.stem}.txt"
        rgb = np.asarray(Image.open(p).convert("RGB"))
        polys = [q for _, q in yolo_lines_to_polygons(lab.read_text().splitlines(), rgb.shape[1], rgb.shape[0])]
        save_annotation_preview(rgb, polys, vdir / f"{p.stem}.png", title=f"{p.parent.name}/{p.name}")
    log.info(f"{min(n, len(imgs))} aperçus de vérification -> {vdir}")


def _write_seg_report(cfg, stats, bad, out: Path) -> None:
    L = ["# Préparation — jeu de segmentation (YOLO-seg)", "",
         f"- Sortie : `{out.relative_to(ROOT).as_posix()}`", f"- Classe(s) : {cfg['prepare']['seg']['class_names']}",
         f"- Split (train/val/test) : {cfg['prepare']['seg']['split']} — graine {cfg['project']['seed']}",
         f"- Variantes aug_data ajoutées au train : {cfg['prepare']['seg']['use_augmented_in_train']}",
         f"- Rotations 90° hors-ligne : {cfg['prepare']['seg']['offline_rot90']}", "",
         "| Split | Images | Instances | Images sans instance |", "|---|---|---|---|"]
    for s in SPLITS:
        L.append(f"| {s} | {stats[s]['images']} | {stats[s]['instances']} | {stats[s]['sans_instance']} |")
    if bad:
        L += ["", "## Fichiers en erreur", ""] + [f"- {b}" for b in bad]
    p = resolve(cfg["project"]["outputs_dir"]) / "prepare_seg_report.md"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text("\n".join(L) + "\n", encoding="utf-8")
    print("\n".join(L))


# =============================================================================
#  CLASSIFICATION
# =============================================================================
def load_cls_manifest(cfg: dict) -> pd.DataFrame:
    """Lit le manifest + les splits fournis directement dans le zip (sans extraction)."""
    zp = resolve(cfg["raw"]["cls_zip"])
    if not zp.exists():
        raise FileNotFoundError(f"Archive introuvable : {zp}")
    with zipfile.ZipFile(zp) as z:
        names = z.namelist()

        def rd(suffix):
            f = next((n for n in names if n.endswith(suffix)), None)
            if f is None:
                raise FileNotFoundError(f"{suffix} absent de {zp.name}")
            return pd.read_csv(io.BytesIO(z.read(f)))

        m = rd("dataset_manifest.csv")
        m["split"] = None
        for s in SPLITS:
            try:
                ids = set(rd(f"{s}_split.csv").image_id)
                m.loc[m.image_id.isin(ids), "split"] = s
            except FileNotFoundError:
                pass
        prefix = next(n for n in names if n.endswith("/images/") or "/images/" in n)
        prefix = prefix[: prefix.index("/images/")]
    m["zip_path"] = prefix + "/" + m.image_path
    return m


def build_cls_table(cfg: dict) -> pd.DataFrame:
    pc = cfg["prepare"]["cls"]
    seed = cfg["project"]["seed"]
    m = load_cls_manifest(cfg)
    m["classe"] = m.canonical_class.map(lambda c: normalize_class_name(c, pc.get("crop_aliases"),
                                                                       pc.get("class_aliases")))
    counts = m.classe.value_counts()
    keep = counts[counts >= pc["min_images_per_class"]].index
    log.info(f"{m.classe.nunique()} classes après harmonisation ; {len(keep)} gardées "
             f"(>= {pc['min_images_per_class']} images) couvrant {counts[keep].sum()}/{len(m)} images")
    m = m[m.classe.isin(keep)].copy()
    # Images sans split -> train (aucune fuite : identifiants uniques, dédoublonnés en amont)
    m.loc[m.split.isna(), "split"] = "train"
    # Garantit au moins 1 image par classe en val et en test (indexation cohérente des classes)
    rng = np.random.RandomState(seed)
    for c in keep:
        for s in ("val", "test"):
            if not ((m.classe == c) & (m.split == s)).any():
                idx = m.index[(m.classe == c) & (m.split == "train")]
                take = rng.choice(idx, size=max(1, len(idx) // 10), replace=False)
                m.loc[take, "split"] = s
    # Plafonnement des classes majoritaires en train (échantillonnage stratifié par source)
    cap = pc.get("max_images_per_class_train")
    if cap:
        tr = m[m.split == "train"]
        kept = []
        for c, g in tr.groupby("classe"):
            if len(g) <= cap:
                kept.append(g)
            else:
                # proportionnel aux sources pour conserver la diversité terrain / labo
                # (boucle explicite : groupby().apply() supprime la colonne de
                #  regroupement depuis pandas 3, ce qui perdrait la source)
                frac = cap / len(g)
                for _, x in g.groupby("source_dataset"):
                    kept.append(x.sample(max(1, int(round(len(x) * frac))), random_state=seed))
        m = pd.concat([pd.concat(kept), m[m.split != "train"]])
    m["dir"] = m.classe.map(safe_dirname)
    return m.sort_values(["split", "dir", "image_id"])


_ZIP: zipfile.ZipFile | None = None  # handle du zip propre à chaque processus


def _md_table(df: pd.DataFrame) -> str:
    """DataFrame -> tableau Markdown (sans dépendance à tabulate)."""
    cols = [df.index.name or ""] + [str(c) for c in df.columns]
    rows = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    rows += ["| " + " | ".join([str(i)] + [str(v) for v in r]) + " |" for i, r in zip(df.index, df.values)]
    return "\n".join(rows)


def _extract_one(args) -> str | None:
    """(worker) Extrait une image depuis le dossier local ou le zip, la redimensionne
    (côté court = size) et l'écrit en JPEG. Renvoie un message d'erreur ou None."""
    zip_path, zp, local, dst, size = args
    dst = Path(dst)
    if dst.exists():
        return None
    try:
        if local and Path(local).exists():
            im = Image.open(local)
        else:
            global _ZIP
            if _ZIP is None:
                _ZIP = zipfile.ZipFile(zip_path)  # un handle par processus
            im = Image.open(io.BytesIO(_ZIP.read(zp)))
        im = im.convert("RGB")
        w, h = im.size
        s = size / min(w, h)
        if s < 1:
            im = im.resize((max(1, round(w * s)), max(1, round(h * s))), Image.BILINEAR)
        dst.parent.mkdir(parents=True, exist_ok=True)
        im.save(dst, quality=92)
        return None
    except Exception as e:  # noqa: BLE001
        return f"{zp}: {e}"


def prepare_cls(cfg: dict) -> None:
    pc = cfg["prepare"]["cls"]
    out = resolve(pc["out_dir"])
    m = build_cls_table(cfg)
    zip_path = str(resolve(cfg["raw"]["cls_zip"]))
    local_dir = resolve(cfg["raw"]["cls_extracted_dir"])
    # Nettoie les dossiers de classes qui ne font plus partie de la sélection
    if out.exists():
        wanted = {(s, d) for s, d in zip(m.split, m.dir)}
        for s in SPLITS:
            for d in (out / s).glob("*") if (out / s).exists() else []:
                if (s, d.name) not in wanted:
                    shutil.rmtree(d)
    jobs = []
    for r in m.itertuples():
        fn = Path(r.image_path).name
        jobs.append((zip_path, r.zip_path, str(local_dir / fn), str(out / r.split / r.dir / f"{r.image_id}.jpg"),
                     pc["image_size"]))
    log.info(f"Extraction de {len(jobs)} images vers {out} ({pc['workers']} processus)…")
    errors = []
    with ProcessPoolExecutor(max_workers=pc["workers"]) as ex:
        for k, e in enumerate(ex.map(_extract_one, jobs, chunksize=64)):
            if e:
                errors.append(e)
            if (k + 1) % 5000 == 0:
                log.info(f"  {k + 1}/{len(jobs)}")
    # Fichiers devenus surnuméraires (sélection modifiée) -> supprimés
    wanted_files = {Path(j[3]).resolve() for j in jobs}
    for f in out.rglob("*.jpg"):
        if f.resolve() not in wanted_files:
            f.unlink()
    m["file"] = [j[3] for j in jobs]
    m = m[[Path(f).exists() for f in m.file]]
    m[["image_id", "split", "classe", "dir", "crop", "source_dataset", "file"]].to_csv(out / "index.csv", index=False)
    classes = sorted(m.dir.unique())
    (out / "classes.json").write_text(json.dumps({d: m[m.dir == d].classe.iloc[0] for d in classes},
                                                 ensure_ascii=False, indent=1), encoding="utf-8")
    _write_cls_report(cfg, m, errors, out)
    log.info(f"[OK] Jeu de classification prêt : {out} ({len(classes)} classes)")


def _write_cls_report(cfg, m: pd.DataFrame, errors, out: Path) -> None:
    t = m.pivot_table(index="classe", columns="split", values="image_id", aggfunc="count", fill_value=0)
    t = t.reindex(columns=[s for s in SPLITS if s in t.columns])
    src = m.pivot_table(index="source_dataset", columns="split", values="image_id", aggfunc="count", fill_value=0)
    src = src.reindex(columns=[s for s in SPLITS if s in src.columns])
    L = ["# Préparation — jeu de classification (diagnostic)", "",
         f"- Sortie : `{out.relative_to(ROOT).as_posix()}`", f"- Classes : {m.classe.nunique()}  |  Cultures : {m.crop.nunique()}",
         f"- Images : {len(m)}  ({', '.join(f'{s}={int((m.split == s).sum())}' for s in SPLITS)})",
         f"- Seuil min. d'images par classe : {cfg['prepare']['cls']['min_images_per_class']} ; "
         f"plafond train : {cfg['prepare']['cls']['max_images_per_class_train']}",
         f"- Erreurs d'extraction : {len(errors)}", "", "## Par source", "", _md_table(src), "",
         "## Par classe", "", _md_table(t)]
    if errors:
        L += ["", "## Erreurs", ""] + [f"- {e}" for e in errors[:50]]
    p = resolve(cfg["project"]["outputs_dir"]) / "prepare_cls_report.md"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text("\n".join(L) + "\n", encoding="utf-8")
    log.info(f"Rapport : {p}")


def main() -> None:
    ap = argparse.ArgumentParser(description="Préparation des jeux de données")
    ap.add_argument("--task", choices=["seg", "cls", "all"], default="seg")
    ap.add_argument("--config", default="config.yaml")
    a = ap.parse_args()
    cfg = load_config(a.config)
    set_seed(cfg["project"]["seed"])
    if a.task in ("seg", "all"):
        prepare_seg(cfg)
    if a.task in ("cls", "all"):
        prepare_cls(cfg)


if __name__ == "__main__":
    main()
