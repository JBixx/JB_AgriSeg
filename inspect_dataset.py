"""
inspect_dataset.py — Exploration et diagnostic du dossier de données brutes.

Détecte automatiquement les formats présents (YOLO segmentation, COCO, masques PNG,
dossiers de classification), compte les images / annotations / classes, repère les
fichiers corrompus ou orphelins, puis rend un verdict : les données sont-elles
exploitables pour de la SEGMENTATION D'INSTANCE ?

Usage :
    python inspect_dataset.py                       # cherche dataset/ ou Dataset/dataset
    python inspect_dataset.py --root chemin/vers/donnees --out outputs/rapport_inspection.md
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from PIL import Image

IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
MASK_HINTS = ("mask", "masks", "label", "labels", "annotation", "annotations", "gt", "groundtruth")


def trouver_racine(arg: str | None) -> Path:
    """Retourne le dossier de données à inspecter (argument ou emplacements par défaut)."""
    candidats = [Path(arg)] if arg else [Path("dataset"), Path("Dataset/dataset"), Path("Dataset")]
    for c in candidats:
        if c.is_dir():
            return c
    sys.exit(f"[ERREUR] Aucun dossier de données trouvé parmi : {[str(c) for c in candidats]}")


def est_masque(p: Path) -> bool:
    """Heuristique : une image est un masque si son chemin contient un mot-clé de masque."""
    parts = [x.lower() for x in p.parts[:-1]] + [p.stem.lower()]
    return any(h in part for part in parts for h in MASK_HINTS)


def lire_yolo_txt(p: Path) -> tuple[str, list[int]]:
    """Analyse un .txt YOLO : renvoie ('seg'|'bbox'|'invalide'|'vide', classes)."""
    classes, types = [], set()
    for ligne in p.read_text(encoding="utf-8", errors="ignore").splitlines():
        v = ligne.split()
        if not v:
            continue
        try:
            nums = [float(x) for x in v]
        except ValueError:
            return "invalide", []
        classes.append(int(nums[0]))
        n = len(nums) - 1
        types.add("bbox" if n == 4 else "seg" if n >= 6 and n % 2 == 0 else "invalide")
    if not classes:
        return "vide", []
    return ("seg" if types == {"seg"} else "bbox" if types == {"bbox"} else "invalide"), classes


def inspecter(root: Path) -> dict:
    fichiers = [p for p in root.rglob("*") if p.is_file()]
    images = [p for p in fichiers if p.suffix.lower() in IMG_EXT and not est_masque(p)]
    masques = [p for p in fichiers if p.suffix.lower() in IMG_EXT and est_masque(p)]
    txts = [p for p in fichiers if p.suffix.lower() == ".txt" and p.name.lower() not in {"readme.txt", "classes.txt"}]
    jsons = [p for p in fichiers if p.suffix.lower() == ".json"]
    autres = Counter(p.suffix.lower() or "(sans ext.)" for p in fichiers
                     if p.suffix.lower() not in IMG_EXT | {".txt", ".json"})

    r: dict = {"root": str(root), "n_fichiers": len(fichiers), "autres_ext": dict(autres), "formats": []}

    # --- Intégrité et tailles des images
    tailles, corrompus = Counter(), []
    for p in images:
        try:
            with Image.open(p) as im:
                im.verify()
            with Image.open(p) as im:
                tailles[f"{im.size[0]}x{im.size[1]}"] += 1
        except Exception as e:  # noqa: BLE001
            corrompus.append(f"{p} ({e.__class__.__name__})")
    r.update(n_images=len(images), tailles=dict(tailles), corrompus=corrompus)

    # --- Format YOLO (labels .txt au même nom que l'image)
    stems_txt = {p.stem: p for p in txts}
    yolo_types, yolo_cls = Counter(), Counter()
    for p in txts:
        t, cls = lire_yolo_txt(p)
        yolo_types[t] += 1
        yolo_cls.update(cls)
    if txts and (yolo_types["seg"] or yolo_types["bbox"]):
        r["formats"].append("YOLO-segmentation" if yolo_types["seg"] else "YOLO-détection (boîtes seulement)")
    r["yolo_types"], r["yolo_classes"] = dict(yolo_types), dict(yolo_cls)

    # --- Format COCO
    coco_ann, coco_seg, coco_cls = 0, 0, Counter()
    for p in jsons:
        try:
            d = json.loads(p.read_text(encoding="utf-8"))
        except Exception:  # noqa: BLE001
            continue
        if isinstance(d, dict) and "annotations" in d and "images" in d:
            noms = {c["id"]: c.get("name", str(c["id"])) for c in d.get("categories", [])}
            for a in d["annotations"]:
                coco_ann += 1
                coco_cls[noms.get(a.get("category_id"), "?")] += 1
                if a.get("segmentation"):
                    coco_seg += 1
    if coco_ann:
        r["formats"].append("COCO-segmentation" if coco_seg else "COCO-détection (boîtes seulement)")
    r.update(coco_annotations=coco_ann, coco_avec_masque=coco_seg, coco_classes=dict(coco_cls))

    # --- Masques PNG : appariement image/masque + statistiques des lésions
    if masques:
        r["formats"].append("Masques PNG (segmentation)")
    r["n_masques"] = len(masques)
    r["masques"] = stats_masques(masques, images)

    # --- Dossiers de classification (images rangées par sous-dossier = classe)
    par_dossier = defaultdict(int)
    for p in images:
        par_dossier[p.parent.name] += 1
    par_split = defaultdict(lambda: defaultdict(int))
    for p in images:
        split = p.parent.parent.name if p.parent.parent != root else "-"
        par_split[split][p.parent.name] += 1
    r["par_split"] = {s: dict(c) for s, c in par_split.items()}
    if images and not txts and not coco_ann and not masques and len(par_dossier) > 1:
        r["formats"].append("Classification (dossier = classe, AUCUN masque)")

    # --- Images sans annotation / annotations orphelines (pertinent pour YOLO)
    if txts:
        stems_img = {p.stem for p in images}
        r["images_sans_label"] = len([p for p in images if p.stem not in stems_txt])
        r["labels_orphelins"] = len([s for s in stems_txt if s not in stems_img])

    # --- Archives zip contenant un manifest (jeu agrégé multi-sources)
    r["zips"] = [inspecter_zip(p) for p in fichiers if p.suffix.lower() == ".zip"]
    for z in r["zips"]:
        if z.get("n_images"):
            r["formats"].append(f"Archive {z['nom']} : classification multi-sources "
                                f"({z['n_images']} images, masques : {z['n_masques']})")

    r["exploitable_segmentation"] = bool(yolo_types["seg"] or coco_seg or masques)
    return r


def stats_masques(masques: list[Path], images: list[Path]) -> dict:
    """Apparie masques et images (même nom, dossier frère) et mesure les lésions."""
    from scipy import ndimage

    from src.conversion import read_binary_mask

    by_dir = defaultdict(set)
    for p in images:
        by_dir[p.parent.parent].add(p.stem)
    orph, fr, nc, vides = 0, [], [], 0
    for m in masques:
        if m.stem not in by_dir.get(m.parent.parent, set()):
            orph += 1
            continue
        b = read_binary_mask(m)
        fr.append(float(b.mean()))
        lab, _ = ndimage.label(b)
        nc.append(int((np.bincount(lab.ravel())[1:] >= 30).sum()))
        vides += int(not b.any())
    if not fr:
        return {}
    return {"masques_orphelins": orph, "masques_vides": vides,
            "pct_surface_malade_moy": round(100 * float(np.mean(fr)), 2),
            "pct_surface_malade_med": round(100 * float(np.median(fr)), 2),
            "foyers_par_image_med": float(np.median(nc)), "foyers_par_image_max": int(np.max(nc)),
            "total_foyers": int(np.sum(nc))}


def inspecter_zip(p: Path) -> dict:
    """Résume une archive zip contenant un dataset_manifest.csv (sans l'extraire)."""
    import io
    import zipfile

    import pandas as pd

    out = {"nom": p.name}
    try:
        with zipfile.ZipFile(p) as z:
            names = z.namelist()
            out["n_fichiers"] = len(names)
            man = next((n for n in names if n.endswith("dataset_manifest.csv")), None)
            if man is None:
                return out
            m = pd.read_csv(io.BytesIO(z.read(man)))
    except zipfile.BadZipFile:
        out["erreur"] = "archive corrompue"
        return out
    out.update(n_images=len(m), n_classes=int(m.canonical_class.nunique()), n_cultures=int(m.crop.nunique()),
               n_masques=int(m.mask_path.notna().sum()) if "mask_path" in m else 0,
               sources=m.source_dataset.value_counts().to_dict(),
               classes_moins_de_50=int((m.canonical_class.value_counts() < 50).sum()))
    return out


def rapport_markdown(r: dict) -> str:
    L = ["# Rapport d'inspection du jeu de données", "", f"**Dossier inspecté :** `{r['root']}`", ""]
    L += ["## Résumé", "",
          f"- Fichiers : {r['n_fichiers']} (autres extensions : {r['autres_ext'] or 'aucune'})",
          f"- Images : {r['n_images']}",
          f"- Formats détectés : {', '.join(r['formats']) or 'aucun format reconnu'}",
          f"- Tailles d'images : {r['tailles']}",
          f"- Fichiers corrompus : {len(r['corrompus'])}",
          f"- Labels YOLO (.txt) : {r['yolo_types'] or 'aucun'}",
          f"- Annotations COCO : {r['coco_annotations']} (dont avec masque : {r['coco_avec_masque']})",
          f"- Masques PNG : {r['n_masques']}", ""]
    if "images_sans_label" in r:
        L += [f"- Images sans annotation : {r['images_sans_label']}",
              f"- Labels orphelins : {r['labels_orphelins']}", ""]
    L += ["## Répartition des images par split et par dossier/classe", "", "| Split | Classe | Images |", "|---|---|---|"]
    for s, d in r["par_split"].items():
        for c, n in sorted(d.items()):
            L.append(f"| {s} | {c} | {n} |")
    if r.get("masques"):
        L += ["", "## Masques de segmentation", ""] + [f"- {k} : {v}" for k, v in r["masques"].items()]
    for z in r.get("zips", []):
        L += ["", f"## Archive `{z['nom']}`", ""] + [f"- {k} : {v}" for k, v in z.items() if k != "nom"]
    if r["corrompus"]:
        L += ["", "## Fichiers corrompus", ""] + [f"- {c}" for c in r["corrompus"]]
    L += ["", "## Verdict", ""]
    if r["exploitable_segmentation"]:
        L.append("✅ Des annotations de segmentation ont été trouvées : on peut passer à `prepare_dataset.py`.")
    else:
        L.append("❌ **Données NON exploitables pour la segmentation d'instance** : aucun polygone YOLO, "
                 "aucune segmentation COCO et aucun masque PNG. Ces images ne portent qu'une étiquette "
                 "globale (classification). Un modèle YOLO-seg ne peut pas apprendre à détourer des zones "
                 "sans exemples de contours.")
    return "\n".join(L) + "\n"


def main() -> None:
    ap = argparse.ArgumentParser(description="Inspection du dossier de données brutes")
    ap.add_argument("--root", default=None, help="dossier des données (défaut : dataset/ ou Dataset/dataset)")
    ap.add_argument("--out", default="outputs/rapport_inspection.md")
    a = ap.parse_args()

    r = inspecter(trouver_racine(a.root))
    md = rapport_markdown(r)
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(md, encoding="utf-8")
    print(md)
    print(f"[OK] Rapport écrit dans {out}")
    sys.exit(0 if r["exploitable_segmentation"] else 2)  # code 2 = arrêt du pipeline


if __name__ == "__main__":
    main()
