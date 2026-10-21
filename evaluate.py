"""
evaluate.py — Évaluation sur le jeu de TEST (jamais vu à l'entraînement).

  * --task seg : mAP50 / mAP50-95 (masques et boîtes), précision, rappel, F1,
                 IoU et Dice au pixel, erreur sur le % de surface infectée,
                 matrices de confusion, exemples vérité terrain vs prédiction,
                 rapport Markdown interprété.
  * --task cls : top-1 / top-5, F1 macro, rapport par classe / par source,
                 matrice de confusion, confusions les plus fréquentes.

Usage :
    python evaluate.py --task seg
    python evaluate.py --task cls
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import yaml
from PIL import Image

from src.conversion import yolo_lines_to_polygons
from src.postprocess import result_to_instances
from src.utils import get_device, get_logger, load_config, resolve, set_seed
from src.visualize import save_gt_vs_pred

log = get_logger()


def _plt():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt


# =============================================================================
#  SEGMENTATION
# =============================================================================
def _gt_mask(label: Path, w: int, h: int) -> tuple[np.ndarray, list[np.ndarray]]:
    polys = [p for _, p in yolo_lines_to_polygons(label.read_text().splitlines(), w, h)] if label.exists() else []
    m = np.zeros((h, w), np.uint8)
    if polys:
        cv2.fillPoly(m, [np.round(p).astype(np.int32) for p in polys], 1)
    return m.astype(bool), polys


def _pred_mask(res, h: int, w: int) -> np.ndarray:
    if res.masks is None or len(res.masks) == 0:
        return np.zeros((h, w), bool)
    m = res.masks.data.cpu().numpy().any(axis=0)
    if m.shape != (h, w):  # sécurité : remise à la taille d'origine
        m = cv2.resize(m.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST).astype(bool)
    return m


def evaluate_seg(cfg: dict, weights: Path, device: str) -> None:
    from ultralytics import YOLO

    data_dir = resolve(cfg["prepare"]["seg"]["out_dir"])
    data_yaml = data_dir / "data.yaml"
    d = yaml.safe_load(data_yaml.read_text(encoding="utf-8"))
    d["path"] = str(data_dir.resolve()).replace("\\", "/")
    data_yaml.write_text(yaml.safe_dump(d, sort_keys=False), encoding="utf-8")
    out = resolve(cfg["project"]["outputs_dir"]) / "evaluation" / "seg"
    out.mkdir(parents=True, exist_ok=True)
    model = YOLO(str(weights))
    scfg = cfg["seg"]

    # ---- 1) métriques COCO-style d'Ultralytics sur le split test
    log.info("Validation Ultralytics sur le split test…")
    met = model.val(data=str(data_yaml), split="test", imgsz=scfg["imgsz"], batch=scfg["batch"], device=device,
                    plots=True, project=str(resolve("runs") / "seg"), name="eval_test", exist_ok=True,
                    verbose=False)
    r = {
        "box_mAP50": float(met.box.map50), "box_mAP50_95": float(met.box.map),
        "box_precision": float(met.box.mp), "box_recall": float(met.box.mr),
        "mask_mAP50": float(met.seg.map50), "mask_mAP50_95": float(met.seg.map),
        "mask_precision": float(met.seg.mp), "mask_recall": float(met.seg.mr),
    }
    p, rc = r["mask_precision"], r["mask_recall"]
    r["mask_F1"] = 2 * p * rc / (p + rc) if p + rc else 0.0
    vdir = Path(met.save_dir)
    for f in ("confusion_matrix.png", "confusion_matrix_normalized.png", "MaskPR_curve.png", "BoxPR_curve.png",
              "MaskF1_curve.png"):
        if (vdir / f).exists():
            shutil.copy2(vdir / f, out / f)

    # ---- 2) métriques au pixel + surface infectée (enjeu métier : chiffrer la surface)
    log.info("Calcul IoU / Dice au pixel…")
    imgs = sorted((data_dir / "images" / "test").glob("*.jpg"))
    rows, cache = [], {}
    tp = fp = fn = tn = 0
    for ip in imgs:
        res = model.predict(str(ip), conf=scfg["conf"], iou=scfg["iou"], imgsz=scfg["imgsz"], device=device,
                            retina_masks=True, verbose=False)[0]
        h, w = res.orig_shape
        gt, gpolys = _gt_mask(data_dir / "labels" / "test" / f"{ip.stem}.txt", w, h)
        pr = _pred_mask(res, h, w)
        i_, u_ = (gt & pr).sum(), (gt | pr).sum()
        tp += i_
        fp += (pr & ~gt).sum()
        fn += (gt & ~pr).sum()
        tn += (~gt & ~pr).sum()
        iou = i_ / u_ if u_ else 1.0
        dice = 2 * i_ / (gt.sum() + pr.sum()) if (gt.sum() + pr.sum()) else 1.0
        rows.append({"image": ip.name, "iou": iou, "dice": dice, "pct_gt": 100 * gt.mean(),
                     "pct_pred": 100 * pr.mean(), "n_gt": len(gpolys),
                     "n_pred": 0 if res.masks is None else len(res.masks)})
        cache[ip.name] = (gpolys, result_to_instances(res))
    df = pd.DataFrame(rows)
    df.to_csv(out / "per_image_metrics.csv", index=False)
    r.update({
        "pixel_IoU_global": float(tp / (tp + fp + fn)) if tp + fp + fn else 0.0,
        "pixel_Dice_global": float(2 * tp / (2 * tp + fp + fn)) if tp + fp + fn else 0.0,
        "pixel_IoU_moyen_par_image": float(df.iou.mean()),
        "pixel_Dice_moyen_par_image": float(df.dice.mean()),
        "pixel_precision": float(tp / (tp + fp)) if tp + fp else 0.0,
        "pixel_recall": float(tp / (tp + fn)) if tp + fn else 0.0,
        "erreur_abs_moy_pct_surface": float((df.pct_pred - df.pct_gt).abs().mean()),
        "biais_pct_surface": float((df.pct_pred - df.pct_gt).mean()),
        "correlation_pct_surface": float(df[["pct_gt", "pct_pred"]].corr().iloc[0, 1]) if len(df) > 2 else None,
        "nb_images_test": len(df),
    })

    # ---- 3) matrice de confusion au pixel (lésion vs sain/fond)
    plt = _plt()
    cm = np.array([[tn, fp], [fn, tp]], dtype=np.float64)
    cmn = cm / cm.sum(axis=1, keepdims=True)
    fig, ax = plt.subplots(figsize=(4.8, 4.2))
    ax.imshow(cmn, cmap="Blues", vmin=0, vmax=1)
    for (i, j), v in np.ndenumerate(cmn):
        ax.text(j, i, f"{v:.1%}\n({int(cm[i, j]):,} px)".replace(",", " "), ha="center", va="center",
                color="white" if v > 0.5 else "black", fontsize=9)
    ax.set_xticks([0, 1], ["prédit sain", "prédit lésion"])
    ax.set_yticks([0, 1], ["réel sain", "réel lésion"])
    ax.set_title("Matrice de confusion au pixel (test)")
    fig.tight_layout()
    fig.savefig(out / "pixel_confusion_matrix.png", dpi=130)
    plt.close(fig)

    # surface réelle vs prédite
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.scatter(df.pct_gt, df.pct_pred, s=14, alpha=0.7)
    mx = max(df.pct_gt.max(), df.pct_pred.max(), 1)
    ax.plot([0, mx], [0, mx], "k--", lw=1)
    ax.set_xlabel("% surface malade réelle")
    ax.set_ylabel("% surface malade prédite")
    ax.set_title("Estimation de la surface infectée (test)")
    fig.tight_layout()
    fig.savefig(out / "surface_reelle_vs_predite.png", dpi=130)
    plt.close(fig)

    # ---- 4) exemples : 3 meilleurs, 3 médians, 3 pires (honnêteté du rapport)
    ex = out / "exemples"
    if ex.exists():
        shutil.rmtree(ex)
    s = df.sort_values("iou").reset_index(drop=True)
    picks = {"pire": s.head(3), "median": s.iloc[len(s) // 2 - 1: len(s) // 2 + 2], "meilleur": s.tail(3)}
    for tag, sub in picks.items():
        for row in sub.itertuples():
            rgb = np.asarray(Image.open(data_dir / "images" / "test" / row.image).convert("RGB"))
            g, pi = cache[row.image]
            save_gt_vs_pred(rgb, g, pi, ex / f"{tag}_{Path(row.image).stem}.png",
                            {"IoU": row.iou, "Dice": row.dice})

    (out / "metrics.json").write_text(json.dumps(r, indent=2, ensure_ascii=False), encoding="utf-8")
    write_seg_report(r, df, out, cfg, weights)


def _qual(v: float, bons=(0.5, 0.7)) -> str:
    return "faible" if v < bons[0] else "correct" if v < bons[1] else "bon"


def write_seg_report(r: dict, df: pd.DataFrame, out: Path, cfg: dict, weights: Path) -> None:
    ts = resolve(cfg["project"]["outputs_dir"]) / "training" / "seg" / "train_summary.json"
    tinfo = json.loads(ts.read_text(encoding="utf-8")) if ts.exists() else {}
    L = ["# Rapport d'évaluation — segmentation des lésions", "",
         f"- Modèle : `{weights}` ({tinfo.get('model', cfg['seg']['model'])})",
         f"- Jeu de test : {r['nb_images_test']} images jamais vues pendant l'entraînement",
         f"- Seuil de confiance : {cfg['seg']['conf']}", "",
         "## Résultats chiffrés", "", "| Métrique | Masques | Boîtes |", "|---|---|---|",
         f"| mAP50 | {r['mask_mAP50']:.3f} | {r['box_mAP50']:.3f} |",
         f"| mAP50-95 | {r['mask_mAP50_95']:.3f} | {r['box_mAP50_95']:.3f} |",
         f"| Précision | {r['mask_precision']:.3f} | {r['box_precision']:.3f} |",
         f"| Rappel | {r['mask_recall']:.3f} | {r['box_recall']:.3f} |", "",
         "| Métrique au pixel | Valeur |", "|---|---|",
         f"| IoU global | {r['pixel_IoU_global']:.3f} |",
         f"| Dice global | {r['pixel_Dice_global']:.3f} |",
         f"| IoU moyen par image | {r['pixel_IoU_moyen_par_image']:.3f} |",
         f"| Dice moyen par image | {r['pixel_Dice_moyen_par_image']:.3f} |",
         f"| Précision pixel | {r['pixel_precision']:.3f} |",
         f"| Rappel pixel | {r['pixel_recall']:.3f} |",
         f"| Erreur moyenne sur le % de surface infectée | {r['erreur_abs_moy_pct_surface']:.2f} points |",
         f"| Biais (prédit − réel) | {r['biais_pct_surface']:+.2f} points |",
         f"| Corrélation surface réelle / prédite | {r['correlation_pct_surface'] or 0:.3f} |", "",
         "## Interprétation en langage simple", ""]
    dice = r["pixel_Dice_global"]
    L.append(f"- **Qualité du détourage ({_qual(dice)})** : en moyenne, la zone malade prédite et la zone "
             f"réelle se recouvrent à {dice:.0%} (Dice). Au-delà de 70 % on considère le détourage fiable "
             f"pour un usage opérationnel ; entre 50 et 70 % il est utile mais imprécis sur les bords.")
    L.append(f"- **Fausses alertes vs oublis** : la précision pixel est de {r['pixel_precision']:.0%} "
             f"(part des pixels signalés qui sont vraiment malades) et le rappel de {r['pixel_recall']:.0%} "
             f"(part des pixels malades retrouvés). " +
             ("Le modèle **oublie** plus qu'il ne sur-détecte : baisser le seuil de confiance augmenterait le rappel."
              if r["pixel_recall"] < r["pixel_precision"] else
              "Le modèle **sur-détecte** un peu : monter le seuil de confiance réduirait les fausses alertes."))
    L.append(f"- **Chiffrage de la surface touchée** : l'écart moyen entre le % de surface infectée estimé et "
             f"le vrai est de {r['erreur_abs_moy_pct_surface']:.1f} points (biais {r['biais_pct_surface']:+.1f}). "
             f"C'est l'indicateur métier le plus direct : il dit à quel point la carte des foyers chiffre "
             f"correctement les dégâts.")
    L.append(f"- **mAP50 masques = {r['mask_mAP50']:.2f}** : mesure la capacité à retrouver chaque foyer "
             f"*individuellement*. Elle est structurellement plus basse que le Dice car les annotations "
             f"regroupent souvent plusieurs taches en une seule zone (ou l'inverse) : le modèle peut bien "
             f"couvrir la surface malade tout en découpant les foyers autrement que l'annotateur.")
    worst = df.sort_values("iou").head(3).image.tolist()
    L += ["", f"Les cas les plus difficiles ({', '.join(worst)}) sont visibles dans `exemples/pire_*.png`.", "",
          "## Améliorations concrètes proposées", "",
          "1. **Plus de données annotées** : 588 images est un petit jeu ; doubler le volume apporte en général "
          "plus que tout réglage d'hyperparamètres. Pré-annoter avec ce modèle puis corriger (CVAT/Roboflow) "
          "divise le temps d'annotation.",
          "2. **Modèle plus grand sur GPU** : passer de `yolo11n-seg` à `yolo11s-seg` / `yolo11m-seg` et "
          "`imgsz` 1024 (Colab) améliore le détourage des petites taches.",
          "3. **Harmoniser la granularité des annotations** : certaines images annotent la feuille entière, "
          "d'autres chaque tache. Un guide d'annotation unique réduirait le bruit des étiquettes.",
          "4. **Ajuster le seuil de confiance** à partir de `MaskF1_curve.png` (le pic de F1 donne le "
          "meilleur compromis précision/rappel).",
          "5. **Vue aérienne** : pour l'usage drone, annoter quelques centaines de tuiles d'orthomosaïques "
          "réelles et affiner ce modèle (fine-tuning) — le changement de point de vue est le principal "
          "risque de dégradation.", "",
          "## Fichiers", "",
          "- `metrics.json`, `per_image_metrics.csv`",
          "- `confusion_matrix*.png` (instances), `pixel_confusion_matrix.png`",
          "- `MaskPR_curve.png`, `MaskF1_curve.png`, `surface_reelle_vs_predite.png`",
          "- `exemples/` : vérité terrain (vert) vs prédiction (rouge)"]
    (out / "rapport_evaluation.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    log.info(f"[OK] Rapport : {out / 'rapport_evaluation.md'}")
    print(json.dumps(r, indent=2, ensure_ascii=False))


# =============================================================================
#  CLASSIFICATION
# =============================================================================
def evaluate_cls(cfg: dict, weights: Path, device: str) -> None:
    from sklearn.metrics import classification_report, confusion_matrix, f1_score
    from ultralytics import YOLO

    data_dir = resolve(cfg["prepare"]["cls"]["out_dir"])
    idx = pd.read_csv(data_dir / "index.csv")
    test = idx[idx.split == "test"].reset_index(drop=True)
    pretty = json.loads((data_dir / "classes.json").read_text(encoding="utf-8"))
    out = resolve(cfg["project"]["outputs_dir"]) / "evaluation" / "cls"
    out.mkdir(parents=True, exist_ok=True)
    model = YOLO(str(weights))
    names = model.names  # index -> nom de dossier
    ccfg = cfg["cls"]

    log.info(f"Prédiction sur {len(test)} images de test…")
    preds, top5 = [], []
    files = test.file.tolist()
    for k in range(0, len(files), 64):
        for res in model.predict(files[k:k + 64], imgsz=ccfg["imgsz"], device=device, verbose=False):
            pr = res.probs
            preds.append(names[int(pr.top1)])
            top5.append([names[int(i)] for i in pr.top5])
    test["pred"] = preds
    test["ok"] = test.pred == test.dir
    test["ok_top5"] = [t in p for t, p in zip(test.dir, top5)]
    test["pred_classe"] = test.pred.map(pretty)
    test.to_csv(out / "predictions_test.csv", index=False)

    r = {"top1": float(test.ok.mean()), "top5": float(test.ok_top5.mean()),
         "f1_macro": float(f1_score(test.dir, test.pred, average="macro")),
         "f1_pondere": float(f1_score(test.dir, test.pred, average="weighted")),
         "nb_images_test": len(test), "nb_classes": int(test.dir.nunique())}
    by_src = test.groupby("source_dataset").ok.agg(["mean", "size"]).sort_values("mean")
    by_crop = test.groupby("crop").ok.agg(["mean", "size"]).sort_values("mean")
    r["accuracy_par_source"] = {k: round(v, 4) for k, v in by_src["mean"].items()}
    rep = classification_report(test.dir, test.pred, output_dict=True, zero_division=0)
    per_cls = pd.DataFrame({pretty.get(k, k): v for k, v in rep.items() if k in pretty}).T
    per_cls.sort_values("f1-score").to_csv(out / "rapport_par_classe.csv")
    conf = (test[~test.ok].groupby(["classe", "pred_classe"]).size().sort_values(ascending=False).head(15))

    # matrice de confusion normalisée (classes triées par culture)
    labels = sorted(test.dir.unique())
    cm = confusion_matrix(test.dir, test.pred, labels=labels).astype(float)
    cm /= np.maximum(cm.sum(axis=1, keepdims=True), 1)
    plt = _plt()
    n = len(labels)
    fig, ax = plt.subplots(figsize=(max(8, n * 0.18), max(7, n * 0.18)))
    ax.imshow(cm, cmap="viridis", vmin=0, vmax=1)
    ax.set_xticks(range(n), [pretty[c] for c in labels], rotation=90, fontsize=max(3, 9 - n // 20))
    ax.set_yticks(range(n), [pretty[c] for c in labels], fontsize=max(3, 9 - n // 20))
    ax.set_xlabel("prédit")
    ax.set_ylabel("réel")
    ax.set_title(f"Matrice de confusion normalisée (test) — top-1 = {r['top1']:.1%}")
    fig.tight_layout()
    fig.savefig(out / "confusion_matrix.png", dpi=150)
    plt.close(fig)
    (out / "metrics.json").write_text(json.dumps(r, indent=2, ensure_ascii=False), encoding="utf-8")

    L = ["# Rapport d'évaluation — diagnostic (classification)", "",
         f"- Modèle : `{weights}`", f"- Test : {r['nb_images_test']} images, {r['nb_classes']} classes", "",
         "| Métrique | Valeur |", "|---|---|",
         f"| Exactitude top-1 | {r['top1']:.3f} |", f"| Exactitude top-5 | {r['top5']:.3f} |",
         f"| F1 macro (chaque classe compte autant) | {r['f1_macro']:.3f} |",
         f"| F1 pondéré | {r['f1_pondere']:.3f} |", "",
         "## Par source d'images", "", "| Source | Exactitude | Images |", "|---|---|---|"]
    L += [f"| {k} | {v['mean']:.3f} | {int(v['size'])} |" for k, v in by_src.iterrows()]
    L += ["", "## Par culture", "", "| Culture | Exactitude | Images |", "|---|---|---|"]
    L += [f"| {k} | {v['mean']:.3f} | {int(v['size'])} |" for k, v in by_crop.iterrows()]
    L += ["", "## Confusions les plus fréquentes", "", "| Réel | Prédit | Nombre |", "|---|---|---|"]
    L += [f"| {a} | {b} | {n_} |" for (a, b), n_ in conf.items()]
    lab_src = [s for s in by_src.index if "PlantVillage" in s]
    gap = (by_src.loc[lab_src[0], "mean"] - test[~test.source_dataset.isin(lab_src)].ok.mean()) if lab_src else None
    L += ["", "## Interprétation en langage simple", "",
          f"- Sur 100 photos jamais vues, le modèle donne la bonne maladie en premier choix "
          f"{100 * r['top1']:.0f} fois, et la bonne réponse figure dans ses 5 propositions "
          f"{100 * r['top5']:.0f} fois.",
          f"- Le F1 macro ({r['f1_macro']:.2f}) traite toutes les maladies à égalité : s'il est nettement "
          f"plus bas que l'exactitude, ce sont les maladies rares qui sont mal reconnues."]
    if gap is not None:
        L.append(f"- Les photos de laboratoire (PlantVillage, fond uni) sont mieux reconnues que les photos "
                 f"terrain (écart ≈ {100 * gap:.0f} points) : c'est l'écart attendu entre conditions "
                 f"contrôlées et réelles, à surveiller pour un usage au champ.")
    L += ["- Les pires classes sont listées dans `rapport_par_classe.csv` (triées par F1).", "",
          "## Améliorations concrètes", "",
          "1. Entraîner `yolo11s-cls` en 224→320 px sur GPU (Colab) et relever le plafond d'images par classe.",
          "2. Collecter des photos terrain pour les classes uniquement issues de PlantVillage.",
          "3. Fusionner les classes confondues de façon systématique si l'agronomie le justifie, "
          "ou ajouter un niveau hiérarchique (culture puis maladie).",
          "4. Calibrer les probabilités (température) avant d'afficher un pourcentage de confiance à un utilisateur."]
    (out / "rapport_evaluation.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    log.info(f"[OK] Rapport : {out / 'rapport_evaluation.md'}")
    print(json.dumps({k: v for k, v in r.items() if k != "accuracy_par_source"}, indent=2))


def main() -> None:
    ap = argparse.ArgumentParser(description="Évaluation sur le jeu de test")
    ap.add_argument("--task", choices=["seg", "cls"], default="seg")
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--weights", help="défaut : models/best.pt (seg) ou models/cls_best.pt (cls)")
    ap.add_argument("--device")
    a = ap.parse_args()
    cfg = load_config(a.config)
    set_seed(cfg["project"]["seed"])
    w = resolve(a.weights or (cfg["predict"]["seg_weights"] if a.task == "seg" else cfg["predict"]["cls_weights"]))
    if not w.exists():
        sys.exit(f"[ERREUR] Poids introuvables : {w} — lancez d'abord train.py --task {a.task}")
    device = a.device or get_device(cfg["project"]["device"])
    (evaluate_seg if a.task == "seg" else evaluate_cls)(cfg, w, device)


if __name__ == "__main__":
    main()
