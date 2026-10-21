"""
train.py — Entraînement supervisé par transfer learning (Ultralytics YOLO).

  * --task seg : segmentation d'instance des lésions (YOLO11/YOLOv8-seg pré-entraîné COCO)
  * --task cls : diagnostic de la maladie (YOLO11-cls pré-entraîné ImageNet)

Early stopping, sauvegarde des meilleurs poids, courbes d'apprentissage,
copie dans models/ et export ONNX.

Usage :
    python train.py --task seg
    python train.py --task cls
    python train.py --task seg --epochs 3 --fraction 0.1   # test rapide (smoke test)
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from pathlib import Path

import pandas as pd
import yaml

from src.utils import get_device, get_logger, in_colab, load_config, resolve, set_seed

log = get_logger()


def _refresh_data_yaml(data_dir: Path) -> Path:
    """Réécrit `path:` dans data.yaml pour qu'il pointe vers l'emplacement actuel
    (le projet peut avoir été déplacé ou copié sur Colab)."""
    p = data_dir / "data.yaml"
    if not p.exists():
        sys.exit(f"[ERREUR] {p} introuvable — lancez d'abord : python prepare_dataset.py --task seg")
    d = yaml.safe_load(p.read_text(encoding="utf-8"))
    d["path"] = str(data_dir.resolve()).replace("\\", "/")
    p.write_text(yaml.safe_dump(d, sort_keys=False, allow_unicode=True), encoding="utf-8")
    return p


def plot_curves(results_csv: Path, out_png: Path, task: str) -> None:
    """Courbes d'apprentissage lisibles : pertes train/val + métriques de validation."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    df = pd.read_csv(results_csv)
    df.columns = [c.strip() for c in df.columns]
    losses = [c for c in df.columns if c.startswith(("train/", "val/")) and "loss" in c]
    metrics = [c for c in df.columns if c.startswith("metrics/")]
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for c in losses:
        axes[0].plot(df["epoch"], df[c], label=c, ls="-" if c.startswith("train") else "--")
    axes[0].set_title("Pertes (trait plein = train, pointillés = validation)")
    axes[0].set_xlabel("époque")
    axes[0].legend(fontsize=7)
    axes[0].grid(alpha=0.3)
    for c in metrics:
        axes[1].plot(df["epoch"], df[c], label=c.replace("metrics/", ""))
    axes[1].set_title("Métriques de validation")
    axes[1].set_xlabel("époque")
    axes[1].legend(fontsize=7)
    axes[1].grid(alpha=0.3)
    fig.suptitle(f"Courbes d'apprentissage — {task}")
    fig.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=120)
    plt.close(fig)


def train(task: str, cfg: dict, args: argparse.Namespace) -> None:
    from ultralytics import YOLO

    tcfg = cfg[task]
    seed = cfg["project"]["seed"]
    set_seed(seed)
    device = args.device or get_device(cfg["project"]["device"])
    log.info(f"Tâche : {task} | device : {device} | Colab : {in_colab()}")
    if device == "cpu":
        log.warning("Entraînement sur CPU : fonctionnel mais lent. GPU recommandé (Google Colab, voir README).")

    if task == "seg":
        data = str(_refresh_data_yaml(resolve(cfg["prepare"]["seg"]["out_dir"])))
    else:
        data_dir = resolve(cfg["prepare"]["cls"]["out_dir"])
        if not (data_dir / "train").is_dir():
            sys.exit(f"[ERREUR] {data_dir}/train introuvable — lancez : python prepare_dataset.py --task cls")
        data = str(data_dir)

    runs_dir = resolve("runs") / task
    name = tcfg["run_name"]
    hp = dict(
        data=data, epochs=args.epochs or tcfg["epochs"], patience=tcfg["patience"],
        imgsz=args.imgsz or tcfg["imgsz"], batch=args.batch or tcfg["batch"], workers=tcfg["workers"],
        device=device, seed=seed, deterministic=True, optimizer=tcfg["optimizer"], lr0=tcfg["lr0"],
        cos_lr=tcfg["cos_lr"], project=str(runs_dir), name=name, exist_ok=True, plots=True,
        fraction=args.fraction, verbose=True, **tcfg["augment"],
    )
    if task == "seg":
        hp.update(lrf=tcfg["lrf"], weight_decay=tcfg["weight_decay"], warmup_epochs=tcfg["warmup_epochs"],
                  close_mosaic=tcfg["close_mosaic"])
    else:
        hp.update(dropout=tcfg["dropout"])

    model_name = args.model or tcfg["model"]
    log.info(f"Modèle pré-entraîné : {model_name}")
    model = YOLO(model_name)
    t0 = time.time()
    model.train(**hp)
    dur = time.time() - t0

    run_dir = Path(model.trainer.save_dir)
    best = run_dir / "weights" / "best.pt"
    if not best.exists():
        sys.exit(f"[ERREUR] Poids introuvables : {best}")

    # ---- copie des meilleurs poids + export ONNX
    mdir = resolve(cfg["project"]["models_dir"])
    mdir.mkdir(parents=True, exist_ok=True)
    stem = "best" if task == "seg" else "cls_best"
    dst = mdir / f"{stem}.pt"
    shutil.copy2(best, dst)
    log.info(f"Meilleurs poids -> {dst}")
    if task == "cls":  # noms lisibles des classes, utilisés par predict.py / app.py
        shutil.copy2(Path(data) / "classes.json", mdir / "cls_classes.json")
    try:
        onnx = YOLO(str(dst)).export(format="onnx", imgsz=hp["imgsz"], simplify=True, dynamic=False)
        shutil.move(str(onnx), mdir / f"{stem}.onnx")
        log.info(f"Export ONNX -> {mdir / f'{stem}.onnx'}")
    except Exception as e:  # noqa: BLE001
        log.error(f"Export ONNX impossible ({e}). Installez : pip install onnx onnxslim")

    # ---- courbes + résumé
    odir = resolve(cfg["project"]["outputs_dir"]) / "training" / task
    odir.mkdir(parents=True, exist_ok=True)
    rcsv = run_dir / "results.csv"
    summary = {"task": task, "model": model_name, "device": device, "duree_min": round(dur / 60, 1),
               "run_dir": str(run_dir), "weights": str(dst), "hyperparametres": {k: v for k, v in hp.items()
                                                                                    if k != "data"}}
    if rcsv.exists():
        plot_curves(rcsv, odir / "learning_curves.png", task)
        shutil.copy2(rcsv, odir / "results.csv")
        df = pd.read_csv(rcsv)
        df.columns = [c.strip() for c in df.columns]
        key = "metrics/mAP50-95(M)" if task == "seg" else "metrics/accuracy_top1"
        if key in df:
            b = df[key].idxmax()
            summary.update(epochs_effectuees=int(df["epoch"].max()), meilleure_epoque=int(df.loc[b, "epoch"]),
                           meilleures_metriques_val={c: round(float(df.loc[b, c]), 4)
                                                     for c in df.columns if c.startswith("metrics/")})
    for f in ("results.png", "confusion_matrix_normalized.png", "MaskPR_curve.png", "labels.jpg"):
        if (run_dir / f).exists():
            shutil.copy2(run_dir / f, odir / f)
    (odir / "train_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    log.info(f"[OK] Entraînement terminé en {dur / 60:.1f} min — résumé : {odir / 'train_summary.json'}")


def main() -> None:
    ap = argparse.ArgumentParser(description="Entraînement YOLO (transfer learning)")
    ap.add_argument("--task", choices=["seg", "cls"], default="seg")
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--model", help="surcharge du modèle (ex. yolo11s-seg.pt)")
    ap.add_argument("--epochs", type=int)
    ap.add_argument("--batch", type=int)
    ap.add_argument("--imgsz", type=int)
    ap.add_argument("--device", help="cpu | 0 | mps (défaut : auto)")
    ap.add_argument("--fraction", type=float, default=1.0, help="fraction du train utilisée (tests rapides)")
    a = ap.parse_args()
    train(a.task, load_config(a.config), a)


if __name__ == "__main__":
    main()
