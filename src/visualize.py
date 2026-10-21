"""Visualisations : vérification des annotations, carte des foyers, comparaison
vérité terrain / prédiction."""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # rendu sans écran (serveur, Colab, Streamlit)
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402
from matplotlib.patches import Polygon as MplPolygon  # noqa: E402

PALETTE = ["#e6194b", "#f58231", "#ffe119", "#911eb4", "#42d4f4", "#f032e6", "#4363d8",
           "#9a6324", "#800000", "#000075", "#fabed4", "#dcbeff", "#aaffc3", "#ffd8b1"]


def _rings(geom):
    """Contours extérieurs d'un Polygon/MultiPolygon shapely."""
    polys = geom.geoms if geom.geom_type == "MultiPolygon" else [geom]
    return [np.asarray(p.exterior.coords) for p in polys]


def _fit(rgb: np.ndarray, max_side: int) -> tuple[np.ndarray, float]:
    """Réduit l'image pour l'affichage ; renvoie (image, facteur d'échelle)."""
    h, w = rgb.shape[:2]
    s = min(1.0, max_side / max(h, w))
    if s < 1.0:
        from PIL import Image

        rgb = np.asarray(Image.fromarray(rgb).resize((int(w * s), int(h * s)), Image.BILINEAR))
    return rgb, s


def draw_polygons(ax, polys, color="#e6194b", alpha=0.45, lw=1.2):
    for p in polys:
        ax.add_patch(MplPolygon(p, closed=True, fc=color, ec=color, alpha=alpha, lw=0))
        ax.add_patch(MplPolygon(p, closed=True, fill=False, ec=color, lw=lw))


def save_annotation_preview(rgb: np.ndarray, polys: list[np.ndarray], out: Path, title: str = "") -> None:
    """Image + polygones YOLO superposés (contrôle visuel de la conversion)."""
    fig, axes = plt.subplots(1, 2, figsize=(10, 5.2))
    for ax in axes:
        ax.imshow(rgb)
        ax.axis("off")
    draw_polygons(axes[1], polys)
    axes[0].set_title("Image")
    axes[1].set_title(f"Annotations YOLO-seg ({len(polys)} instances)")
    fig.suptitle(title, fontsize=9)
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=110)
    plt.close(fig)


def render_map(rgb: np.ndarray, instances, summary: dict, out: Path, max_side: int = 3000,
               max_legend: int = 15, title: str = "Carte des foyers de maladie") -> None:
    """Carte : zones infectées colorées et numérotées + légende + résumé chiffré."""
    img, s = _fit(rgb, max_side)
    h, w = img.shape[:2]
    fig_w = 13
    fig, ax = plt.subplots(figsize=(fig_w, fig_w * h / w * 0.78 + 1.2))
    ax.imshow(img)
    ax.set_axis_off()
    handles = []
    font = max(6, min(12, int(0.018 * max(h, w) / 10) + 6))
    for k, inst in enumerate(instances):
        col = PALETTE[k % len(PALETTE)]
        draw_polygons(ax, [r * s for r in _rings(inst.polygon)], color=col, alpha=0.40, lw=1.0)
        c = inst.polygon.representative_point()
        ax.text(c.x * s, c.y * s, str(inst.id), color="white", fontsize=font, weight="bold",
                ha="center", va="center",
                bbox=dict(boxstyle="circle,pad=0.15", fc=col, ec="white", lw=0.6, alpha=0.9))
        if k < max_legend:
            a = inst.extra.get("surface_m2")
            lab = f"#{inst.id} {inst.cls_name} — {inst.conf:.0%}" + (f" — {a:.3g} m²" if a is not None else "")
            handles.append(Patch(fc=col, ec=col, alpha=0.7, label=lab))
    if len(instances) > max_legend:
        handles.append(Patch(fc="none", ec="none", label=f"… +{len(instances) - max_legend} autres foyers"))
    if handles:
        ax.legend(handles=handles, loc="upper left", bbox_to_anchor=(1.01, 1.0), fontsize=8,
                  title="Foyers (du plus grand au plus petit)", title_fontsize=9, frameon=True)
    txt = [f"Foyers détectés : {summary['nb_foyers']}",
           f"Surface infectée : {summary['pct_image_infectee']:.2f} % de l'image",
           f"                  {summary['surface_infectee_m2']:.4g} m²"]
    if "pct_vegetation_infectee" in summary:
        txt.append(f"                  {summary['pct_vegetation_infectee']:.2f} % de la végétation")
    for d in (summary.get("diagnostic") or [])[:3]:
        txt.append(f"Diagnostic : {d['classe']} ({d['probabilite']:.0%})")
    ax.text(1.01, 0.0, "\n".join(txt), transform=ax.transAxes, fontsize=9, va="bottom", ha="left",
            family="monospace", bbox=dict(boxstyle="round", fc="#f7f7f7", ec="#999999"))
    ax.set_title(title, fontsize=12, weight="bold")
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=130, bbox_inches="tight")
    plt.close(fig)


def save_gt_vs_pred(rgb: np.ndarray, gt_polys: list[np.ndarray], pred_instances, out: Path,
                    metrics: dict | None = None) -> None:
    """Trois panneaux : image, vérité terrain (vert), prédiction (rouge)."""
    fig, axes = plt.subplots(1, 3, figsize=(14, 5))
    for ax, t in zip(axes, ["Image", f"Vérité terrain ({len(gt_polys)})", f"Prédiction ({len(pred_instances)})"]):
        ax.imshow(rgb)
        ax.set_title(t)
        ax.axis("off")
    draw_polygons(axes[1], gt_polys, color="#2ca02c")
    for inst in pred_instances:
        draw_polygons(axes[2], _rings(inst.polygon), color="#d62728")
    if metrics:
        fig.suptitle("  |  ".join(f"{k} = {v:.3f}" for k, v in metrics.items()), fontsize=10)
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=100)
    plt.close(fig)
