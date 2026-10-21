"""
app.py — Interface Streamlit : déposez une image, obtenez la carte des foyers,
les statistiques et les fichiers à télécharger.

Lancement :
    streamlit run app.py
"""
from __future__ import annotations

import tempfile
from pathlib import Path

import pandas as pd
import streamlit as st

from predict import Analyzer, load_image
from src.utils import load_config, resolve

st.set_page_config(page_title="Détection des maladies des cultures", page_icon="🌿", layout="wide")


@st.cache_resource(show_spinner="Chargement des modèles…")
def get_analyzer(seg_w: str, cls_w: str) -> Analyzer:
    return Analyzer(load_config("config.yaml"), seg_w, cls_w)


cfg = load_config("config.yaml")
st.title("🌿 Détection et détourage des zones malades")
st.caption("Segmentation d'instance YOLO — chaque foyer est détouré, numéroté et chiffré.")

with st.sidebar:
    st.header("Paramètres")
    conf = st.slider("Seuil de confiance", 0.05, 0.95, float(cfg["predict"]["conf"]), 0.05,
                     help="Plus bas = plus de foyers détectés (et plus de fausses alertes).")
    gsd = st.number_input("Résolution au sol (cm/pixel)", 0.001, 500.0, float(cfg["predict"]["gsd_cm_per_px"]),
                          help="Sert à convertir les pixels en m². Ignorée pour un GeoTIFF projeté.")
    tile = st.select_slider("Taille des tuiles (grandes images)", [320, 480, 640, 800, 1024],
                            value=int(cfg["predict"]["tile_size"]))
    overlap = st.slider("Recouvrement des tuiles", 0.0, 0.5, float(cfg["predict"]["tile_overlap"]), 0.05)
    seg_w = st.text_input("Poids segmentation", cfg["predict"]["seg_weights"])
    cls_w = st.text_input("Poids diagnostic (optionnel)", cfg["predict"]["cls_weights"])

up = st.file_uploader("Déposez une image (JPG, PNG, TIFF/GeoTIFF)",
                      type=["jpg", "jpeg", "png", "bmp", "webp", "tif", "tiff"])
if up is None:
    st.info("Déposez une image pour lancer l'analyse.")
    st.stop()

if not resolve(seg_w).exists():
    st.error(f"Modèle introuvable : `{seg_w}`. Lancez d'abord `python train.py --task seg`.")
    st.stop()

an = get_analyzer(seg_w, cls_w)
an.p["conf"] = conf
with tempfile.TemporaryDirectory() as tmp:
    src = Path(tmp) / up.name
    src.write_bytes(up.getvalue())
    with st.spinner("Analyse en cours…"):
        rgb, geo = load_image(src)
        res = an.analyze(rgb, geo, gsd_cm=gsd, tile_size=tile, overlap=overlap)
        paths = an.export(rgb, res, Path(tmp) / "out", Path(up.name).stem, up.name)
        files = {k: p.read_bytes() for k, p in paths.items()}

s = res["summary"]
c1, c2, c3, c4 = st.columns(4)
c1.metric("Foyers détectés", s["nb_foyers"])
c2.metric("Surface infectée (image)", f"{s['pct_image_infectee']:.2f} %")
c3.metric("Surface infectée", f"{s['surface_infectee_m2']:.4g} m²")
if "pct_vegetation_infectee" in s:
    c4.metric("Part de la végétation touchée", f"{s['pct_vegetation_infectee']:.2f} %")

if s.get("diagnostic"):
    st.subheader("Diagnostic probable")
    st.dataframe(pd.DataFrame(s["diagnostic"]).assign(probabilite=lambda d: (d.probabilite * 100).round(1))
                 .rename(columns={"classe": "Maladie", "probabilite": "Probabilité (%)"}),
                 hide_index=True, use_container_width=True)
    st.caption("Diagnostic indicatif sur l'image entière ; à confirmer par un agronome.")

st.subheader("Carte des foyers")
st.image(files["carte"], use_container_width=True)

st.subheader("Détail par foyer")
st.dataframe(pd.DataFrame(res["rows"]), hide_index=True, use_container_width=True)

st.subheader("Téléchargements")
stem = Path(up.name).stem
d1, d2, d3, d4 = st.columns(4)
d1.download_button("🗺️ Carte (PNG)", files["carte"], f"{stem}_carte.png", "image/png")
d2.download_button("📍 GeoJSON", files["geojson"], f"{stem}_foyers.geojson", "application/geo+json")
d3.download_button("📊 CSV des foyers", files["csv"], f"{stem}_foyers.csv", "text/csv")
d4.download_button("🧾 Résumé (JSON)", files["resume"], f"{stem}_resume.json", "application/json")
if not s.get("georeference"):
    st.caption("Image non géoréférencée : le GeoJSON est en coordonnées pixels. "
               "Déposez un GeoTIFF pour obtenir des coordonnées WGS84.")
