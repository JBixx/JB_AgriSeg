# 🌿 Crop Disease Segmentation — détection et détourage des maladies des cultures

Segmentation d'instance **YOLO11-seg** (apprentissage supervisé + transfer learning) qui détoure
chaque zone malade d'une image de culture, numérote les foyers, **chiffre la surface touchée**
(px et m²) et exporte une carte, un **GeoJSON** et un **CSV**. Un second modèle **YOLO11-cls**
propose un diagnostic (top-3 des maladies probables) et un indice de végétation **non supervisé**
(ExG + Otsu) exprime la surface malade en % de la végétation.

👉 Toutes les décisions (données, modèles, augmentations, métriques, limites) sont justifiées dans
**[METHODOLOGIE.md](METHODOLOGIE.md)**.

## Fonctionnalités

- Conversion automatique masques PNG / COCO / YOLO → YOLO-seg unifié, split 70/20/10 reproductible et stratifié
- Entraînement avec early stopping, meilleurs poids, courbes d'apprentissage, export **ONNX**
- Évaluation sur test : mAP50 et mAP50-95 (masques et boîtes), précision, rappel, **IoU, Dice**, matrices de confusion, erreur sur le % de surface, exemples vérité terrain / prédiction, rapport Markdown interprété
- Inférence sur image, dossier ou **orthomosaïque découpée en tuiles avec recouvrement** puis recollée
- **GeoTIFF** : coordonnées WGS84 et surfaces en m² lues depuis le géoréférencement (rasterio)
- Application **Streamlit** : glisser-déposer, carte, statistiques, téléchargements
- GPU détecté automatiquement (CUDA / Apple MPS / CPU) ; compatible **Google Colab**

## Structure

```
├── Dataset/                 données brutes (lecture seule, non versionnées)
├── data/processed/          données converties (seg/ YOLO-seg, cls/ dossiers par classe)
├── models/                  best.pt, best.onnx, cls_best.pt, cls_best.onnx
├── outputs/                 rapports, vérifications, courbes, évaluations, prédictions
├── src/                     conversion, tiling, postprocess, visualize, utils
├── config.yaml              TOUS les paramètres
├── inspect_dataset.py       1. audit des données
├── prepare_dataset.py       2. conversion + splits + data.yaml + aperçus
├── train.py                 3. entraînement
├── evaluate.py              4. évaluation sur test + rapport
├── predict.py               5. inférence (carte, GeoJSON, CSV, résumé)
├── app.py                   6. interface Streamlit
├── METHODOLOGIE.md          décisions et justification
└── requirements.txt
```

## Installation

Python 3.10+.

```bash
python -m venv .venv
# Windows : .venv\Scripts\activate    |  Linux/macOS : source .venv/bin/activate
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu   # ou la version CUDA
pip install -r requirements.txt
```

> Derrière un proxy/antivirus qui inspecte le HTTPS (erreur `CERTIFICATE_VERIFY_FAILED`) :
> `pip install --use-feature=truststore ...` utilise le magasin de certificats du système.

## Données

| Dossier | Source | Rôle |
|---|---|---|
| `Dataset/archive (1)/data/data/{images,masks}` | jeu Kaggle *leaf disease segmentation* (588 images + masques) | segmentation |
| `Dataset/archive.zip` | agrégat multi-sources de 66 953 images (PlantVillage, PlantSeg, PlantDoc, Paddy, Multicrop…) avec manifest | diagnostic |

Les chemins sont modifiables dans `config.yaml` (`raw:`).

## Utilisation — dans l'ordre

```bash
# 1. Audit des données (formats, classes, tailles, fichiers corrompus/orphelins)
python inspect_dataset.py

# 2. Préparation (seg, cls ou all) -> data/processed/ + outputs/verification/
python prepare_dataset.py --task seg
python prepare_dataset.py --task cls

# 3. Entraînement (paramètres dans config.yaml)
python train.py --task seg
python train.py --task cls
python train.py --task seg --epochs 3 --fraction 0.2     # test rapide de la chaîne

# 4. Évaluation sur le jeu de test -> outputs/evaluation/<task>/rapport_evaluation.md
python evaluate.py --task seg
python evaluate.py --task cls

# 5. Prédiction -> outputs/predictions/<image>/
python predict.py --source chemin/image.jpg
python predict.py --source chemin/dossier/ --gsd 1.5
python predict.py --source orthomosaique.tif --tile-size 640 --overlap 0.25

# 6. Interface
streamlit run app.py
```

### Sorties de `predict.py`

| Fichier | Contenu |
|---|---|
| `<nom>_carte.png` | zones infectées colorées et numérotées, légende, résumé |
| `<nom>_foyers.geojson` | un polygone par foyer (WGS84 si GeoTIFF, sinon pixels) |
| `<nom>_foyers.csv` | id, classe, confiance, surface_px, surface_m2, centroïde (x, y, lon, lat) |
| `<nom>_resume.json` | nb de foyers, % image / % végétation infectés, m², diagnostic top-3 |

## Entraîner sur Google Colab (GPU recommandé)

```python
# Exécution > Modifier le type d'exécution > GPU (T4)
from google.colab import drive; drive.mount('/content/drive')
%cd /content/drive/MyDrive/crop-disease-segmentation      # copie du projet + Dataset/
!pip install -q -r requirements.txt
!python prepare_dataset.py --task all
!python train.py --task seg --model yolo11s-seg.pt --batch 16
!python train.py --task cls --model yolo11s-cls.pt
!python evaluate.py --task seg && python evaluate.py --task cls
```

Le device GPU est détecté automatiquement ; `data.yaml` est réécrit avec le bon chemin à chaque
entraînement, le projet peut donc être déplacé librement.

## Résultats

Voir [METHODOLOGIE.md § 12](METHODOLOGIE.md#12-résultats-obtenus) et les rapports détaillés
dans `outputs/evaluation/`.

<img width="1100" height="572" alt="leaf_disease_seg__00014" src="https://github.com/user-attachments/assets/525a376c-9a19-4371-a031-698c9ab146a9" />

<img width="1100" height="572" alt="leaf_disease_seg__00295" src="https://github.com/user-attachments/assets/9d2ad2da-54dd-4d05-9473-6693a49afc1a" />


## Limites

Les modèles sont entraînés sur des **photos rapprochées de feuilles**. La chaîne aérienne
(tuilage, GSD, GeoTIFF) est opérationnelle, mais un **affinage sur des images de drone
annotées** est nécessaire avant un usage sur orthomosaïques réelles. Détails : METHODOLOGIE.md § 9.

## Licence et crédits

Code du projet sous licence **MIT** — © 2026 HIRWA Jean Baptiste (voir [LICENSE](LICENSE)).

Dépendances et données soumises à leurs propres licences :
[Ultralytics YOLO](https://github.com/ultralytics/ultralytics) est sous **AGPL-3.0** (un usage
commercial ou un service en ligne basé sur ces modèles impose de respecter l'AGPL ou d'acquérir
une licence Ultralytics Enterprise) ; les jeux de données restent soumis à la licence de leurs
auteurs (Kaggle, PlantVillage, PlantSeg, PlantDoc…).

**Auteur :** HIRWA Jean Baptiste
