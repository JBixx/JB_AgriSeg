# 🌿 JB_AgriSeg — Détection et détourage des maladies des cultures par vision par ordinateur

![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-2.14-EE4C2C?logo=pytorch&logoColor=white)
![Ultralytics](https://img.shields.io/badge/Ultralytics-YOLO11-111F68)
![Licence](https://img.shields.io/badge/licence-MIT-green)
![Tests](https://img.shields.io/badge/tests-6%2F6-brightgreen)

> **Statut :** pipeline complet validé de bout en bout (tests unitaires, entraînement court,
> évaluation, prédiction, export ONNX). **Entraînement final en cours** — les métriques seront
> publiées dans la section [Résultats](#-résultats).

---

## 🌍 Contexte

Les maladies foliaires (mildiou, rouilles, taches bactériennes, oïdium…) peuvent détruire une
part importante d'une récolte lorsqu'elles sont détectées trop tard. Or **inspecter des milliers
d'hectares à pied est impossible** : l'observation est lente, coûteuse, subjective, et les foyers
précoces passent inaperçus.

Les drones et l'imagerie à haute résolution rendent possible une surveillance à grande échelle,
à condition de savoir **analyser automatiquement** ces images.

## 🎯 Problématique et enjeux

Il ne suffit pas de dire *« cette plante est malade »*. Pour agir, un agronome doit savoir :

| Question terrain | Réponse apportée par le projet |
|---|---|
| **Où** sont les foyers ? | détourage précis de chaque zone malade (segmentation d'instance) |
| **Combien** de surface est touchée ? | surface par foyer en pixels et en **m²**, % de l'image et de la végétation |
| **Quelle** maladie est probable ? | diagnostic indicatif (top-3) par un second modèle |
| **Comment l'exploiter** ? | carte légendée, **GeoJSON** (SIG), **CSV**, application web |

**Enjeux** : une sous-détection laisse un foyer se propager ; une sur-détection entraîne des
traitements inutiles (coût, environnement). La métrique clé n'est donc pas seulement la
précision du modèle, mais la **justesse de la surface infectée estimée**.

## 💡 Approche

| Brique | Méthode | Type d'apprentissage |
|---|---|---|
| Détourage des lésions | **YOLO11-seg** pré-entraîné COCO, affiné sur 588 images annotées au masque | supervisé + transfer learning |
| Diagnostic de la maladie | **YOLO11-cls** pré-entraîné ImageNet, affiné sur 49 303 images, 97 classes, 31 cultures | supervisé + transfer learning |
| Part de végétation touchée | indice **Excess Green + seuil d'Otsu** | non supervisé |

L'apprentissage par renforcement a été écarté (pas d'agent, d'environnement ni de récompense
séquentielle dans ce problème). Chaque décision — données, modèles, augmentations, métriques,
limites — est justifiée dans **[METHODOLOGIE.md](METHODOLOGIE.md)**.

```
image / dossier / orthomosaïque (GeoTIFF)
        │  grande image → tuiles 640 px, recouvrement 25 %
        ▼
YOLO11-seg ── masques par foyer ── fusion des doublons entre tuiles
        │
        ├── surfaces px → m² (GSD ou géoréférencement)      ExG + Otsu → % végétation
        └── YOLO11-cls → diagnostic top-3
        ▼
carte PNG légendée · GeoJSON · CSV par foyer · résumé JSON · application Streamlit
```

---

## ✨ Fonctionnalités

- **Audit automatique des données** : formats (masques PNG, COCO, YOLO, archives), classes, tailles, fichiers corrompus ou orphelins
- **Conversion** vers YOLO-seg unifié, split 70/20/10 reproductible et stratifié par surface malade, sans fuite entre splits
- **Harmonisation** de 332 libellés bruts en 174 classes réelles (table d'alias paramétrable)
- **Entraînement** : early stopping, meilleurs poids, courbes d'apprentissage, export **ONNX**
- **Évaluation** sur test : mAP50 / mAP50-95 (masques et boîtes), précision, rappel, **IoU, Dice**, erreur sur le % de surface, matrices de confusion, exemples vérité terrain vs prédiction, rapport interprété
- **Inférence** sur image, dossier ou **orthomosaïque** découpée en tuiles puis recollée
- **GeoTIFF** : coordonnées WGS84 et surfaces en m² lues depuis le géoréférencement
- **Application Streamlit** : glisser-déposer, carte, statistiques, téléchargements
- GPU détecté automatiquement (CUDA / Apple MPS / CPU), compatible **Google Colab**

## 📦 Données

| Jeu | Contenu | Annotations | Usage |
|---|---|---|---|
| *Leaf disease segmentation* (Kaggle) | 588 photos de feuilles en conditions réelles | masques binaires lésion / sain | segmentation |
| Agrégat multi-sources (PlantVillage, PlantSeg, PlantDoc, Paddy, Multicrop, BPLD, Sugarcane) | 66 953 images, 42 cultures | étiquette de classe par image | diagnostic |

Les données ne sont pas versionnées (volume et licences des sources). Placez-les dans `Dataset/`
et ajustez les chemins dans `config.yaml` (section `raw:`).

**Vérification des annotations** — polygones YOLO-seg relus après conversion des masques
(contrôle visuel de la préparation, *pas* des prédictions du modèle) :

<img width="1100" height="572" alt="Annotation convertie — leaf_disease_seg__00014" src="https://github.com/user-attachments/assets/525a376c-9a19-4371-a031-698c9ab146a9" />

<img width="1100" height="572" alt="Annotation convertie — leaf_disease_seg__00295" src="https://github.com/user-attachments/assets/9d2ad2da-54dd-4d05-9473-6693a49afc1a" />

Les 10 aperçus sont dans [`outputs/verification/`](outputs/verification/) ; les statistiques des
jeux préparés dans [`outputs/prepare_seg_report.md`](outputs/prepare_seg_report.md) et
[`outputs/prepare_cls_report.md`](outputs/prepare_cls_report.md).

---

## 🗂️ Structure du projet

```
JB_AgriSeg/
├── config.yaml              tous les paramètres (chemins, splits, hyperparamètres, inférence)
├── inspect_dataset.py       1. audit des données
├── prepare_dataset.py       2. conversion, splits, data.yaml, aperçus de vérification
├── train.py                 3. entraînement (transfer learning, early stopping, export ONNX)
├── evaluate.py              4. évaluation sur le jeu de test + rapport interprété
├── predict.py               5. inférence : carte, GeoJSON, CSV, résumé
├── app.py                   6. interface Streamlit
├── src/                     conversion · tiling · postprocess · visualize · utils
├── tests/                   tests unitaires (pytest)
├── outputs/                 rapports, vérifications, courbes, évaluations
├── models/                  modèles finaux (best.pt/.onnx, cls_best.pt/.onnx)
├── METHODOLOGIE.md          décisions et justifications
├── requirements.txt         dépendances épinglées
└── LICENSE
```

## ⚙️ Installation

Python 3.10 ou plus.

```bash
git clone https://github.com/JBixx/JB_AgriSeg.git
cd JB_AgriSeg
python -m venv .venv
# Windows : .venv\Scripts\activate     Linux / macOS : source .venv/bin/activate

# PyTorch : variante CPU ci-dessous, ou CUDA via https://pytorch.org/get-started/locally/
pip install torch==2.14.1 torchvision==0.29.1 --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
python -m pytest -q tests/        # vérification de l'installation
```

> Derrière un proxy ou un antivirus qui inspecte le HTTPS (`CERTIFICATE_VERIFY_FAILED`),
> ajoutez `--use-feature=truststore` aux commandes `pip`.

## 🚀 Utilisation — dans l'ordre

```bash
# 1. Audit des données
python inspect_dataset.py

# 2. Préparation -> data/processed/ et outputs/verification/
python prepare_dataset.py --task seg        # ou cls, ou all

# 3. Entraînement
python train.py --task seg
python train.py --task cls
python train.py --task seg --epochs 3 --fraction 0.2      # test rapide de la chaîne

# 4. Évaluation sur le jeu de test -> outputs/evaluation/<task>/rapport_evaluation.md
python evaluate.py --task seg
python evaluate.py --task cls

# 5. Prédiction -> outputs/predictions/<image>/
python predict.py --source chemin/image.jpg
python predict.py --source chemin/dossier/ --gsd 1.5
python predict.py --source orthomosaique.tif --tile-size 640 --overlap 0.25

# 6. Interface web
streamlit run app.py
```

### Sorties de `predict.py`

| Fichier | Contenu |
|---|---|
| `<nom>_carte.png` | zones infectées colorées et numérotées, légende, résumé chiffré |
| `<nom>_foyers.geojson` | un polygone par foyer (WGS84 si GeoTIFF, sinon coordonnées pixels) |
| `<nom>_foyers.csv` | id, classe, confiance, surface px et m², centroïde (x, y, lon, lat) |
| `<nom>_resume.json` | nombre de foyers, % de l'image et de la végétation infectés, m², diagnostic top-3 |

### Entraîner sur Google Colab (GPU recommandé)

```python
# Exécution > Modifier le type d'exécution > GPU (T4)
from google.colab import drive; drive.mount('/content/drive')
!git clone https://github.com/JBixx/JB_AgriSeg.git
%cd JB_AgriSeg
!ln -s /content/drive/MyDrive/Dataset Dataset        # vos données déposées sur Drive
!pip install -q -r requirements.txt
!python prepare_dataset.py --task all
!python train.py --task seg --model yolo11s-seg.pt --batch 16
!python train.py --task cls --model yolo11s-cls.pt
!python evaluate.py --task seg && python evaluate.py --task cls
```

Le GPU est détecté automatiquement et `data.yaml` est réécrit à chaque entraînement : le projet
peut être déplacé librement.

---

## 📊 Résultats

*Entraînement final en cours.* Les métriques sur le jeu de test (mAP, IoU, Dice, erreur sur la
surface infectée, exactitude du diagnostic) et des exemples de cartes seront publiés ici et dans
[METHODOLOGIE.md § 12](METHODOLOGIE.md#12-résultats-obtenus).

## ⚠️ Limites

- Les modèles sont entraînés sur des **photos rapprochées de feuilles**, pas sur des images de
  drone. La chaîne aérienne (tuilage, GSD, GeoTIFF) est opérationnelle, mais un **affinage sur
  des images aériennes annotées** est nécessaire avant un usage sur orthomosaïques réelles.
- La granularité des annotations varie (feuille entière vs taches individuelles), ce qui limite
  la mAP par instance.
- Le diagnostic est **indicatif** et ne remplace pas l'avis d'un agronome.

Détails et pistes d'amélioration : [METHODOLOGIE.md § 9 et § 10](METHODOLOGIE.md#9-limites-et-risques--à-lire-avant-tout-usage-terrain).

## 🛣️ Feuille de route

- [x] Audit, conversion et harmonisation des données
- [x] Pipeline d'entraînement, d'évaluation et d'inférence validé
- [x] Tuilage d'orthomosaïques, GeoTIFF, GeoJSON, application Streamlit
- [ ] Entraînement final sur GPU et publication des métriques
- [ ] Affinage sur images de drone annotées
- [ ] Récupération des masques PlantSeg (×18 données de segmentation)

## 📄 Licence et crédits

Code sous licence **MIT** — © 2026 HIRWA Jean Baptiste (voir [LICENSE](LICENSE)).

[Ultralytics YOLO](https://github.com/ultralytics/ultralytics) est sous **AGPL-3.0** : un usage
commercial ou un service en ligne basé sur ces modèles impose de respecter l'AGPL ou d'acquérir
une licence Ultralytics Enterprise. Les jeux de données restent soumis à la licence de leurs
auteurs (Kaggle, PlantVillage, PlantSeg, PlantDoc…).

## 👤 Auteur

**HIRWA Jean Baptiste** — [GitHub @JBixx](https://github.com/JBixx)
