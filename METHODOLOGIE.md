# Méthodologie — Détection et détourage des maladies des cultures

> Document de décision : chaque choix de données, de modèle, d'entraînement et d'évaluation
> est listé ici avec sa justification. Les résultats chiffrés sont produits automatiquement
> dans `outputs/evaluation/` et résumés en fin de document.

---

## 1. Problématique et enjeux

**Objectif métier.** Inspecter des milliers d'hectares à pied est impossible. Le système doit
analyser des images de cultures, **détourer chaque zone malade** (foyer), les **numéroter**,
**chiffrer la surface touchée** (pixels → m² via la résolution au sol) et produire une carte et
des fichiers exploitables (GeoJSON pour un SIG, CSV pour un tableur).

**Ce que cela implique techniquement.**

| Question métier | Tâche de vision | Type d'apprentissage |
|---|---|---|
| *Où* sont les zones malades, et quelle surface ? | Segmentation d'instance (un masque par foyer) | **Supervisé** (masques annotés) |
| *Quelle* maladie est probablement en cause ? | Classification d'image | **Supervisé** (étiquette par image) |
| Quelle part de la *végétation* est touchée ? | Séparation végétation / sol | **Non supervisé** (indice ExG + seuil d'Otsu) |

**Enjeux de qualité.** Une sous-détection laisse un foyer se propager ; une sur-détection fait
traiter inutilement. La métrique métier la plus directe est donc l'**erreur sur le % de surface
infectée**, en plus des métriques standard (mAP, IoU, Dice).

---

## 2. Audit des données disponibles

Inspection réelle du dossier `Dataset/` (`inspect_dataset.py`, rapport dans
`outputs/rapport_inspection.md`).

| Élément | Contenu | Annotations | Utilisable pour |
|---|---|---|---|
| `archive (1)/data/data` | 588 photos de feuilles en conditions réelles, tailles variées (256² à 4000 px) | **588 masques PNG binaires** (lésion / reste), 0 orphelin | **Segmentation** ✅ |
| `archive (1)/aug_data/aug_data` | 2 940 images = 5 variantes augmentées de chacune des 588 | masques correspondants | ⚠️ redondant / risque de fuite |
| `archive.zip` (4,6 Go) | 66 953 images issues de 10 sources (PlantVillage, PlantSeg, Multicrop, Paddy, PlantDoc, BPLD, Sugarcane) | **étiquette de classe uniquement** (`mask_path` vide partout), splits train/val/test fournis | **Classification** ✅ |
| `master_images/` | extraction partielle du zip (≈ 26 400 / 66 953 images) | — | lu directement dans le zip |
| `*.ipynb` | notebooks Kaggle de référence (ResNet50 classification, U-Net végétation) | — | inspiration (indices de végétation) |

**Constats chiffrés de l'audit :**

- **Masques** : encodage hétérogène — mode palette (indice 1 = lésion) ou RGB avec
  anti-aliasing (rouge 128 + valeurs parasites 30, 53, 63…). Un seuil naïf `> 127` donne des
  masques **vides** : il faut un décodage dédié (`src/conversion.read_binary_mask`).
- Surface malade : **16 % de l'image en moyenne** (médiane 12 %, de 0,1 % à 91 %).
- Foyers par image (composantes connexes) : **médiane 5, p90 ≈ 25, max 80** → vraie tâche
  multi-instances.
- **Granularité d'annotation hétérogène** : certaines images détourent chaque tache, d'autres la
  feuille entière (constat visuel). Le modèle héritera de cette ambiguïté.
- Quelques images de banques d'images portent un **filigrane** (dreamstime/alamy).
- **Classification** : 332 libellés bruts, mais beaucoup de doublons de nommage (espaces finaux,
  suffixes « Google/Bing/Baidu » de collecte web, « - Copy », « Paddy » vs « Rice »,
  « downey » vs « downy »…). Après harmonisation : **185 classes**, dont 96 ont ≥ 100 images
  (62 293 images couvertes). **Longue traîne** : 51 classes ont < 10 images.
- **Déséquilibre** : de 1 à 3 360 images par classe ; 8 936 « sain » vs 58 017 « malade ».
- **Biais de domaine** : 31 % des images viennent de PlantVillage (feuille isolée sur fond
  uni, laboratoire) — un modèle peut y apprendre le fond plutôt que la maladie.
- **Fuite** : aucun fichier d'origine n'apparaît dans plus d'un split fourni (vérifié par
  `source_dataset + original_filename`).
- **Ce ne sont pas des images aériennes** : toutes les images sont des vues rapprochées de
  feuilles. Voir § 9 (limites).

---

## 3. Choix du type d'apprentissage

**Retenu : apprentissage supervisé par transfert (transfer learning)**, complété d'une brique
non supervisée.

- **Supervisé** parce que nous disposons d'annotations de qualité (588 masques, 67 k étiquettes) :
  c'est l'approche qui donne de loin les meilleures performances quand des étiquettes existent.
- **Transfert** parce que 588 images est trop peu pour apprendre de zéro : on part de réseaux
  pré-entraînés (COCO pour la segmentation, ImageNet pour la classification) qui savent déjà
  extraire bords, textures et couleurs ; on n'apprend que la spécialisation « lésion ».
- **Non supervisé (auxiliaire)** : l'indice *Excess Green* (2G − R − B) seuillé par Otsu
  sépare végétation et sol sans aucune annotation. Il sert à exprimer la surface malade en
  % de la végétation (plus parlant qu'un % d'image qui inclut du sol). C'est l'idée du notebook
  *cassava-leaf-segmentation-using-vegetation-indices* fourni, réutilisée là où elle est fiable.
- **Apprentissage par renforcement : écarté.** Le RL apprend une *politique d'actions* par
  essais/récompenses dans un environnement (ex. un drone qui choisit où voler). Ici il n'y a ni
  actions, ni environnement, ni récompense séquentielle : il s'agit de reconnaître des motifs
  dans des images annotées. Le RL serait pertinent plus tard pour **planifier les survols**
  (prioriser les parcelles à risque), pas pour détecter les maladies.
- **Pseudo-étiquetage / semi-supervisé : reporté.** Les 67 k images du zip pourraient recevoir
  des masques prédits par le modèle de segmentation (pseudo-labels) pour un second
  entraînement. C'est une piste d'amélioration (§ 10), pas la base, car des pseudo-labels
  bruités sur un domaine différent (PlantVillage) peuvent dégrader le modèle.

---

## 4. Architecture de la solution

```
                ┌──────────────────────────┐
 image / dossier│  chargement (PIL/rasterio)│── GeoTIFF : transform + CRS
 / orthomosaïque└────────────┬─────────────┘
                             │ grande image ? → tuiles 640 px, recouvrement 25 %
                ┌────────────▼─────────────┐
                │ Modèle 1 : YOLO11-seg     │  supervisé — 1 classe « lésion »
                │ masques d'instance        │
                └────────────┬─────────────┘
                             │ fusion des doublons entre tuiles (union-find + shapely)
         ┌───────────────────┼──────────────────────┐
         ▼                   ▼                      ▼
 Modèle 2 : YOLO11-cls   ExG + Otsu            surfaces px → m² (GSD ou GeoTIFF)
 diagnostic top-3        % végétation          centroïdes, numérotation
 (supervisé)             (non supervisé)
         └───────────────────┼──────────────────────┘
                             ▼
     carte PNG légendée · GeoJSON · CSV par foyer · résumé JSON · app Streamlit
```

**Pourquoi deux modèles plutôt qu'un YOLO-seg multi-classes ?** Les masques disponibles ne
portent **aucune** information de maladie (une seule classe « lésion ») et les 67 k images
étiquetées par maladie n'ont **aucun** masque. Aucune donnée ne permet d'apprendre « contour +
maladie » en même temps. On découple donc : le modèle 1 dit *où et combien*, le modèle 2 dit
*quoi*. Les deux sont indépendants et remplaçables.

---

## 5. Préparation des données (`prepare_dataset.py`)

### 5.1 Segmentation → format YOLO-seg

| Décision | Valeur | Justification |
|---|---|---|
| Classe | `lesion` (unique) | seule information présente dans les masques |
| Définition d'une instance | composante connexe du masque | les masques sont sémantiques ; un foyer = une zone d'un seul tenant |
| Aire minimale | 30 px | élimine le bruit d'anti-aliasing et les pixels isolés |
| Contours | externes, simplifiés Douglas-Peucker ε = 1 px | YOLO-seg = un polygone externe par instance ; ε = 1 px garde la précision en réduisant la taille des labels |
| Redimensionnement | côté max 1024 px (masque en plus-proche-voisin) | accélère le chargement sans perte utile pour `imgsz` 640 ; le masque n'est jamais interpolé |
| EXIF | pixels bruts réécrits sans EXIF | évite tout décalage image/masque dû à une rotation EXIF |
| Split | **70 / 20 / 10**, graine 42 | demandé ; reproductible |
| Stratification | par quartile de surface malade | val et test contiennent petits *et* grands foyers → métriques représentatives |
| `aug_data` | **exclu** (option `use_augmented_in_train`) | variantes des mêmes 588 images : (1) fuite si une variante d'une image de test va en train, (2) redondant avec l'augmentation en ligne, plus riche. Si activé, seules les variantes des images *train* sont ajoutées. |
| Vérification | 10 images avec polygones superposés → `outputs/verification/` | contrôle visuel de la conversion |

### 5.2 Classification → dossiers par classe

| Décision | Valeur | Justification |
|---|---|---|
| Harmonisation des noms | automatique + table d'alias dans `config.yaml` | 332 → 185 classes réelles ; paramétrable |
| Seuil minimal | ≥ 100 images / classe | en dessous, ni l'apprentissage ni l'évaluation ne sont fiables |
| Plafond train | 800 images / classe, échantillonnage proportionnel aux sources | réduit le déséquilibre (1 → 3 360) et le temps de calcul, tout en gardant la diversité labo/terrain |
| Splits | ceux fournis dans le zip (80/10/10), sans fuite vérifiée | respecte la déduplication faite en amont par l'auteur |
| Stockage | extraction directe depuis le zip, côté court 256 px | pas besoin de décompresser 4,6 Go ; 224 px suffisent au modèle |
| Évaluation | val/test **non plafonnés** | mesure sur la distribution réelle |

---

## 6. Modèles et entraînement (`train.py`)

### 6.1 Modèle 1 — segmentation des lésions

| Paramètre | Valeur | Justification |
|---|---|---|
| Architecture | **YOLO11n-seg** (CPU) / **YOLO11s-seg** (GPU, recommandé) | état de l'art temps réel en segmentation d'instance, export ONNX natif, une seule bibliothèque (`ultralytics`) |
| Pré-entraînement | COCO | transfert de caractéristiques génériques |
| `imgsz` | 640 | compromis détail des petites taches / mémoire |
| Époques / patience | 100 / 25 | early stopping : arrêt si la validation ne progresse plus pendant 25 époques |
| Optimiseur | AdamW, lr 1e-3, cosinus → 1e-5, warmup 3 époques | convergence stable sur petit jeu |
| Sélection du meilleur modèle | fitness Ultralytics (mAP50-95 boîtes + masques) | les poids `best.pt` correspondent à la meilleure époque, pas à la dernière |
| Reproductibilité | `seed=42`, `deterministic=True` | |

**Augmentations** (en ligne, différentes à chaque époque) :

| Augmentation | Valeur | Pourquoi |
|---|---|---|
| Rotation | ±90° | une lésion n'a pas d'orientation ; indispensable en vue de dessus |
| Flips H et V | 0,5 / 0,5 | idem ; combinés aux rotations couvrent les 8 orientations |
| HSV (teinte/saturation/luminosité) | 0,015 / 0,6 / 0,4 | variations d'éclairage, de capteur, d'heure de prise de vue |
| Mosaic | 1,0 (coupée les 10 dernières époques) | combine 4 images : plus de contexte et de petites instances ; désactivé à la fin pour finir sur des images réalistes |
| Translate / scale | 0,1 / 0,5 | robustesse à la distance de prise de vue (≈ altitude de vol) |
| Copy-paste | 0,1 | colle des lésions d'une image à l'autre : plus d'instances variées |
| Rotations 90° hors-ligne | option `offline_rot90` | pour le futur jeu aérien |

### 6.2 Modèle 2 — diagnostic

| Paramètre | Valeur | Justification |
|---|---|---|
| Architecture | YOLO11n-cls (s-cls sur GPU) | même écosystème, rapide, export ONNX |
| Pré-entraînement | ImageNet | |
| `imgsz` | 224 | standard classification |
| Époques / patience | 30 / 8 | jeu plus gros → converge plus vite |
| Régularisation | dropout 0,2, RandAugment, random erasing 0,4 | limite le sur-apprentissage sur les fonds PlantVillage |

### 6.3 Matériel

Device détecté automatiquement (CUDA → MPS → CPU). La machine de développement est un
i7-6600U (2 cœurs, sans GPU) : l'entraînement y est **fonctionnel mais lent**. Le README
fournit la procédure **Google Colab (GPU T4)** recommandée pour l'entraînement final, notamment
pour `yolo11s-seg` et le modèle de diagnostic.

---

## 7. Protocole d'évaluation (`evaluate.py`)

Toutes les métriques sont calculées sur le **jeu de test**, jamais utilisé pour
l'entraînement ni pour choisir l'époque.

| Métrique | Ce qu'elle mesure | Pourquoi on la suit |
|---|---|---|
| mAP50, mAP50-95 (masques et boîtes) | qualité de détection *par foyer* à différents niveaux d'exigence de recouvrement | standard COCO, comparable à la littérature |
| Précision / Rappel / F1 | fausses alertes vs oublis | arbitrage du seuil de confiance |
| IoU et Dice au pixel (global et moyen par image) | qualité du **détourage** | indépendant du découpage en instances |
| Erreur absolue moyenne sur le % de surface infectée | **justesse du chiffrage des dégâts** | indicateur métier principal |
| Matrices de confusion (instances et pixels) | où le modèle se trompe | diagnostic |
| Exemples (3 pires, 3 médians, 3 meilleurs) | vérité terrain vs prédiction | transparence : on montre aussi les échecs |

Pour le diagnostic : exactitude top-1 / top-5, **F1 macro** (chaque maladie compte autant, ce
qui révèle les classes rares mal apprises), exactitude **par source** (labo vs terrain) et par
culture, confusions les plus fréquentes.

---

## 8. Inférence et livrables (`predict.py`, `app.py`)

- **Entrées** : une image, un dossier, ou une **orthomosaïque** (y compris GeoTIFF 16 bits).
- **Tuilage** : au-delà de 1 600 px de côté, découpe en tuiles de 640 px avec **25 % de
  recouvrement** ; une lésion coupée par un bord de tuile est vue entière dans la tuile voisine.
  Les doublons sont fusionnés (polygones de même classe se recouvrant à > 5 % → union).
- **Surfaces** : `m² = pixels × (GSD_cm / 100)²` ; pour un GeoTIFF projeté, la taille du pixel
  est lue dans la transformation affine (le GSD saisi est alors ignoré).
- **Sorties** : carte PNG (zones colorées, numérotées du plus grand au plus petit foyer, légende,
  résumé chiffré), GeoJSON (WGS84 si géoréférencé, conforme RFC 7946 ; sinon pixels), CSV par
  foyer (id, classe, confiance, surface px et m², centroïde, lon/lat si géoréférencé), résumé
  JSON (nombre de foyers, % de l'image et de la végétation infectés, diagnostic top-3).
- **Application** : `streamlit run app.py` — dépôt d'image, réglage du seuil, du GSD et du tuilage,
  carte, statistiques, boutons de téléchargement.
- **Export** : `models/best.pt` + `models/best.onnx` (et `cls_best.*` pour le diagnostic).

---

## 9. Limites et risques — à lire avant tout usage terrain

1. **Écart de domaine (le plus important).** Les modèles sont entraînés sur des **photos
   rapprochées de feuilles**, pas sur des images de drone. Toute la chaîne aérienne est en place
   (tuilage, GSD, GeoTIFF, GeoJSON), mais la **performance sur orthomosaïque n'est pas garantie**
   tant que le modèle n'a pas été affiné sur quelques centaines de tuiles aériennes annotées.
2. **Granularité d'annotation incohérente** (feuille entière vs taches individuelles) : plafonne
   la mAP par instance ; le Dice et l'erreur de surface sont plus représentatifs.
3. **Petit jeu de segmentation** (588 images, ~59 en test) : les métriques ont une marge
   d'incertitude de plusieurs points.
4. **Biais PlantVillage** pour le diagnostic : l'exactitude par source est publiée pour le rendre
   visible.
5. **Le diagnostic est indicatif** : il classe l'image entière, ne remplace pas un agronome, et ses
   probabilités ne sont pas calibrées.

---

## 10. Pistes d'amélioration (par ordre de rendement attendu)

1. Annoter des tuiles aériennes réelles (pré-annotation par ce modèle + correction CVAT/Roboflow),
   puis fine-tuning.
2. Entraîner `yolo11s-seg`/`yolo11m-seg` à `imgsz` 1024 sur GPU.
3. Récupérer les **masques originaux de PlantSeg** (≈ 10 k images du zip en proviennent, mais leurs
   masques n'ont pas été inclus dans l'archive) : ×18 de données de segmentation, en conditions réelles.
4. Pseudo-étiquetage semi-supervisé des images du zip, filtré par confiance.
5. Validation croisée à 5 plis pour des intervalles de confiance sur les métriques.
6. Classification hiérarchique (culture → maladie) et calibration des probabilités.

---

## 11. Reproductibilité

```bash
python inspect_dataset.py                 # audit
python prepare_dataset.py --task all      # conversions + splits + aperçus
python train.py --task seg                # modèle 1
python train.py --task cls                # modèle 2 (GPU conseillé)
python evaluate.py --task seg
python evaluate.py --task cls
python predict.py --source <image|dossier|ortho.tif> --gsd 1.5
streamlit run app.py
```

Toutes les décisions chiffrées ci-dessus sont des paramètres de `config.yaml`.

---

## 12. Résultats obtenus

*Section complétée après exécution — voir `outputs/evaluation/seg/rapport_evaluation.md` et
`outputs/evaluation/cls/rapport_evaluation.md`.*
