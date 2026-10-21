# Préparation — jeu de segmentation (YOLO-seg)

- Sortie : `data/processed/seg`
- Classe(s) : ['lesion']
- Split (train/val/test) : [0.7, 0.2, 0.1] — graine 42
- Variantes aug_data ajoutées au train : False
- Rotations 90° hors-ligne : False

| Split | Images | Instances | Images sans instance |
|---|---|---|---|
| train | 411 | 3304 | 1 |
| val | 118 | 937 | 1 |
| test | 59 | 622 | 0 |
