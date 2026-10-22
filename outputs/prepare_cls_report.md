# Préparation — jeu de classification (diagnostic)

- Sortie : `data/processed/cls`
- Classes : 97  |  Cultures : 31
- Images : 49303  (train=36709, val=6280, test=6314)
- Seuil min. d'images par classe : 100 ; plafond train : 800
- Erreurs d'extraction : 0

## Par source

| source_dataset | train | val | test |
|---|---|---|---|
| Multicrop/test | 1292 | 200 | 205 |
| Multicrop/train | 8914 | 1488 | 1495 |
| Multicrop/valid | 2528 | 440 | 440 |
| PlantDoc-Dataset/test | 114 | 18 | 20 |
| PlantDoc-Dataset/train | 1302 | 189 | 193 |
| PlantVillageDataset/PlantVillage | 9776 | 2061 | 2070 |
| Sugarcane Leaf Disease Dataset | 250 | 30 | 32 |
| blackgram/BPLD | 806 | 100 | 101 |
| paddy-disease-classification/train_images | 6136 | 1018 | 1024 |
| plantseg | 5591 | 736 | 734 |

## Par classe

| classe | train | val | test |
|---|---|---|---|
| Apple - rust | 180 | 22 | 23 |
| Apple - scab | 268 | 33 | 34 |
| Banana - black leaf streak | 122 | 14 | 15 |
| Banana - bract mosaic virus | 319 | 40 | 40 |
| Banana - bunchy top | 108 | 13 | 14 |
| Banana - cordana | 475 | 58 | 57 |
| Banana - healthy | 800 | 108 | 108 |
| Banana - insect pest | 550 | 69 | 69 |
| Banana - moko | 356 | 44 | 45 |
| Banana - panama | 800 | 117 | 116 |
| Banana - pestalotiopsis | 447 | 56 | 56 |
| Banana - sigatoka | 800 | 135 | 136 |
| Banana - yb sigatoka | 800 | 278 | 278 |
| Bean - rust | 91 | 11 | 12 |
| Bell Pepper - bacterial spot | 800 | 111 | 112 |
| Bell Pepper - healthy | 800 | 154 | 154 |
| Blackgram - anthracnose | 184 | 23 | 23 |
| Blackgram - healthy | 177 | 22 | 22 |
| Blackgram - leaf crinkle | 122 | 15 | 15 |
| Blackgram - powdery mildew | 144 | 18 | 18 |
| Blackgram - yellow mosaic | 179 | 22 | 23 |
| Blueberry - healthy | 92 | 11 | 12 |
| Cabbage - black rot | 90 | 11 | 11 |
| Cauliflower - bacterial soft rot | 153 | 19 | 19 |
| Cauliflower - downy mildew | 123 | 15 | 16 |
| Cauliflower - healthy | 165 | 20 | 21 |
| Cherry - leaf spot | 85 | 10 | 11 |
| Chilli - anthracnose | 156 | 19 | 20 |
| Chilli - healthy | 408 | 51 | 51 |
| Citrus - canker | 307 | 39 | 39 |
| Citrus - greening disease | 106 | 13 | 14 |
| Coffee - leaf rust | 122 | 15 | 16 |
| Corn - gray leaf spot | 127 | 15 | 16 |
| Corn - northern leaf blight | 244 | 30 | 31 |
| Corn - rust | 216 | 27 | 27 |
| Corn - smut | 162 | 20 | 20 |
| Cucumber - angular leaf spot | 144 | 18 | 19 |
| Cucumber - bacterial wilt | 84 | 11 | 10 |
| Cucumber - powdery mildew | 144 | 17 | 19 |
| Garlic - rust | 84 | 10 | 11 |
| Grape - black rot | 142 | 18 | 18 |
| Grape - downy mildew | 222 | 27 | 28 |
| Groundnut - early leaf spot | 800 | 121 | 121 |
| Groundnut - early rust | 800 | 146 | 147 |
| Groundnut - healthy | 325 | 40 | 41 |
| Groundnut - late leaf spot | 800 | 198 | 199 |
| Groundnut - nutrition deficiency | 800 | 166 | 166 |
| Groundnut - rust | 800 | 171 | 172 |
| Maple - tar spot | 91 | 11 | 11 |
| Peach - brown rot | 126 | 16 | 16 |
| Peach - healthy | 90 | 11 | 11 |
| Peach - leaf curl | 146 | 18 | 18 |
| Potato - early blight | 800 | 122 | 123 |
| Potato - healthy | 122 | 15 | 15 |
| Potato - late blight | 800 | 119 | 120 |
| Radish - black leaf spot | 413 | 52 | 52 |
| Radish - downy mildew | 465 | 58 | 58 |
| Radish - flea beetle | 370 | 46 | 47 |
| Radish - healthy | 478 | 60 | 60 |
| Radish - mosaic | 437 | 55 | 55 |
| Raspberry - healthy | 92 | 11 | 12 |
| Rice - bacterial leaf blight | 373 | 46 | 47 |
| Rice - bacterial leaf streak | 293 | 37 | 37 |
| Rice - bacterial panicle blight | 269 | 33 | 34 |
| Rice - blast | 800 | 179 | 179 |
| Rice - brown spot | 749 | 94 | 94 |
| Rice - dead heart | 800 | 141 | 142 |
| Rice - downy mildew | 488 | 61 | 61 |
| Rice - healthy | 800 | 174 | 175 |
| Rice - hispa | 800 | 155 | 156 |
| Rice - tungro | 800 | 107 | 107 |
| Soybean - downy mildew | 98 | 13 | 12 |
| Soybean - frog eye leaf spot | 155 | 20 | 19 |
| Soybean - mosaic | 88 | 12 | 10 |
| Soybean - rust | 86 | 10 | 11 |
| Squash - powdery mildew | 226 | 27 | 29 |
| Sugarcane - healthy | 94 | 11 | 12 |
| Sugarcane - red rot | 156 | 19 | 20 |
| Tomato - bacterial spot | 801 | 233 | 234 |
| Tomato - early blight | 801 | 122 | 124 |
| Tomato - healthy | 800 | 165 | 165 |
| Tomato - late blight | 800 | 215 | 217 |
| Tomato - leaf mold | 800 | 119 | 120 |
| Tomato - mosaic virus | 391 | 49 | 49 |
| Tomato - septoria leaf spot | 799 | 202 | 201 |
| Tomato - spider mites | 800 | 167 | 168 |
| Tomato - target spot | 800 | 140 | 141 |
| Tomato - yellow leaf curl virus | 800 | 336 | 337 |
| Wheat - bacterial leaf streak (black chaff) | 87 | 13 | 10 |
| Wheat - head scab | 213 | 26 | 27 |
| Wheat - leaf rust | 99 | 13 | 12 |
| Wheat - loose smut | 139 | 19 | 17 |
| Wheat - powdery mildew | 203 | 25 | 26 |
| Wheat - septoria blotch | 154 | 21 | 17 |
| Wheat - stem rust | 101 | 13 | 12 |
| Wheat - stripe rust | 256 | 31 | 32 |
| Zucchini - powdery mildew | 137 | 17 | 17 |
