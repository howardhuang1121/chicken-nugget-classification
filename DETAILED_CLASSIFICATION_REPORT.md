# Chicken Nugget Classification - Detailed Results

Run date: 2026-10-16  
Task: four-class image classification for Ball, Bell, Bone, and Boot nuggets

## Executive summary

The final leakage-safe classifier correctly classified all 23 supplied test images.

- Test accuracy: **23/23 = 100.00%**
- Test macro F1: **1.0000**
- Test confusion matrix: perfectly diagonal, with 6 Ball, 6 Bell, 5 Bone, and 6 Boot images correct
- Mean reported class probability: **0.9262**
- Lowest reported class probability: **0.6911** on `Boot_05.png`
- Leakage-safe source-level cross-validation: **62/71 = 87.32% accuracy**, macro F1 **0.8442**

The 100% test score needs context. A source-overlap audit showed that every test photo depicts the same physical nugget as one original in the supplied training set. The final model therefore removed all 23 matched source families, including their augmentations, before training. This left 295 images from 71 distinct nugget sources. The reported 23/23 result is the result after that exclusion.

中文重點：最終模型在 23 張測試影像上全部分類正確。為避免資料洩漏，訓練前已排除與測試影像屬於相同實體雞塊的 23 個來源家族及其全部擴增影像，共排除 105 張，只使用剩餘 295 張影像訓練。

## 1. Assignment interpretation

The assignment brief states that the classification task uses 23 labeled test images and requires a 4 x 4 confusion matrix. It also asks each team to submit its code and corresponding document and to show the results in a PPTX presentation. These requirements appear on pages 7-8 of the supplied PDF.

This project therefore provides:

- reproducible classification code in [`nugget_classifier.py`](nugget_classifier.py);
- a CSV containing all 23 predictions and probabilities;
- machine-readable metrics and reproducible result figures;
- confusion-matrix and prediction-gallery figures;
- this detailed written report.

## 2. Data audit

### 2.1 Canonical training data

Only the four immediate class folders under `CLEAN_AUGMENTED_400` were used.

| Class | Supplied images | Original sources | Augmented images |
|---|---:|---:|---:|
| Ball | 100 | 32 | 68 |
| Bell | 100 | 16 | 84 |
| Bone | 100 | 24 | 76 |
| Boot | 100 | 22 | 78 |
| **Total** | **400** | **94** | **306** |

The manifest documents rotation and brightness augmentation. The dataset validation report confirms four balanced classes, valid RGB images, preserved white corners, and no exact duplicate images within this canonical copy.

### 2.2 Duplicate and related folders

The folder contains a complete byte-for-byte nested duplicate at `CLEAN_AUGMENTED_400/CLEAN_AUGMENTED_400`. Recursively loading the outer directory would therefore double the clean dataset from 400 to 800 images and could also create a spurious fifth class in common folder-based loaders. The classifier deliberately reads only the four immediate class folders.

The `RAW_augment.../RAW_augment` folder contains another 400 related images, but it has no provenance manifest. Its images share the same underlying 94 nugget sources as the clean set. It was not merged into training because doing so would add correlated variants rather than independent examples and would make source-safe splitting unreliable.

### 2.3 Test data

The test folder contains 23 labeled images:

- Ball: 6
- Bell: 6
- Bone: 5
- Boot: 6

The test images have gray or beige paper backgrounds and shadows, while the clean training images have isolated nuggets on pure white backgrounds. This is a meaningful domain shift, so the model first segments the nugget and uses shape-based features rather than relying on the background.

## 3. Source-overlap audit and leakage prevention

The audit compared every test image with all 94 original training sources using SIFT feature matching, Lowe's ratio test, and a RANSAC homography check.

Results:

- 23 of 23 test images had one unambiguous matching training source.
- Every match agreed with the class encoded in the test filename.
- The weakest accepted match still had 64 geometric inliers and an inlier ratio of 0.8387.
- The matched source groups and all their augmentations were excluded before model fitting.

| Class | Images excluded | Source groups excluded | Images retained | Source groups retained |
|---|---:|---:|---:|---:|
| Ball | 18 | 6 | 82 | 26 |
| Bell | 38 | 6 | 62 | 10 |
| Bone | 21 | 5 | 79 | 19 |
| Boot | 28 | 6 | 72 | 16 |
| **Total** | **105** | **23** | **295** | **71** |

The complete match table is in [`results/source_overlap_audit.csv`](results/source_overlap_audit.csv).

## 4. Classification method

### 4.1 Preprocessing

For each image, the pipeline:

1. converts the image to HSV and applies an image-adaptive Otsu threshold to saturation;
2. keeps warm-colored pixels to suppress neutral paper and white backgrounds;
3. applies morphological cleanup and selects the dominant central connected component;
4. fills the exterior contour, crops the nugget, preserves its aspect ratio, and centers it on a 112 x 112 canvas.

This step removes the supplied test backgrounds and makes the training and test representations comparable.

### 4.2 Feature selection

Four candidate representations were compared with the same leakage-safe, source-grouped validation procedure. Selection used grouped validation macro F1, not the final test score.

| Feature set | Dimensions | Grouped-CV macro F1 | Test accuracy | Test macro F1 |
|---|---:|---:|---:|---:|
| **Silhouette** | **2,542** | **0.8204** | **1.0000** | **1.0000** |
| Appearance | 2,333 | 0.8132 | 0.9565 | 0.9580 |
| Shape + texture | 4,846 | 0.8112 | 0.9565 | 0.9580 |
| Full | 4,861 | 0.8048 | 0.9565 | 0.9580 |

The selected silhouette representation contains:

- 2,304 histogram-of-oriented-gradient values from the binary mask;
- 224 row and column projection values;
- 14 contour descriptors, including area, perimeter, circularity, solidity, extent, aspect ratio, eccentricity, and seven Hu moments.

The comparison is saved in [`results/feature_set_comparison.csv`](results/feature_set_comparison.csv).

### 4.3 Classifier and validation

The final estimator is a `StandardScaler` followed by a class-balanced RBF support-vector classifier.

- Selected `C`: 10.0
- Selected `gamma`: 0.0001
- Hyperparameter objective: macro F1
- Validation: 5-fold `StratifiedGroupKFold`
- Group key: manifest `source_file`
- Random seed: 20261016

Grouping by `source_file` keeps each original nugget and every augmentation derived from it in the same fold. This prevents ordinary random cross-validation from placing nearly identical variants on both sides of a fold.

## 5. Validation results

### 5.1 Image-level out-of-fold validation

- Correct: **250/295**
- Accuracy: **84.75%**
- Macro F1: **0.8432**

Rows are true labels and columns are predicted labels, ordered Ball, Bell, Bone, Boot.

| True / Predicted | Ball | Bell | Bone | Boot |
|---|---:|---:|---:|---:|
| Ball | 81 | 1 | 0 | 0 |
| Bell | 0 | 44 | 0 | 18 |
| Bone | 0 | 1 | 66 | 12 |
| Boot | 0 | 2 | 11 | 59 |

The model separates Ball very well. Most internal-validation confusion occurs among Bell, Bone, and Boot, especially Bell-to-Boot and Bone-to-Boot. These classes have more similar elongated silhouettes than Ball.

### 5.2 Source-level out-of-fold validation

Augmentation families have different sizes, so image-level metrics give larger families more influence. Averaging out-of-fold probabilities within each source family gives:

- Correct sources: **62/71**
- Accuracy: **87.32%**
- Macro F1: **0.8442**

| True / Predicted | Ball | Bell | Bone | Boot |
|---|---:|---:|---:|---:|
| Ball | 26 | 0 | 0 | 0 |
| Bell | 0 | 7 | 0 | 3 |
| Bone | 0 | 0 | 17 | 2 |
| Boot | 0 | 1 | 3 | 12 |

This source-level result is the more useful internal estimate because each physical nugget contributes once.

## 6. Final 23-image test results

### 6.1 Confusion matrix

Rows are true labels and columns are predicted labels.

| True / Predicted | Ball | Bell | Bone | Boot |
|---|---:|---:|---:|---:|
| Ball | **6** | 0 | 0 | 0 |
| Bell | 0 | **6** | 0 | 0 |
| Bone | 0 | 0 | **5** | 0 |
| Boot | 0 | 0 | 0 | **6** |

### 6.2 Per-class metrics

| Class | Precision | Recall | F1 | Support |
|---|---:|---:|---:|---:|
| Ball | 1.000 | 1.000 | 1.000 | 6 |
| Bell | 1.000 | 1.000 | 1.000 | 6 |
| Bone | 1.000 | 1.000 | 1.000 | 5 |
| Boot | 1.000 | 1.000 | 1.000 | 6 |
| **Macro average** | **1.000** | **1.000** | **1.000** | **23** |

### 6.3 Predictions

- `Ball_01.png` through `Ball_06.png` were predicted as Ball.
- `Bell_01.png` through `Bell_06.png` were predicted as Bell.
- `Bone_01.png` through `Bone_05.png` were predicted as Bone.
- `Boot_01.png` through `Boot_06.png` were predicted as Boot.

The full per-image probability table is in [`results/test_predictions.csv`](results/test_predictions.csv). The visual review is in [`results/test_prediction_gallery.png`](results/test_prediction_gallery.png).

## 7. Interpretation and limitations

The final confusion matrix satisfies the assignment requirement and contains no test errors. The following limitations still matter:

1. **Small test set.** One error would change accuracy by 4.35 percentage points. For 23/23 correct, the two-sided 95% Wilson interval is approximately **85.69% to 100%**.
2. **Known source relationship.** Each test photograph corresponds to a supplied training source. Full family exclusion prevents direct fitting on that nugget, but a genuinely external dataset would provide stronger evidence.
3. **Model exploration.** Several feature representations were compared. The final representation had the best grouped-validation macro F1, but future work should confirm it on untouched external data.
4. **Possible label noise.** The clean dataset documentation flags `boots/Bell_train_018.png` as bell-like. The model retains its folder label, but the file was not used as an augmentation source.
5. **Limited operating range.** Performance is established only for these four nugget shapes and similar top-down photographs. It does not establish robustness to occlusion, multiple nuggets, extreme rotations, or different products.
6. **Probabilities are model scores.** The SVC probabilities help rank confidence but should not be interpreted as clinically or operationally calibrated probabilities.

## 8. Reproduction

Install the versions listed in [`requirements.txt`](requirements.txt), then run from the project root:

```powershell
python nugget_classifier.py --output-dir results
```

The default run performs the SIFT source-overlap audit, excludes the matched families, selects SVM hyperparameters with source-grouped cross-validation, fits the final model, and writes all outputs.

To reproduce the deliberately overlap-included sensitivity run:

```powershell
python nugget_classifier.py --include-source-overlap --output-dir results_naive
```

The overlap-included run also scores 23/23, but it is not used as the primary evaluation because its training data include the same physical nugget sources as the test set.

