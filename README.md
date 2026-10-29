# Chicken Nugget Shape Classification

This repository contains a reproducible, leakage-aware classifier for four chicken nugget shapes: **Ball**, **Bell**, **Bone**, and **Boot**.

The main experiment audits source overlap between the supplied training and test images, removes every matching training family, keeps all augmentations from the same original nugget in the same cross-validation fold, and evaluates a shape-based RBF-SVM.

## Main results

| Evaluation | Accuracy | Macro F1 |
|---|---:|---:|
| Image-level grouped cross-validation | 84.75% (250/295) | 0.8432 |
| Source-level grouped cross-validation | 87.32% (62/71) | 0.8442 |
| Fixed 23-image test set | 100.00% (23/23) | 1.0000 |

The perfect test score should be interpreted carefully because the test set contains only 23 images. The source audit found that each test image matched one original training source, so all 23 matched source families and their augmentations were excluded before final training. This removed 105 images and left 295 training images from 71 source groups.

## Repository contents

```text
.
|-- nugget_classifier.py
|-- requirements.txt
|-- DETAILED_CLASSIFICATION_REPORT.md
|-- CLEAN_AUGMENTED_400/
|   |-- augmentation_manifest.csv
|   |-- validation_report.json
|   |-- ball/
|   |-- bell/
|   |-- bone/
|   `-- boots/
|-- Test Data for Classification-20261016/
|-- results/
`-- tools/
    |-- compare_feature_sets.py
    `-- shape_knn_benchmark.py
```

`CLEAN_AUGMENTED_400` contains the canonical 400-image training set only. A byte-for-byte nested duplicate from the working folder is intentionally not included.

## Method

The main pipeline:

1. segments the warm-colored nugget from the background;
2. normalizes the silhouette on a 112 x 112 canvas;
3. extracts 2,304 mask-HOG values, 224 projection values, and 14 contour descriptors;
4. selects an RBF-SVM with five-fold `StratifiedGroupKFold` using the original source file as the group;
5. fits the retained source-safe training data and evaluates the fixed test set.

All images derived from one original nugget remain in the same fold. This prevents an augmented version of a nugget from appearing in training while another version of the same nugget appears in validation.

## Setup and reproduction

Python 3.11 was used for the reported run.

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python nugget_classifier.py
```

The default command writes the model metrics, prediction table, source-overlap audit, confusion matrices, and prediction gallery to `results/`.

Optional diagnostics:

```powershell
python tools/compare_feature_sets.py
python tools/shape_knn_benchmark.py
```

## Important limitations

- The fixed test set is small: one error changes accuracy by about 4.35 percentage points.
- The source audit reduces direct train/test source leakage, but this test set is not a newly collected external dataset.
- Broader generalization should be confirmed with independently collected nuggets under different backgrounds and lighting conditions.

See [DETAILED_CLASSIFICATION_REPORT.md](DETAILED_CLASSIFICATION_REPORT.md) for the full experimental explanation and [results/metrics.json](results/metrics.json) for machine-readable parameters and metrics.

## Repository scope

This clean upload contains only the source-grouped SVM analysis and the files needed to reproduce it. PowerPoint files, presentation build artifacts, temporary files, duplicate datasets, exploratory leakage-prone results, and the separate five-model workflow are intentionally excluded.
