# Repository scope and provenance

This upload is intentionally limited to the leakage-aware, source-grouped SVM experiment represented by `nugget_classifier.py`, its diagnostic tools, its final result files, and the data required to reproduce that experiment.

## Included analysis files

- `nugget_classifier.py`
- `tools/compare_feature_sets.py`
- `tools/shape_knn_benchmark.py`
- `DETAILED_CLASSIFICATION_REPORT.md`
- `requirements.txt`
- final CSV, JSON, and image outputs under `results/`

## Included data

- the canonical outer copy of `CLEAN_AUGMENTED_400` (400 training images and its manifest)
- the 23 supplied test images

Before publishing the repository publicly, the repository owner should confirm that the supplied image data may be redistributed.

## Intentionally excluded

- the separate five-model workflow under `nuggets_classification_2026/`
- all PowerPoint files and presentation-generation sources
- the nested duplicate of `CLEAN_AUGMENTED_400`
- the unused raw augmentation folder
- the deliberately leakage-prone sensitivity baseline in `results_naive/`
- generated model binaries, caches, renders, and temporary inspection files

These exclusions prevent unrelated code, duplicate data, and presentation artifacts from being mixed into the reproducible classifier repository.
