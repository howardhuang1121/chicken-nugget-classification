"""Reproduce the leakage-aware 2,049-feature shape 1-NN benchmark.

The benchmark is deliberately independent of ``nugget_classifier.py`` and is
read-only by default.  It:

1. reads only the four immediate class directories listed by the manifest;
2. excludes every augmentation family whose source photograph appears in the
   supplied 23-image test set;
3. segments and normalizes the nugget silhouette;
4. evaluates a standardized Euclidean 1-NN with source-grouped five-fold CV;
5. reports both image-level and source-family-level OOF metrics; and
6. fits the retained data and classifies the 23 test images.

Run from anywhere with:

    python tools/shape_knn_benchmark.py

The script prints results only; it does not write into ``results/``.
"""

from __future__ import annotations

import argparse
import csv
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import cv2
import numpy as np
from PIL import Image
from scipy import ndimage
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from sklearn.model_selection import StratifiedGroupKFold, cross_val_predict
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


CLASS_NAMES = ("ball", "bell", "bone", "boots")
DISPLAY_NAMES = ("Ball", "Bell", "Bone", "Boot")
CLASS_TO_INDEX = {name: index for index, name in enumerate(CLASS_NAMES)}
TEST_PREFIX_TO_CLASS = {
    "ball": "ball",
    "bell": "bell",
    "bone": "bone",
    "boot": "boots",
    "boots": "boots",
}
RANDOM_SEED = 20261016
NORMALIZED_SIZE = 112
FEATURE_COUNT = 2049

# SIFT/RANSAC source audit result.  The CSV produced by nugget_classifier.py is
# used when present; this embedded copy makes the benchmark self-contained.
EXPECTED_SOURCE_BY_TEST = {
    "Ball_01.png": "ball/Ball_train_011.png",
    "Ball_02.png": "ball/Ball_train_012.png",
    "Ball_03.png": "ball/Ball_train_013.png",
    "Ball_04.png": "ball/Ball_train_014.png",
    "Ball_05.png": "ball/Ball_train_015.png",
    "Ball_06.png": "ball/Ball_train_002.png",
    "Bell_01.png": "bell/Bell_train_010.png",
    "Bell_02.png": "bell/Bell_train_011.png",
    "Bell_03.png": "bell/Bell_train_012.png",
    "Bell_04.png": "bell/Bell_train_013.png",
    "Bell_05.png": "bell/Bell_train_014.png",
    "Bell_06.png": "bell/Bell_train_004.png",
    "Bone_01.png": "bone/Bone_train_011.png",
    "Bone_02.png": "bone/Bone_train_014.png",
    "Bone_03.png": "bone/Bone_train_017.png",
    "Bone_04.png": "bone/Bone_train_018.png",
    "Bone_05.png": "bone/Bone_train_006.png",
    "Boot_01.png": "boots/Boot_train_014.png",
    "Boot_02.png": "boots/Boot_train_018.png",
    "Boot_03.png": "boots/Boot_train_020.png",
    "Boot_04.png": "boots/Boot_train_019.png",
    "Boot_05.png": "boots/Boot_train_001.png",
    "Boot_06.png": "boots/Boot_train_005.png",
}


@dataclass(frozen=True)
class ImageRecord:
    path: Path
    label: str
    group: str


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--train-dir",
        type=Path,
        default=root / "CLEAN_AUGMENTED_400",
        help="Canonical clean dataset containing the manifest and four class folders.",
    )
    parser.add_argument(
        "--test-dir",
        type=Path,
        default=root / "Test Data for Classification-20261016",
        help="Folder containing the 23 labeled test images.",
    )
    parser.add_argument(
        "--overlap-audit",
        type=Path,
        default=root / "results" / "source_overlap_audit.csv",
        help=(
            "Optional SIFT source-audit CSV. The embedded verified mapping is used "
            "when this file does not exist."
        ),
    )
    parser.add_argument(
        "--cv-folds",
        type=int,
        default=5,
        help="Number of stratified source-grouped folds (default: 5).",
    )
    return parser.parse_args()


def read_rgb(path: Path) -> np.ndarray:
    with Image.open(path) as image:
        return np.asarray(image.convert("RGB"), dtype=np.uint8)


def segment_nugget(rgb: np.ndarray) -> np.ndarray:
    """Segment the dominant warm object and return one filled boolean contour."""

    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    red = rgb[..., 0].astype(np.int16)
    green = rgb[..., 1].astype(np.int16)
    blue = rgb[..., 2].astype(np.int16)
    saturation = hsv[..., 1]
    value = hsv[..., 2]

    warm = (red >= green - 2) & (green >= blue - 8) & (red >= blue + 8)
    chromatic = (saturation >= 28) & (value <= 252)
    dark_warm = (saturation >= 18) & (value <= 205) & (red >= blue + 5)
    mask = ((warm & chromatic) | dark_warm).astype(np.uint8)

    min_dimension = min(rgb.shape[:2])
    kernel_size = max(3, int(round(min_dimension * 0.012)))
    if kernel_size % 2 == 0:
        kernel_size += 1
    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (kernel_size, kernel_size)
    )
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)

    component_count, labels, stats, centroids = cv2.connectedComponentsWithStats(
        mask, 8
    )
    if component_count <= 1:
        raise ValueError("No foreground component found")

    height, width = mask.shape
    center = np.asarray([width / 2.0, height / 2.0])
    candidates: list[tuple[float, int]] = []
    for component in range(1, component_count):
        area = float(stats[component, cv2.CC_STAT_AREA])
        if area < 0.002 * height * width:
            continue
        distance = np.linalg.norm(centroids[component] - center) / max(width, height)
        score = area * max(0.20, 1.0 - distance)
        candidates.append((score, component))
    if candidates:
        selected_component = max(candidates)[1]
    else:
        selected_component = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))

    selected = ndimage.binary_fill_holes(labels == selected_component)
    selected = cv2.morphologyEx(
        selected.astype(np.uint8), cv2.MORPH_CLOSE, kernel, iterations=1
    )
    contours, _ = cv2.findContours(
        selected, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    if not contours:
        raise ValueError("Foreground mask contains no contour")
    contour = max(contours, key=cv2.contourArea)
    filled = np.zeros_like(selected)
    cv2.drawContours(filled, [contour], -1, 1, thickness=cv2.FILLED)
    return filled.astype(bool)


def normalize_mask(mask: np.ndarray, size: int = NORMALIZED_SIZE) -> np.ndarray:
    """Tight-crop and center a mask with preserved aspect ratio and 8 px margin."""

    ys, xs = np.nonzero(mask)
    if len(xs) == 0:
        raise ValueError("Cannot normalize an empty mask")
    x0, x1 = int(xs.min()), int(xs.max()) + 1
    y0, y1 = int(ys.min()), int(ys.max()) + 1
    cropped = mask[y0:y1, x0:x1].astype(np.uint8)

    margin = 8
    scale = min(
        (size - 2 * margin) / cropped.shape[1],
        (size - 2 * margin) / cropped.shape[0],
    )
    target_width = max(1, int(round(cropped.shape[1] * scale)))
    target_height = max(1, int(round(cropped.shape[0] * scale)))
    resized = cv2.resize(
        cropped,
        (target_width, target_height),
        interpolation=cv2.INTER_NEAREST,
    ).astype(bool)

    canvas = np.zeros((size, size), dtype=bool)
    left = (size - target_width) // 2
    top = (size - target_height) // 2
    canvas[top : top + target_height, left : left + target_width] = resized
    return canvas


def contour_features(mask: np.ndarray) -> np.ndarray:
    """Seven geometry values followed by seven signed log-Hu moments."""

    mask_u8 = mask.astype(np.uint8)
    contours, _ = cv2.findContours(
        mask_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE
    )
    contour = max(contours, key=cv2.contourArea)
    area = float(cv2.contourArea(contour))
    perimeter = float(cv2.arcLength(contour, closed=True))
    _, _, width, height = cv2.boundingRect(contour)
    hull_area = float(cv2.contourArea(cv2.convexHull(contour)))
    circularity = 4.0 * math.pi * area / max(perimeter * perimeter, 1e-12)
    solidity = area / max(hull_area, 1e-12)
    extent = area / max(float(width * height), 1e-12)
    aspect = width / max(float(height), 1.0)
    if len(contour) >= 5:
        _, axes, _ = cv2.fitEllipse(contour)
        eccentricity = min(axes) / max(max(axes), 1e-12)
    else:
        eccentricity = 1.0
    hu = cv2.HuMoments(cv2.moments(contour)).flatten()
    hu = -np.sign(hu) * np.log10(np.maximum(np.abs(hu), 1e-30))
    return np.asarray(
        [
            area / mask.size,
            perimeter / (2.0 * (mask.shape[0] + mask.shape[1])),
            circularity,
            solidity,
            extent,
            aspect,
            eccentricity,
            *hu.tolist(),
        ],
        dtype=np.float32,
    )


def shape_features(mask: np.ndarray) -> np.ndarray:
    """Return the exact 2,049-dimensional winning shape representation.

    Layout (half-open indices):
      0:784       28x28 antialiased binary silhouette
      784:1360    24x24 antialiased signed-distance silhouette
      1360:1696   112 row triples: first, last, foreground fraction
      1696:2032   112 column triples: first, last, foreground fraction
      2032:2046   seven contour scalars plus seven Hu moments
      2046:2049   left/right, up/down, and 180-degree asymmetry
    """

    m = mask.astype(np.uint8)
    silhouette = cv2.resize(
        m.astype(np.float32), (28, 28), interpolation=cv2.INTER_AREA
    ).ravel()
    inside = cv2.distanceTransform(m, cv2.DIST_L2, 3)
    outside = cv2.distanceTransform(1 - m, cv2.DIST_L2, 3)
    signed_distance = cv2.resize(
        (inside - outside) / 96.0,
        (24, 24),
        interpolation=cv2.INTER_AREA,
    ).ravel()

    row_profiles: list[float] = []
    for index in range(NORMALIZED_SIZE):
        foreground = np.flatnonzero(m[index])
        if len(foreground):
            row_profiles.extend(
                [
                    foreground[0] / 111.0,
                    foreground[-1] / 111.0,
                    len(foreground) / 112.0,
                ]
            )
        else:
            row_profiles.extend([0.5, 0.5, 0.0])

    column_profiles: list[float] = []
    for index in range(NORMALIZED_SIZE):
        foreground = np.flatnonzero(m[:, index])
        if len(foreground):
            column_profiles.extend(
                [
                    foreground[0] / 111.0,
                    foreground[-1] / 111.0,
                    len(foreground) / 112.0,
                ]
            )
        else:
            column_profiles.extend([0.5, 0.5, 0.0])

    asymmetry = np.asarray(
        [
            np.mean(m != np.fliplr(m)),
            np.mean(m != np.flipud(m)),
            np.mean(m != np.rot90(m, 2)),
        ],
        dtype=np.float32,
    )
    features = np.concatenate(
        [
            silhouette,
            signed_distance,
            np.asarray(row_profiles, dtype=np.float32),
            np.asarray(column_profiles, dtype=np.float32),
            contour_features(mask),
            asymmetry,
        ]
    ).astype(np.float32)
    if len(features) != FEATURE_COUNT:
        raise AssertionError(f"Expected {FEATURE_COUNT} features, got {len(features)}")
    return features


def load_training_records(train_dir: Path) -> list[ImageRecord]:
    """Load manifest rows, avoiding the byte-identical nested dataset copy."""

    manifest_path = train_dir / "augmentation_manifest.csv"
    records: list[ImageRecord] = []
    with manifest_path.open("r", encoding="utf-8-sig", newline="") as stream:
        for row in csv.DictReader(stream):
            label = row["class"].strip().lower()
            if label not in CLASS_TO_INDEX:
                raise ValueError(f"Unexpected class in manifest: {label}")
            output_name = row["output_file"].strip()
            source_name = row["source_file"].strip()
            path = train_dir / label / output_name
            if not path.is_file():
                raise FileNotFoundError(path)
            records.append(
                ImageRecord(path=path, label=label, group=f"{label}/{source_name}")
            )
    return records


def load_test_records(test_dir: Path) -> list[ImageRecord]:
    records: list[ImageRecord] = []
    for path in sorted(test_dir.glob("*.png")):
        prefix = path.stem.split("_", 1)[0].lower()
        if prefix not in TEST_PREFIX_TO_CLASS:
            raise ValueError(f"Cannot derive reference label from {path.name}")
        records.append(
            ImageRecord(
                path=path,
                label=TEST_PREFIX_TO_CLASS[prefix],
                group=path.name,
            )
        )
    return records


def load_excluded_groups(
    audit_path: Path, test_records: list[ImageRecord]
) -> set[str]:
    """Read the verified SIFT audit, falling back to the embedded mapping."""

    mapping = dict(EXPECTED_SOURCE_BY_TEST)
    if audit_path.is_file():
        with audit_path.open("r", encoding="utf-8-sig", newline="") as stream:
            audit_mapping = {
                row["test_filename"]: row["matched_group"]
                for row in csv.DictReader(stream)
            }
        disagreements = {
            name: (mapping.get(name), group)
            for name, group in audit_mapping.items()
            if mapping.get(name) != group
        }
        if disagreements:
            raise ValueError(f"Overlap audit disagrees with embedded mapping: {disagreements}")
        mapping = audit_mapping

    test_names = {record.path.name for record in test_records}
    if set(mapping) != test_names:
        raise ValueError(
            "Test files and source mapping differ: "
            f"missing={sorted(test_names - set(mapping))}, "
            f"extra={sorted(set(mapping) - test_names)}"
        )
    return set(mapping.values())


def featurize(
    records: Iterable[ImageRecord],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    features: list[np.ndarray] = []
    labels: list[int] = []
    groups: list[str] = []
    for record in records:
        raw_mask = segment_nugget(read_rgb(record.path))
        mask = normalize_mask(raw_mask)
        features.append(shape_features(mask))
        labels.append(CLASS_TO_INDEX[record.label])
        groups.append(record.group)
    return (
        np.vstack(features),
        np.asarray(labels, dtype=np.int64),
        np.asarray(groups),
    )


def build_model():
    return make_pipeline(
        StandardScaler(),
        KNeighborsClassifier(
            n_neighbors=1,
            weights="distance",
            metric="minkowski",
            p=2,
        ),
    )


def metric_line(name: str, truth: np.ndarray, prediction: np.ndarray) -> str:
    return (
        f"{name}: n={len(truth)}, accuracy={accuracy_score(truth, prediction):.6f}, "
        f"macro_f1={f1_score(truth, prediction, average='macro'):.6f}"
    )


def aggregate_source_probabilities(
    probabilities: np.ndarray,
    labels: np.ndarray,
    groups: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    true_labels: list[int] = []
    predictions: list[int] = []
    ordered_groups = sorted(set(groups.tolist()))
    for group in ordered_groups:
        indices = np.flatnonzero(groups == group)
        group_labels = np.unique(labels[indices])
        if len(group_labels) != 1:
            raise ValueError(f"Source group {group} has multiple labels")
        true_labels.append(int(group_labels[0]))
        predictions.append(int(np.argmax(probabilities[indices].mean(axis=0))))
    return (
        np.asarray(true_labels, dtype=np.int64),
        np.asarray(predictions, dtype=np.int64),
        ordered_groups,
    )


def main() -> None:
    args = parse_args()
    all_training = load_training_records(args.train_dir)
    test_records = load_test_records(args.test_dir)
    excluded_groups = load_excluded_groups(args.overlap_audit, test_records)
    training = [record for record in all_training if record.group not in excluded_groups]

    print(
        f"training: retained {len(training)}/{len(all_training)} images, "
        f"{len(set(record.group for record in training))} source groups; "
        f"excluded {len(all_training) - len(training)} images in "
        f"{len(excluded_groups)} test-overlap groups"
    )
    print(
        "features: 2049 = 784 silhouette + 576 signed distance + "
        "336 row profiles + 336 column profiles + 14 contour/Hu + 3 asymmetry"
    )

    train_x, train_y, groups = featurize(training)
    test_x, test_y, _ = featurize(test_records)
    cv = StratifiedGroupKFold(
        n_splits=args.cv_folds,
        shuffle=True,
        random_state=RANDOM_SEED,
    )
    oof_probabilities = cross_val_predict(
        build_model(),
        train_x,
        train_y,
        groups=groups,
        cv=cv,
        method="predict_proba",
        n_jobs=1,
    )
    oof_predictions = np.argmax(oof_probabilities, axis=1)
    print(metric_line("image-level grouped OOF", train_y, oof_predictions))
    print(
        "image-level confusion [Ball, Bell, Bone, Boot]:\n"
        f"{confusion_matrix(train_y, oof_predictions, labels=range(4))}"
    )

    source_y, source_predictions, source_groups = aggregate_source_probabilities(
        oof_probabilities, train_y, groups
    )
    print(metric_line("source-level grouped OOF", source_y, source_predictions))
    print(
        "source-level confusion [Ball, Bell, Bone, Boot]:\n"
        f"{confusion_matrix(source_y, source_predictions, labels=range(4))}"
    )
    print("source-level errors:")
    for group, truth, prediction in zip(
        source_groups, source_y.tolist(), source_predictions.tolist()
    ):
        if truth != prediction:
            print(f"  {group}: {DISPLAY_NAMES[truth]} -> {DISPLAY_NAMES[prediction]}")

    model = build_model().fit(train_x, train_y)
    test_predictions = model.predict(test_x)
    test_probabilities = model.predict_proba(test_x)
    print(metric_line("23-image test", test_y, test_predictions))
    print(
        "test confusion [Ball, Bell, Bone, Boot]:\n"
        f"{confusion_matrix(test_y, test_predictions, labels=range(4))}"
    )
    print("test predictions:")
    for record, prediction, probabilities in zip(
        test_records, test_predictions.tolist(), test_probabilities
    ):
        print(
            f"  {record.path.name}: {DISPLAY_NAMES[prediction]} "
            f"(p={probabilities[prediction]:.6f})"
        )


if __name__ == "__main__":
    main()
