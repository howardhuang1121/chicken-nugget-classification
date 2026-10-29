"""Leakage-aware four-class chicken nugget image classifier.

The pipeline is intentionally compact and reproducible:

1. Segment the warm/brown nugget from the nearly neutral background.
2. Normalize its silhouette without changing the aspect ratio.
3. Extract silhouette, gradient, projection, contour, and color candidates.
4. Select an RBF-SVM setting with source-grouped stratified cross-validation.
5. Fit all supplied training images and evaluate the 23 labeled test images.

Run from the repository root with:

    python nugget_classifier.py
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import platform
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import cv2
import joblib
import matplotlib
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from scipy import ndimage
from skimage.feature import hog
from sklearn import __version__ as sklearn_version
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from sklearn.model_selection import GridSearchCV, StratifiedGroupKFold, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


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
HOG_LENGTH = len(
    hog(
        np.zeros((NORMALIZED_SIZE, NORMALIZED_SIZE), dtype=np.float32),
        orientations=9,
        pixels_per_cell=(12, 12),
        cells_per_block=(2, 2),
        block_norm="L2-Hys",
        feature_vector=True,
    )
)
PROJECTION_LENGTH = 2 * NORMALIZED_SIZE
CONTOUR_LENGTH = 14
COLOR_LENGTH = 15


@dataclass(frozen=True)
class ImageRecord:
    path: Path
    label: str
    group: str


@dataclass(frozen=True)
class SourceMatch:
    test_filename: str
    test_label: str
    matched_group: str
    matched_label: str
    source_filename: str
    good_matches: int
    inliers: int
    inlier_ratio: float
    runner_up_group: str
    runner_up_inliers: int


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--train-dir",
        type=Path,
        default=root / "CLEAN_AUGMENTED_400",
        help="Folder containing ball/bell/bone/boots subfolders.",
    )
    parser.add_argument(
        "--test-dir",
        type=Path,
        default=root / "Test Data for Classification-20261016",
        help="Folder containing the 23 labeled test PNGs.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=root / "results",
        help="Directory for metrics, figures, predictions, and model artifact.",
    )
    parser.add_argument(
        "--cv-folds",
        type=int,
        default=5,
        help="Number of source-grouped stratified validation folds.",
    )
    parser.add_argument(
        "--include-source-overlap",
        action="store_true",
        help=(
            "Keep training families that match test photos. This produces a deliberately "
            "leakage-prone sensitivity baseline; the default excludes them."
        ),
    )
    parser.add_argument(
        "--feature-set",
        choices=("silhouette", "appearance", "shape_texture", "full"),
        default="silhouette",
        help=(
            "Feature representation. The default silhouette set was selected by "
            "source-grouped validation and is least sensitive to background shift."
        ),
    )
    return parser.parse_args()


def read_rgb(path: Path) -> np.ndarray:
    with Image.open(path) as image:
        return np.asarray(image.convert("RGB"), dtype=np.uint8)


def segment_nugget(rgb: np.ndarray) -> np.ndarray:
    """Return a filled binary mask for the dominant warm-colored object."""

    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    red = rgb[..., 0].astype(np.int16)
    green = rgb[..., 1].astype(np.int16)
    blue = rgb[..., 2].astype(np.int16)
    saturation = hsv[..., 1]
    value = hsv[..., 2]

    # Nuggets are consistently warm and chromatic; paper/white backgrounds are
    # neutral. An image-adaptive Otsu threshold handles both pure-white cutouts
    # and the gray/beige paper backgrounds in the supplied test photographs.
    warm = (red >= green - 2) & (green >= blue - 8) & (red >= blue + 8)
    otsu_threshold, _ = cv2.threshold(
        saturation, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )
    saturation_threshold = max(45, int(round(0.90 * otsu_threshold)))
    mask = warm & (saturation >= saturation_threshold) & (value <= 254)
    mask = mask.astype(np.uint8)

    min_dimension = min(rgb.shape[:2])
    kernel_size = max(3, int(round(min_dimension * 0.012)))
    if kernel_size % 2 == 0:
        kernel_size += 1
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)

    count, labels, stats, centroids = cv2.connectedComponentsWithStats(mask, 8)
    if count <= 1:
        raise ValueError("No foreground component found")

    height, width = mask.shape
    center = np.array([width / 2.0, height / 2.0])
    candidates: list[tuple[float, int]] = []
    for component in range(1, count):
        area = float(stats[component, cv2.CC_STAT_AREA])
        if area < 0.002 * height * width:
            continue
        distance = np.linalg.norm(centroids[component] - center) / max(width, height)
        score = area * max(0.20, 1.0 - distance)
        candidates.append((score, component))
    if not candidates:
        component = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    else:
        component = max(candidates)[1]

    selected = labels == component
    selected = ndimage.binary_fill_holes(selected)
    selected = cv2.morphologyEx(
        selected.astype(np.uint8), cv2.MORPH_CLOSE, kernel, iterations=1
    )

    contours, _ = cv2.findContours(selected, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        raise ValueError("Foreground mask contains no contour")
    contour = max(contours, key=cv2.contourArea)
    filled = np.zeros_like(selected)
    cv2.drawContours(filled, [contour], -1, 1, thickness=cv2.FILLED)
    return filled.astype(bool)


def normalize_object(
    rgb: np.ndarray, mask: np.ndarray, size: int = NORMALIZED_SIZE
) -> tuple[np.ndarray, np.ndarray]:
    """Crop, scale with preserved aspect ratio, and center the foreground."""

    ys, xs = np.nonzero(mask)
    if len(xs) == 0:
        raise ValueError("Cannot normalize an empty mask")
    x0, x1 = int(xs.min()), int(xs.max()) + 1
    y0, y1 = int(ys.min()), int(ys.max()) + 1
    object_rgb = rgb[y0:y1, x0:x1]
    object_mask = mask[y0:y1, x0:x1].astype(np.uint8)

    margin = 8
    scale = min((size - 2 * margin) / object_rgb.shape[1], (size - 2 * margin) / object_rgb.shape[0])
    target_width = max(1, int(round(object_rgb.shape[1] * scale)))
    target_height = max(1, int(round(object_rgb.shape[0] * scale)))
    resized_rgb = cv2.resize(
        object_rgb, (target_width, target_height), interpolation=cv2.INTER_AREA
    )
    resized_mask = cv2.resize(
        object_mask, (target_width, target_height), interpolation=cv2.INTER_NEAREST
    ).astype(bool)

    canvas_rgb = np.full((size, size, 3), 255, dtype=np.uint8)
    canvas_mask = np.zeros((size, size), dtype=bool)
    left = (size - target_width) // 2
    top = (size - target_height) // 2
    canvas_rgb[top : top + target_height, left : left + target_width] = resized_rgb
    canvas_mask[top : top + target_height, left : left + target_width] = resized_mask
    canvas_rgb[~canvas_mask] = 255
    return canvas_rgb, canvas_mask


def _contour_features(mask: np.ndarray) -> np.ndarray:
    mask_u8 = mask.astype(np.uint8)
    contours, _ = cv2.findContours(mask_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    contour = max(contours, key=cv2.contourArea)
    area = float(cv2.contourArea(contour))
    perimeter = float(cv2.arcLength(contour, closed=True))
    x, y, width, height = cv2.boundingRect(contour)
    hull = cv2.convexHull(contour)
    hull_area = float(cv2.contourArea(hull))
    circularity = 4.0 * math.pi * area / max(perimeter * perimeter, 1e-12)
    solidity = area / max(hull_area, 1e-12)
    extent = area / max(float(width * height), 1e-12)
    aspect = width / max(float(height), 1.0)

    if len(contour) >= 5:
        _, axes, _ = cv2.fitEllipse(contour)
        eccentricity = min(axes) / max(max(axes), 1e-12)
    else:
        eccentricity = 1.0

    moments = cv2.moments(contour)
    hu = cv2.HuMoments(moments).flatten()
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


def extract_features(rgb: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    mask = segment_nugget(rgb)
    normalized_rgb, normalized_mask = normalize_object(rgb, mask)

    mask_float = normalized_mask.astype(np.float32)
    gray = cv2.cvtColor(normalized_rgb, cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0
    # A constant background prevents paper texture from becoming a shortcut.
    gray[~normalized_mask] = 1.0

    mask_hog = hog(
        mask_float,
        orientations=9,
        pixels_per_cell=(12, 12),
        cells_per_block=(2, 2),
        block_norm="L2-Hys",
        feature_vector=True,
    )
    gray_hog = hog(
        gray,
        orientations=9,
        pixels_per_cell=(12, 12),
        cells_per_block=(2, 2),
        block_norm="L2-Hys",
        feature_vector=True,
    )

    horizontal = normalized_mask.mean(axis=1)
    vertical = normalized_mask.mean(axis=0)
    contour = _contour_features(normalized_mask)

    foreground = normalized_rgb[normalized_mask].astype(np.float32) / 255.0
    color_statistics = np.concatenate(
        [
            foreground.mean(axis=0),
            foreground.std(axis=0),
            np.quantile(foreground, (0.10, 0.50, 0.90), axis=0).reshape(-1),
        ]
    )
    features = np.concatenate(
        [
            mask_hog,
            gray_hog,
            horizontal,
            vertical,
            contour,
            color_statistics,
        ]
    ).astype(np.float32)
    return features, normalized_rgb, normalized_mask


def feature_columns(feature_set: str) -> np.ndarray:
    mask_start = 0
    gray_start = HOG_LENGTH
    projection_start = 2 * HOG_LENGTH
    contour_start = projection_start + PROJECTION_LENGTH
    color_start = contour_start + CONTOUR_LENGTH
    full_end = color_start + COLOR_LENGTH
    selections = {
        "silhouette": np.r_[mask_start:gray_start, projection_start:color_start],
        "appearance": np.r_[gray_start:projection_start, contour_start:full_end],
        "shape_texture": np.r_[mask_start:color_start],
        "full": np.r_[mask_start:full_end],
    }
    if feature_set not in selections:
        raise ValueError(f"Unknown feature set: {feature_set}")
    return selections[feature_set]


def load_manifest_groups(train_dir: Path) -> dict[tuple[str, str], str]:
    manifest_path = train_dir / "augmentation_manifest.csv"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Training manifest not found: {manifest_path}")
    mapping: dict[tuple[str, str], str] = {}
    with manifest_path.open("r", encoding="utf-8-sig", newline="") as stream:
        for row in csv.DictReader(stream):
            label = row["class"].strip().lower()
            output_name = row["output_file"].strip()
            source_name = row["source_file"].strip()
            mapping[(label, output_name)] = f"{label}/{source_name}"
    return mapping


def discover_original_records(train_dir: Path) -> list[ImageRecord]:
    manifest_path = train_dir / "augmentation_manifest.csv"
    records: list[ImageRecord] = []
    with manifest_path.open("r", encoding="utf-8-sig", newline="") as stream:
        for row in csv.DictReader(stream):
            if row["is_augmented"].strip().lower() != "false":
                continue
            label = row["class"].strip().lower()
            filename = row["output_file"].strip()
            source = row["source_file"].strip()
            records.append(
                ImageRecord(
                    path=train_dir / label / filename,
                    label=label,
                    group=f"{label}/{source}",
                )
            )
    return records


def _sift_keypoints(path: Path) -> tuple[list[cv2.KeyPoint], np.ndarray | None]:
    gray = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if gray is None:
        raise ValueError(f"Cannot decode image for SIFT audit: {path}")
    detector = cv2.SIFT_create(nfeatures=2500)
    return detector.detectAndCompute(gray, None)


def _geometric_match_score(
    query: tuple[list[cv2.KeyPoint], np.ndarray | None],
    candidate: tuple[list[cv2.KeyPoint], np.ndarray | None],
) -> tuple[int, int, float]:
    query_keypoints, query_descriptors = query
    candidate_keypoints, candidate_descriptors = candidate
    if query_descriptors is None or candidate_descriptors is None:
        return 0, 0, 0.0
    matcher = cv2.BFMatcher(cv2.NORM_L2)
    pairs = matcher.knnMatch(query_descriptors, candidate_descriptors, k=2)
    good = [first for first, second in pairs if first.distance < 0.75 * second.distance]
    if len(good) < 4:
        return len(good), 0, 0.0
    query_points = np.float32([query_keypoints[item.queryIdx].pt for item in good])
    candidate_points = np.float32(
        [candidate_keypoints[item.trainIdx].pt for item in good]
    )
    _, inlier_mask = cv2.findHomography(
        query_points, candidate_points, cv2.RANSAC, ransacReprojThreshold=5.0
    )
    inliers = int(inlier_mask.sum()) if inlier_mask is not None else 0
    return len(good), inliers, inliers / len(good)


def audit_source_overlap(
    train_dir: Path, test_records: list[ImageRecord]
) -> list[SourceMatch]:
    """Match each test photo against all unaugmented training sources using SIFT."""

    originals = discover_original_records(train_dir)
    original_features = {
        record.group: _sift_keypoints(record.path) for record in originals
    }
    matches: list[SourceMatch] = []
    for test_record in test_records:
        test_features = _sift_keypoints(test_record.path)
        ranked: list[tuple[int, float, int, ImageRecord]] = []
        for original in originals:
            good, inliers, ratio = _geometric_match_score(
                test_features, original_features[original.group]
            )
            ranked.append((inliers, ratio, good, original))
        ranked.sort(key=lambda item: (item[0], item[1], item[2]), reverse=True)
        best, runner_up = ranked[0], ranked[1]
        inliers, ratio, good, original = best
        if inliers < 25 or ratio < 0.75 or inliers < 2 * max(runner_up[0], 1):
            raise ValueError(
                f"No unambiguous source match for {test_record.path.name}: "
                f"best={original.group} ({inliers} inliers, ratio={ratio:.3f}), "
                f"runner-up={runner_up[3].group} ({runner_up[0]} inliers)"
            )
        matches.append(
            SourceMatch(
                test_filename=test_record.path.name,
                test_label=test_record.label,
                matched_group=original.group,
                matched_label=original.label,
                source_filename=original.path.name,
                good_matches=good,
                inliers=inliers,
                inlier_ratio=ratio,
                runner_up_group=runner_up[3].group,
                runner_up_inliers=runner_up[0],
            )
        )
    return matches


def save_source_overlap_audit(matches: list[SourceMatch], output_path: Path) -> None:
    with output_path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(SourceMatch.__dataclass_fields__))
        writer.writeheader()
        for match in matches:
            writer.writerow(
                {
                    **match.__dict__,
                    "inlier_ratio": f"{match.inlier_ratio:.6f}",
                }
            )


def discover_training_records(train_dir: Path) -> list[ImageRecord]:
    group_mapping = load_manifest_groups(train_dir)
    records: list[ImageRecord] = []
    for label in CLASS_NAMES:
        class_dir = train_dir / label
        if not class_dir.is_dir():
            raise FileNotFoundError(f"Training class directory not found: {class_dir}")
        for path in sorted(class_dir.glob("*.png")):
            key = (label, path.name)
            if key not in group_mapping:
                raise KeyError(f"No manifest row for {label}/{path.name}")
            records.append(ImageRecord(path=path, label=label, group=group_mapping[key]))
    return records


def discover_test_records(test_dir: Path) -> list[ImageRecord]:
    records: list[ImageRecord] = []
    for path in sorted(test_dir.glob("*.png")):
        prefix = path.stem.split("_", 1)[0].lower()
        if prefix not in TEST_PREFIX_TO_CLASS:
            raise ValueError(f"Cannot derive test label from filename: {path.name}")
        label = TEST_PREFIX_TO_CLASS[prefix]
        records.append(ImageRecord(path=path, label=label, group=path.name))
    return records


def featurize(
    records: Iterable[ImageRecord],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[np.ndarray], list[np.ndarray]]:
    features: list[np.ndarray] = []
    labels: list[int] = []
    groups: list[str] = []
    normalized_images: list[np.ndarray] = []
    normalized_masks: list[np.ndarray] = []
    for record in records:
        feature, normalized_rgb, normalized_mask = extract_features(read_rgb(record.path))
        features.append(feature)
        labels.append(CLASS_TO_INDEX[record.label])
        groups.append(record.group)
        normalized_images.append(normalized_rgb)
        normalized_masks.append(normalized_mask)
    return (
        np.vstack(features),
        np.asarray(labels, dtype=np.int64),
        np.asarray(groups),
        normalized_images,
        normalized_masks,
    )


def build_pipeline() -> Pipeline:
    return Pipeline(
        [
            ("scale", StandardScaler()),
            (
                "svc",
                SVC(
                    kernel="rbf",
                    class_weight="balanced",
                    probability=True,
                    random_state=RANDOM_SEED,
                ),
            ),
        ]
    )


def save_confusion_matrix(
    matrix: np.ndarray, title: str, output_path: Path, subtitle: str
) -> None:
    figure, axis = plt.subplots(figsize=(7.4, 6.4), dpi=160)
    image = axis.imshow(matrix, cmap="Blues", vmin=0, vmax=max(int(matrix.max()), 1))
    axis.set_xticks(range(len(DISPLAY_NAMES)), labels=DISPLAY_NAMES)
    axis.set_yticks(range(len(DISPLAY_NAMES)), labels=DISPLAY_NAMES)
    axis.set_xlabel("Predicted label")
    axis.set_ylabel("True label")
    axis.set_title(f"{title}\n{subtitle}", pad=14)
    for row in range(matrix.shape[0]):
        for column in range(matrix.shape[1]):
            value = int(matrix[row, column])
            color = "white" if value > matrix.max() / 2 else "#17324d"
            axis.text(column, row, str(value), ha="center", va="center", color=color, fontsize=13)
    figure.colorbar(image, ax=axis, fraction=0.046, pad=0.04, label="Image count")
    figure.tight_layout()
    figure.savefig(output_path, bbox_inches="tight")
    plt.close(figure)


def save_prediction_gallery(
    records: list[ImageRecord],
    normalized_images: list[np.ndarray],
    true_labels: np.ndarray,
    predictions: np.ndarray,
    confidences: np.ndarray,
    output_path: Path,
) -> None:
    columns = 6
    tile_width, tile_height = 220, 230
    rows = math.ceil(len(records) / columns)
    gallery = Image.new("RGB", (columns * tile_width, rows * tile_height), "white")
    draw = ImageDraw.Draw(gallery)
    font = ImageFont.load_default()
    for index, (record, normalized) in enumerate(zip(records, normalized_images)):
        row, column = divmod(index, columns)
        x, y = column * tile_width, row * tile_height
        preview = Image.fromarray(normalized).resize((180, 180), Image.Resampling.LANCZOS)
        correct = int(true_labels[index]) == int(predictions[index])
        border = "#25843a" if correct else "#bd2828"
        draw.rectangle((x + 15, y + 5, x + 205, y + 222), outline=border, width=4)
        gallery.paste(preview, (x + 20, y + 10))
        predicted_name = DISPLAY_NAMES[int(predictions[index])]
        true_name = DISPLAY_NAMES[int(true_labels[index])]
        draw.text((x + 20, y + 194), record.path.name, fill="black", font=font)
        draw.text(
            (x + 20, y + 208),
            f"true={true_name} pred={predicted_name} p={confidences[index]:.3f}",
            fill=border,
            font=font,
        )
    gallery.save(output_path)


def json_safe_classification_report(report: dict) -> dict:
    result: dict = {}
    for key, value in report.items():
        if isinstance(value, dict):
            result[key] = {subkey: float(number) for subkey, number in value.items()}
        else:
            result[key] = float(value)
    return result


def aggregate_probabilities_by_group(
    probabilities: np.ndarray, labels: np.ndarray, groups: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    group_labels: list[int] = []
    group_probabilities: list[np.ndarray] = []
    ordered_groups = np.unique(groups)
    for group in ordered_groups:
        indices = np.flatnonzero(groups == group)
        labels_in_group = np.unique(labels[indices])
        if len(labels_in_group) != 1:
            raise ValueError(f"Source group spans multiple labels: {group}")
        group_labels.append(int(labels_in_group[0]))
        group_probabilities.append(probabilities[indices].mean(axis=0))
    return ordered_groups, np.asarray(group_labels), np.vstack(group_probabilities)


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    all_train_records = discover_training_records(args.train_dir)
    test_records = discover_test_records(args.test_dir)
    source_matches = audit_source_overlap(args.train_dir, test_records)
    save_source_overlap_audit(
        source_matches, args.output_dir / "source_overlap_audit.csv"
    )
    mismatched_classes = [
        match
        for match in source_matches
        if match.test_label != match.matched_label
    ]
    if mismatched_classes:
        details = ", ".join(
            f"{match.test_filename}->{match.matched_group}" for match in mismatched_classes
        )
        raise ValueError(f"Source matches conflict with test filename labels: {details}")

    overlapping_groups = {match.matched_group for match in source_matches}
    if args.include_source_overlap:
        train_records = all_train_records
    else:
        train_records = [
            record for record in all_train_records if record.group not in overlapping_groups
        ]

    print(f"Training images supplied: {len(all_train_records)}")
    print(f"Matched test/training source families: {len(overlapping_groups)}")
    print(f"Training images used: {len(train_records)}")
    print(f"Independent test images: {len(test_records)}")

    train_x_all, train_y, train_groups, _, _ = featurize(train_records)
    test_x_all, test_y, _, test_normalized, _ = featurize(test_records)
    selected_columns = feature_columns(args.feature_set)
    train_x = train_x_all[:, selected_columns]
    test_x = test_x_all[:, selected_columns]
    print(f"Feature set: {args.feature_set}")
    print(f"Feature dimensions: {train_x.shape[1]}")
    print(f"Unique source groups: {len(np.unique(train_groups))}")

    cv = StratifiedGroupKFold(
        n_splits=args.cv_folds, shuffle=True, random_state=RANDOM_SEED
    )
    parameter_grid = {
        "svc__C": [1.0, 10.0, 30.0],
        "svc__gamma": ["scale", 0.0001],
    }
    search = GridSearchCV(
        build_pipeline(),
        parameter_grid,
        scoring="f1_macro",
        cv=cv,
        n_jobs=1,
        refit=True,
        return_train_score=False,
    )
    search.fit(train_x, train_y, groups=train_groups)
    best_model = search.best_estimator_
    print(f"Best grouped-CV macro F1: {search.best_score_:.4f}")
    print(f"Best parameters: {search.best_params_}")

    # Out-of-fold estimates use the selected settings while preserving source groups.
    cv_probabilities = cross_val_predict(
        build_pipeline().set_params(**search.best_params_),
        train_x,
        train_y,
        groups=train_groups,
        cv=cv,
        method="predict_proba",
        n_jobs=1,
    )
    cv_predictions = cv_probabilities.argmax(axis=1)
    cv_accuracy = accuracy_score(train_y, cv_predictions)
    cv_macro_f1 = f1_score(train_y, cv_predictions, average="macro")
    cv_matrix = confusion_matrix(train_y, cv_predictions, labels=range(len(CLASS_NAMES)))
    _, cv_source_y, cv_source_probabilities = aggregate_probabilities_by_group(
        cv_probabilities, train_y, train_groups
    )
    cv_source_predictions = cv_source_probabilities.argmax(axis=1)
    cv_source_accuracy = accuracy_score(cv_source_y, cv_source_predictions)
    cv_source_macro_f1 = f1_score(
        cv_source_y, cv_source_predictions, average="macro"
    )
    cv_source_matrix = confusion_matrix(
        cv_source_y, cv_source_predictions, labels=range(len(CLASS_NAMES))
    )

    test_probabilities = best_model.predict_proba(test_x)
    test_predictions = best_model.classes_[test_probabilities.argmax(axis=1)]
    test_confidences = test_probabilities.max(axis=1)
    test_accuracy = accuracy_score(test_y, test_predictions)
    test_macro_f1 = f1_score(test_y, test_predictions, average="macro")
    test_matrix = confusion_matrix(test_y, test_predictions, labels=range(len(CLASS_NAMES)))
    test_report = classification_report(
        test_y,
        test_predictions,
        labels=range(len(CLASS_NAMES)),
        target_names=DISPLAY_NAMES,
        output_dict=True,
        zero_division=0,
    )

    predictions_path = args.output_dir / "test_predictions.csv"
    with predictions_path.open("w", encoding="utf-8", newline="") as stream:
        fieldnames = [
            "filename",
            "true_label",
            "predicted_label",
            "correct",
            "confidence",
            *[f"probability_{name}" for name in CLASS_NAMES],
        ]
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        for index, record in enumerate(test_records):
            row = {
                "filename": record.path.name,
                "true_label": record.label,
                "predicted_label": CLASS_NAMES[int(test_predictions[index])],
                "correct": bool(test_y[index] == test_predictions[index]),
                "confidence": f"{test_confidences[index]:.6f}",
            }
            for class_index, class_name in enumerate(CLASS_NAMES):
                row[f"probability_{class_name}"] = f"{test_probabilities[index, class_index]:.6f}"
            writer.writerow(row)

    save_confusion_matrix(
        cv_matrix,
        "Source-grouped cross-validation confusion matrix",
        args.output_dir / "cross_validation_confusion_matrix.png",
        f"accuracy={cv_accuracy:.1%}, macro F1={cv_macro_f1:.3f}, n={len(train_y)}",
    )
    save_confusion_matrix(
        cv_source_matrix,
        "Source-level cross-validation confusion matrix",
        args.output_dir / "cross_validation_source_confusion_matrix.png",
        (
            f"accuracy={cv_source_accuracy:.1%}, macro F1={cv_source_macro_f1:.3f}, "
            f"n={len(cv_source_y)} sources"
        ),
    )
    save_confusion_matrix(
        test_matrix,
        "Independent test-set confusion matrix",
        args.output_dir / "test_confusion_matrix.png",
        f"accuracy={test_accuracy:.1%}, macro F1={test_macro_f1:.3f}, n={len(test_y)}",
    )
    save_prediction_gallery(
        test_records,
        test_normalized,
        test_y,
        test_predictions,
        test_confidences,
        args.output_dir / "test_prediction_gallery.png",
    )

    joblib.dump(
        {
            "model": best_model,
            "classes": CLASS_NAMES,
            "display_names": DISPLAY_NAMES,
            "feature_version": 1,
            "feature_set": args.feature_set,
            "feature_columns": selected_columns,
            "normalized_size": NORMALIZED_SIZE,
        },
        args.output_dir / "nugget_classifier.joblib",
    )

    metrics = {
        "run": {
            "random_seed": RANDOM_SEED,
            "python": platform.python_version(),
            "scikit_learn": sklearn_version,
            "command": " ".join(sys.argv),
        },
        "data": {
            "training_images_supplied": int(len(all_train_records)),
            "training_images": int(len(train_records)),
            "training_source_groups": int(len(np.unique(train_groups))),
            "training_class_counts": {
                CLASS_NAMES[index]: int((train_y == index).sum())
                for index in range(len(CLASS_NAMES))
            },
            "test_images": int(len(test_records)),
            "test_class_counts": {
                CLASS_NAMES[index]: int((test_y == index).sum())
                for index in range(len(CLASS_NAMES))
            },
            "feature_dimensions": int(train_x.shape[1]),
            "candidate_feature_dimensions": int(train_x_all.shape[1]),
        },
        "source_overlap_audit": {
            "method": "SIFT ratio test plus RANSAC homography against 94 original sources",
            "matched_test_images": int(len(source_matches)),
            "matched_source_groups": int(len(overlapping_groups)),
            "excluded_from_training": not args.include_source_overlap,
            "excluded_training_images": int(len(all_train_records) - len(train_records)),
            "all_matches_agree_with_filename_labels": not bool(mismatched_classes),
            "minimum_inliers": int(min(match.inliers for match in source_matches)),
            "minimum_inlier_ratio": float(
                min(match.inlier_ratio for match in source_matches)
            ),
        },
        "model_selection": {
            "cv_method": f"{args.cv_folds}-fold StratifiedGroupKFold by source_file",
            "scoring": "macro F1",
            "best_score": float(search.best_score_),
            "best_parameters": {
                key: (float(value) if isinstance(value, (float, int)) else value)
                for key, value in search.best_params_.items()
            },
            "feature_set": args.feature_set,
        },
        "cross_validation": {
            "accuracy": float(cv_accuracy),
            "macro_f1": float(cv_macro_f1),
            "confusion_matrix": cv_matrix.tolist(),
            "source_level_accuracy": float(cv_source_accuracy),
            "source_level_macro_f1": float(cv_source_macro_f1),
            "source_level_confusion_matrix": cv_source_matrix.tolist(),
        },
        "test": {
            "accuracy": float(test_accuracy),
            "macro_f1": float(test_macro_f1),
            "confusion_matrix": test_matrix.tolist(),
            "classification_report": json_safe_classification_report(test_report),
            "mean_confidence": float(test_confidences.mean()),
            "minimum_confidence": float(test_confidences.min()),
            "incorrect_count": int((test_y != test_predictions).sum()),
        },
    }
    with (args.output_dir / "metrics.json").open("w", encoding="utf-8") as stream:
        json.dump(metrics, stream, ensure_ascii=False, indent=2)

    print(f"Grouped-CV accuracy: {cv_accuracy:.4f}; macro F1: {cv_macro_f1:.4f}")
    print(
        "Source-level grouped-CV accuracy: "
        f"{cv_source_accuracy:.4f}; macro F1: {cv_source_macro_f1:.4f}"
    )
    print(f"Test accuracy: {test_accuracy:.4f}; macro F1: {test_macro_f1:.4f}")
    print(f"Results written to: {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
