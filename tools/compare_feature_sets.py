"""Diagnostic model comparison using only source-grouped validation for selection."""

from pathlib import Path
import csv
import sys

import numpy as np
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import GridSearchCV, StratifiedGroupKFold

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import nugget_classifier as nc  # noqa: E402


train_dir = ROOT / "CLEAN_AUGMENTED_400"
test_dir = ROOT / "Test Data for Classification-20261016"
all_train = nc.discover_training_records(train_dir)
test_records = nc.discover_test_records(test_dir)
overlap = {match.matched_group for match in nc.audit_source_overlap(train_dir, test_records)}
train_records = [record for record in all_train if record.group not in overlap]

train_x, train_y, groups, _, _ = nc.featurize(train_records)
test_x, test_y, _, _, _ = nc.featurize(test_records)

feature_sets = {
    name: nc.feature_columns(name)
    for name in ("silhouette", "appearance", "shape_texture", "full")
}
cv = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=nc.RANDOM_SEED)
grid = {"svc__C": [1.0, 10.0, 30.0], "svc__gamma": ["scale", 0.0001]}

rows = []
for name, columns in feature_sets.items():
    search = GridSearchCV(
        nc.build_pipeline(), grid, scoring="f1_macro", cv=cv, n_jobs=1, refit=True
    )
    search.fit(train_x[:, columns], train_y, groups=groups)
    predictions = search.best_estimator_.predict(test_x[:, columns])
    row = {
        "feature_set": name,
        "feature_dimensions": len(columns),
        "grouped_cv_macro_f1": f"{search.best_score_:.6f}",
        "test_accuracy": f"{accuracy_score(test_y, predictions):.6f}",
        "test_macro_f1": f"{f1_score(test_y, predictions, average='macro'):.6f}",
        "best_C": search.best_params_["svc__C"],
        "best_gamma": search.best_params_["svc__gamma"],
        "test_errors": ";".join(
            test_records[index].path.name
            for index in np.flatnonzero(predictions != test_y)
        ),
    }
    rows.append(row)
    print(row)

output_path = ROOT / "results" / "feature_set_comparison.csv"
output_path.parent.mkdir(parents=True, exist_ok=True)
with output_path.open("w", encoding="utf-8", newline="") as stream:
    writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)
print(output_path)
