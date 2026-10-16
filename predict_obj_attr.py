#!/usr/bin/env python3
"""Evaluate SAP RPT on eligible masked object-attribute labels.

For every input CSV and every requested seed, the script:

1. keeps rows whose target is an observed binary label (0 or 1),
2. selects a stratified 10% query set (the simulated masked cells),
3. fits SAP RPT on the remaining 90% context rows,
4. predicts the query labels, and
5. writes run-, file-, and overall-level classification metrics.


The raw value column corresponding to the selected ``__exists`` target and
``object_id`` are excluded because they would leak the target or encourage
identifier-prefix memorisation. Experiment settings are defined below.
"""

from __future__ import annotations

import csv
import gc
import math
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)


DEFAULT_SEEDS = (42, 43, 44)
INPUT_DIR = Path("container_logistics_tabular")
# INPUT_DIR = Path("Manufacturing/manufact_tabular")
INPUT_PATTERNS = ("object_level.csv", "object_level_*.csv")
# OUTPUT_DIR = Path("Manufacturing/results_object_attribute")
OUTPUT_DIR = Path("results_object_attribute")
MISSING_RATE = 0.10
SEEDS = DEFAULT_SEEDS
MAX_CONTEXT_SIZE = 8192 # 2048 # 8192
BAGGING = 8
KEEP_OBJECT_ID = False
KEEP_TARGET_VALUE = False

# False forces the model to infer attribute presence without object_type.
WITH_OBJECT_TYPE = False
METRIC_COLUMNS = (
    "accuracy",
    "balanced_accuracy",
    "precision",
    "recall",
    "specificity",
    "f1",
    "f1_macro",
    "f1_weighted",
    "mcc",
    "roc_auc",
    "average_precision",
)


def detect_delimiter(path: Path) -> str:
    """Detect comma, semicolon, or tab delimiters from the beginning of a CSV."""
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        sample = handle.read(65_536)
    try:
        return csv.Sniffer().sniff(sample, delimiters=",;\t").delimiter
    except csv.Error:
        # The supplied object-level files use semicolons; this also remains a
        # safe fallback for similarly generated files.
        return ";"


def load_object_table(path: Path, target: str) -> tuple[pd.DataFrame, pd.Series, int]:
    delimiter = detect_delimiter(path)
    df = pd.read_csv(path, sep=delimiter, low_memory=False, encoding="utf-8-sig")
    df.columns = [str(column).strip() for column in df.columns]

    if target not in df.columns:
        raise KeyError(f"Required target column {target!r} is missing from {path}.")

    numeric_target = pd.to_numeric(df[target], errors="coerce")
    valid = numeric_target.isin([0, 1])
    dropped_rows = int((~valid).sum())
    clean_df = df.loc[valid].reset_index(drop=True)
    y = numeric_target.loc[valid].astype(np.int64).reset_index(drop=True)

    if clean_df.empty:
        raise ValueError(f"No observed binary labels found in {target!r} for {path}.")
    return clean_df, y, dropped_rows


def find_targets(df: pd.DataFrame) -> list[str]:
    """Find binary ``obj_attr__*__exists`` targets."""
    if "object_type" not in df.columns:
        raise KeyError("Required column 'object_type' is missing.")

    targets: list[str] = []
    for column in df.columns:
        column_name = str(column)
        if not (
            column_name.startswith("obj_attr__")
            and column_name.endswith("__exists")
        ):
            continue

        numeric = pd.to_numeric(df[column], errors="coerce")
        observed = numeric.dropna()
        if observed.empty or not observed.isin([0, 1]).all():
            continue

        targets.append(column_name)
            
    return targets


def exclusion_reason(
    df: pd.DataFrame,
    y: pd.Series,
) -> tuple[str | None, list[str]]:
    """Return why a file-target pair cannot support stratified evaluation."""
    n_positive = int((y == 1).sum())
    n_negative = int((y == 0).sum())

    if "object_type" not in df.columns:
        return "required object_type column is missing", []

    positive_object_types = sorted(
        {
            str(value).strip()
            for value in df.loc[y.eq(1), "object_type"].dropna()
            if str(value).strip()
        }
    )

    if n_positive < 2:
        return "positive class contains fewer than 2 object instances", positive_object_types
    if n_negative < 2:
        return (
            f"negative class contains only {n_negative} object instance(s); "
            "at least 2 are required",
            positive_object_types,
        )

    return None, positive_object_types


def stratified_mask_indices(
    y: pd.Series,
    missing_rate: float,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Randomly mask ``missing_rate`` of all rows, stratified by class.

    At least one row from each class is put in the query set and at least one
    is retained as context.
    """
    rng = np.random.default_rng(seed)
    y_array = y.to_numpy(dtype=np.int64)
    class_indices = {
        label: np.flatnonzero(y_array == label)
        for label in (0, 1)
    }

    n_mask_total = int(round(len(y_array) * missing_rate))
    n_mask_total = max(2, min(n_mask_total, len(y_array) - 2))

    positive_share = len(class_indices[1]) / len(y_array)
    n_positive = int(round(n_mask_total * positive_share))
    n_positive = max(1, min(n_positive, len(class_indices[1]) - 1))
    n_negative = n_mask_total - n_positive

    if n_negative < 1:
        n_negative = 1
        n_positive = n_mask_total - 1
    elif n_negative > len(class_indices[0]) - 1:
        n_negative = len(class_indices[0]) - 1
        n_positive = n_mask_total - n_negative

    query_parts = [
        rng.choice(class_indices[0], size=n_negative, replace=False),
        rng.choice(class_indices[1], size=n_positive, replace=False),
    ]

    query_indices = np.sort(np.concatenate(query_parts))
    context_indices = np.setdiff1d(
        np.arange(len(y_array)), query_indices, assume_unique=True
    )
    return context_indices, query_indices


def make_features(
    df: pd.DataFrame,
    target: str,
    keep_object_id: bool,
    keep_target_value: bool,
    with_object_type: bool,
) -> pd.DataFrame:
    drop_columns = [target]
    target_value_column = (
        target[: -len("__exists")] if target.endswith("__exists") else None
    )
    if (
        not keep_target_value
        and target_value_column
        and target_value_column in df.columns
    ):
        drop_columns.append(target_value_column)
    if not keep_object_id and "object_id" in df.columns:
        drop_columns.append("object_id")
    if not with_object_type and "object_type" in df.columns:
        drop_columns.append("object_type")

    X = df.drop(columns=drop_columns, errors="ignore").copy()
    # Completely empty features provide no information and can cause avoidable
    # embedding work or dtype problems.
    X = X.dropna(axis=1, how="all")
    if X.shape[1] == 0:
        raise ValueError("No predictor columns remain after leakage filtering.")
    return X


def positive_probabilities(classifier: object, X_query: pd.DataFrame) -> np.ndarray:
    probabilities = np.asarray(classifier.predict_proba(X_query))
    if probabilities.ndim == 1:
        return probabilities.astype(float)
    if probabilities.ndim != 2:
        raise ValueError(f"Unexpected predict_proba shape: {probabilities.shape}")

    classes = np.asarray(getattr(classifier, "classes_", []))
    if classes.size == probabilities.shape[1]:
        numeric_classes = pd.to_numeric(pd.Series(classes), errors="coerce").to_numpy()
        positive_indices = np.flatnonzero(numeric_classes == 1)
        if positive_indices.size == 1:
            return probabilities[:, int(positive_indices[0])].astype(float)

    if probabilities.shape[1] == 2:
        return probabilities[:, 1].astype(float)
    raise ValueError("Could not identify the probability column for positive class 1.")


def safe_probability_metric(metric_fn, y_true: np.ndarray, scores: np.ndarray) -> float:
    try:
        return float(metric_fn(y_true, scores))
    except ValueError:
        return math.nan


def calculate_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_score: np.ndarray,
) -> dict[str, float | int]:
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    specificity = tn / (tn + fp) if (tn + fp) else math.nan

    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, pos_label=1, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, pos_label=1, zero_division=0)),
        "specificity": float(specificity),
        "f1": float(f1_score(y_true, y_pred, pos_label=1, zero_division=0)),
        "f1_macro": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "f1_weighted": float(
            f1_score(y_true, y_pred, average="weighted", zero_division=0)
        ),
        "mcc": float(matthews_corrcoef(y_true, y_pred)),
        "roc_auc": safe_probability_metric(roc_auc_score, y_true, y_score),
        "average_precision": safe_probability_metric(
            average_precision_score, y_true, y_score
        ),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }


def release_model(classifier: object) -> None:
    del classifier
    gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except ImportError:
        pass


def write_run_checkpoint(records: Iterable[dict], output_path: Path) -> None:
    pd.DataFrame(records).to_csv(output_path, index=False)


def aggregate_results(per_run: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    metadata = {
        "n_runs": ("seed", "count"),
        "n_rows": ("n_rows", "first"),
        "mean_n_query": ("n_query", "mean"),
        "positive_rate": ("positive_rate", "first"),
    }
    metric_aggs = {metric: (metric, "mean") for metric in METRIC_COLUMNS}
    per_file = (
        per_run.groupby(["file", "target"], as_index=False)
        .agg(**metadata, **metric_aggs)
        .sort_values(["file", "target"])
        .reset_index(drop=True)
    )

    overall_rows = []
    for target, target_rows in per_file.groupby("target", sort=True):
        for metric in METRIC_COLUMNS:
            values = pd.to_numeric(target_rows[metric], errors="coerce")
            overall_rows.append(
                {
                    "target": target,
                    "metric": metric,
                    "mean_across_files": float(values.mean()),
                    "std_across_files": (
                        float(values.std(ddof=1)) if values.count() > 1 else 0.0
                    ),
                    "n_files": int(values.count()),
                }
            )
    overall = pd.DataFrame(overall_rows)
    return per_file, overall


def main() -> None:
    try:
        from sap_rpt_oss import SAP_RPT_OSS_Classifier
    except ImportError as exc:
        raise SystemExit(
            "sap_rpt_oss is not installed. Install the SAP RPT repository in your "
            "environment before running this script."
        ) from exc

    input_dir = INPUT_DIR.expanduser().resolve()
    output_dir = OUTPUT_DIR.expanduser().resolve()
    if not input_dir.is_dir():
        raise SystemExit(f"Input directory does not exist: {input_dir}")

    files = sorted(
        {
            path
            for pattern in INPUT_PATTERNS
            for path in input_dir.glob(pattern)
            if path.is_file() and output_dir not in path.parents
        }
    )
    if not files:
        raise SystemExit(
            f"No files matching {INPUT_PATTERNS!r} found in {input_dir}"
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    per_run_path = output_dir / "per_run_metrics.csv"
    per_file_path = output_dir / "per_file_mean_metrics.csv"
    overall_path = output_dir / "overall_mean_std_across_files.csv"
    skipped_path = output_dir / "skipped_file_targets.csv"

    run_records: list[dict] = []
    skipped_records: list[dict] = []

    print(f"Found {len(files)} input file(s).")
    print(f"Missing rate: {MISSING_RATE:.1%}")

    for file_number, path in enumerate(files, start=1):
        delimiter = detect_delimiter(path)
        source_df = pd.read_csv(
            path, sep=delimiter, low_memory=False, encoding="utf-8-sig"
        )
        source_df.columns = [str(column).strip() for column in source_df.columns]
        targets = find_targets(source_df)
        print(
            f"\n[{file_number}/{len(files)}] {path.name}: "
            f"found {len(targets)} eligible target(s)"
        )
        if not targets:
            skipped_records.append(
                {
                    "file": path.name,
                    "target": "",
                    "reason": "no binary obj_attr__*__exists column found",
                }
            )
            pd.DataFrame(skipped_records).to_csv(skipped_path, index=False)
            continue

        for target in targets:
            df, y, dropped_rows = load_object_table(path, target)
            n_positive = int((y == 1).sum())
            n_negative = int((y == 0).sum())
            reason, positive_object_types = exclusion_reason(
                df,
                y,
            )
            if reason is not None:
                skipped_records.append(
                    {
                        "file": path.name,
                        "target": target,
                        "n_valid_rows": len(y),
                        "n_positive": n_positive,
                        "n_negative": n_negative,
                        "n_positive_object_types": len(positive_object_types),
                        "positive_object_types": "|".join(positive_object_types),
                        "excluded_invalid_target_rows": dropped_rows,
                        "reason": reason,
                    }
                )
                pd.DataFrame(skipped_records).to_csv(skipped_path, index=False)
                print(f"  {target}: SKIPPED - {reason}")
                continue

            X = make_features(
                df,
                target=target,
                keep_object_id=KEEP_OBJECT_ID,
                keep_target_value=KEEP_TARGET_VALUE,
                with_object_type=WITH_OBJECT_TYPE,
            )
            print(
                f"  {target}: {len(df)} valid rows, "
                f"{dropped_rows} excluded rows, positive rate={y.mean():.3%}"
            )

            for seed in SEEDS:
                context_indices, query_indices = stratified_mask_indices(
                    y,
                    missing_rate=MISSING_RATE,
                    seed=seed,
                )

                X_context = X.iloc[context_indices].reset_index(drop=True)
                y_context = y.iloc[context_indices].reset_index(drop=True)
                X_query = X.iloc[query_indices].reset_index(drop=True)
                y_query = y.iloc[query_indices].to_numpy(dtype=np.int64)

                print(
                    f"    seed={seed}: context={len(context_indices)}, "
                    f"masked/query={len(query_indices)}"
                )

                classifier = SAP_RPT_OSS_Classifier(
                    max_context_size=MAX_CONTEXT_SIZE,
                    bagging=BAGGING,
                )
                classifier.fit(X_context, y_context)

                raw_predictions = classifier.predict(X_query)
                predictions = pd.to_numeric(
                    pd.Series(np.asarray(raw_predictions).reshape(-1)), errors="raise"
                ).astype(np.int64).to_numpy()
                scores = positive_probabilities(classifier, X_query)

                metrics = calculate_metrics(y_query, predictions, scores)
                run_records.append(
                    {
                        "file": path.name,
                        "seed": seed,
                        "missing_rate": MISSING_RATE,
                        "target": target,
                        "n_rows": len(df),
                        "n_positive": n_positive,
                        "n_negative": n_negative,
                        "n_positive_object_types": len(positive_object_types),
                        "positive_object_types": "|".join(positive_object_types),
                        "n_context": len(context_indices),
                        "n_query": len(query_indices),
                        "n_query_positive": int(y_query.sum()),
                        "n_query_negative": int(len(y_query) - y_query.sum()),
                        "positive_rate": float(y.mean()),
                        "excluded_invalid_target_rows": dropped_rows,
                        "max_context_size": MAX_CONTEXT_SIZE,
                        "bagging": BAGGING,
                        **metrics,
                    }
                )
                write_run_checkpoint(run_records, per_run_path)
                release_model(classifier)

    if skipped_records:
        pd.DataFrame(skipped_records).to_csv(skipped_path, index=False)

    if not run_records:
        raise SystemExit(
            "No eligible file-target pairs remained after filtering. "
            f"See {skipped_path} for details."
        )

    per_run = pd.DataFrame(run_records)
    per_file, overall = aggregate_results(per_run)
    per_file.to_csv(per_file_path, index=False)
    overall.to_csv(overall_path, index=False)

    print("\nSaved results:")
    print(f"  {per_run_path}")
    print(f"  {per_file_path}")
    print(f"  {overall_path}")
    if skipped_records:
        print(f"  {skipped_path}")


if __name__ == "__main__":
    main()
