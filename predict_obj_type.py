"""
Generated for SAP RPT 1-OSS object-level missing-data experiments.

Requirements:
    pip install git+https://github.com/SAP-samples/sap-rpt-1-oss
    pip install pandas numpy scikit-learn

Authenticate with Hugging Face before first use because the SAP checkpoint
is downloaded from SAP/sap-rpt-1-oss.

The scripts use:
    missingness rate = 10%
    seeds = [42, 43, 44]
    max_context_size = 2048
    bagging = 1

Change FILES and/or INPUT_GLOB below to point at your object_level CSV files.
"""

from pathlib import Path
import warnings

import numpy as np
import pandas as pd

from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
)

from sap_rpt_oss import SAP_RPT_OSS_Classifier


# ============================================================
# CONFIG
# ============================================================

FILES = [
    # Add individual CSV files here when needed.
]

# Match object_level.csv, object_level_2.csv, and other object_level*.csv files.
INPUT_GLOB = "Manufacturing/manufact_tabular/object_level*.csv"



OUTPUT_DIR = Path("Manufacturing/results_object_type")

MISSING_RATE = 0.10
SEEDS = [42, 43, 44]

MAX_CONTEXT_SIZE = 8192
BAGGING = 8

TARGET = "object_type"

# object_id can reveal type through prefixes such as cr / co / td / vh.
DROP_OBJECT_ID = True

# CRITICAL:
# __exists column encodes the object schema and therefore
# directly leak object type. It is removed from this task.
DROP_SCHEMA_INDICATOR_COLUMNS = True


# ============================================================
# Helpers
# ============================================================

def read_csv_auto(path):
    return pd.read_csv(path, sep=None, engine="python")


def discover_files():
    paths = [Path(p) for p in FILES]
    paths.extend(Path().glob(INPUT_GLOB))

    result = []
    seen = set()

    for p in sorted(paths, key=lambda path: str(path)):
        p = p.resolve()
        if p.is_file() and p not in seen:
            result.append(p)
            seen.add(p)

    return result


def classification_metrics(y_true, y_pred):
    y_true = pd.Series(y_true).astype(str)
    y_pred = pd.Series(y_pred).astype(str)

    p_macro, r_macro, f1_macro, _ = precision_recall_fscore_support(
        y_true,
        y_pred,
        average="macro",
        zero_division=0,
    )

    p_weighted, r_weighted, f1_weighted, _ = precision_recall_fscore_support(
        y_true,
        y_pred,
        average="weighted",
        zero_division=0,
    )

    return {
        "Accuracy": accuracy_score(y_true, y_pred),
        "Precision_macro": p_macro,
        "Recall_macro": r_macro,
        "F1_macro": f1_macro,
        "Precision_weighted": p_weighted,
        "Recall_weighted": r_weighted,
        "F1_weighted": f1_weighted,
    }


def stratified_query_indices(y, rate, seed):
    """
    Mask approximately 10% within every object-type class.

    This is preferable to purely random masking for rare object types because
    it gives each sufficiently represented class a chance to appear in the
    query set while preserving context examples.
    """
    rng = np.random.default_rng(seed)

    selected = []

    for cls in y.dropna().unique():
        cls_idx = y.index[y == cls].to_numpy()

        if len(cls_idx) <= 1:
            continue

        n = max(
            1,
            int(round(len(cls_idx) * rate)),
        )

        # preserve at least one context example
        n = min(n, len(cls_idx) - 1)

        selected.extend(
            rng.choice(
                cls_idx,
                size=n,
                replace=False,
            ).tolist()
        )

    return np.asarray(selected)


def prepare_features(df):
    drop_cols = [TARGET]

    if DROP_OBJECT_ID and "object_id" in df.columns:
        drop_cols.append("object_id")

    if DROP_SCHEMA_INDICATOR_COLUMNS:
        drop_cols.extend([
            c for c in df.columns
            if c.endswith("__exists")
        ])

    return df.drop(
        columns=drop_cols,
        errors="ignore",
    )


# ============================================================
# One experiment
# ============================================================

def run_one(df, file_name, seed):
    if TARGET not in df.columns:
        warnings.warn(
            f"{file_name}: {TARGET!r} not found; skipped."
        )
        return None

    valid_idx = df.index[df[TARGET].notna()]
    y = df.loc[valid_idx, TARGET].astype(str)

    if y.nunique() < 2:
        warnings.warn(
            f"{file_name}: object_type has fewer than 2 classes; skipped."
        )
        return None

    query_idx = stratified_query_indices(
        y,
        MISSING_RATE,
        seed,
    )

    if len(query_idx) == 0:
        warnings.warn(
            f"{file_name} / seed {seed}: no query rows; skipped."
        )
        return None

    context_idx = np.setdiff1d(
        valid_idx.to_numpy(),
        query_idx,
    )

    y_context = df.loc[
        context_idx,
        TARGET,
    ].astype(str)

    y_true = df.loc[
        query_idx,
        TARGET,
    ].astype(str)

    if y_context.nunique() < 2:
        warnings.warn(
            f"{file_name} / seed {seed}: "
            f"context has fewer than 2 classes; skipped."
        )
        return None

    X = prepare_features(df)

    X_context = X.loc[
        context_idx
    ].copy()

    X_query = X.loc[
        query_idx
    ].copy()

    model = SAP_RPT_OSS_Classifier(
        max_context_size=MAX_CONTEXT_SIZE,
        bagging=BAGGING,
    )

    model.fit(
        X_context,
        y_context,
    )

    y_pred = model.predict(
        X_query
    )

    metrics = classification_metrics(
        y_true,
        y_pred,
    )

    return {
        "file": file_name,
        "target": TARGET,
        "seed": seed,
        "missing_rate": MISSING_RATE,
        "n_objects": len(valid_idx),
        "n_masked": len(query_idx),
        "n_classes": y.nunique(),
        **metrics,
    }


# ============================================================
# Aggregation
# ============================================================

def save_results(results):
    result_df = pd.DataFrame(results)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    result_df.to_csv(
        OUTPUT_DIR / "per_run_metrics.csv",
        index=False,
    )

    id_cols = {
        "file",
        "target",
        "seed",
        "missing_rate",
        "n_objects",
        "n_masked",
        "n_classes",
    }

    metric_cols = [
        c for c in result_df.columns
        if c not in id_cols
        and pd.api.types.is_numeric_dtype(result_df[c])
    ]

    per_file = (
        result_df
        .groupby(
            ["file", "target"],
            dropna=False,
        )[metric_cols]
        .mean()
        .reset_index()
    )

    per_file.to_csv(
        OUTPUT_DIR / "per_file_mean_metrics.csv",
        index=False,
    )

    overall = (
        per_file
        .groupby(
            ["target"],
            dropna=False,
        )[metric_cols]
        .agg(["mean", "std"])
    )

    overall.to_csv(
        OUTPUT_DIR / "overall_mean_std_across_files.csv"
    )

    return result_df, per_file, overall


def main():
    files = discover_files()

    if not files:
        raise FileNotFoundError(
            "No files found."
        )

    results = []

    for file_path in files:
        print(f"\n=== {file_path.name} ===")

        df = read_csv_auto(file_path)

        for seed in SEEDS:
            print(f"  object_type | seed={seed}")

            result = run_one(
                df,
                file_path.name,
                seed,
            )

            if result is not None:
                results.append(result)

    if not results:
        print("No valid experiments were run.")
        return

    result_df, per_file, overall = save_results(results)

    print("\n=== Per-run metrics ===")
    print(result_df.to_string(index=False))

    print("\n=== Per-file mean over 3 seeds ===")
    print(per_file.to_string(index=False))

    print("\n=== Overall mean ± SD across files ===")
    print(overall)

    print(f"\nSaved to: {OUTPUT_DIR.resolve()}")


if __name__ == "__main__":
    main()
