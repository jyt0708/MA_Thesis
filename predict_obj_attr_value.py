"""
Generated for SAP RPT 1-OSS object-level missing-data experiments.

Authenticate with Hugging Face before first use because the SAP checkpoint
is downloaded from SAP/sap-rpt-1-oss.

The scripts use:
    missingness rate = 10%
    seeds = [42, 43, 44]
    max_context_size = 8192
    bagging = 8

"""

from pathlib import Path
import warnings

warnings.filterwarnings(
    "ignore",
    message=r"Data type .* not recognized! Defaulting to string",
)

import numpy as np
import pandas as pd

from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)

from sap_rpt_oss import SAP_RPT_OSS_Classifier, SAP_RPT_OSS_Regressor


# ============================================================
# CONFIG
# ============================================================

# INPUT_DIR = Path("Manufacturing/manufact_tabular/") 
INPUT_DIR = Path("container_logistics_tabular_new")
INPUT_PATTERNS = ("object_level.csv", "object_level_*.csv")

# OUTPUT_DIR = Path("Manufacturing/results_object_attribute_values")
OUTPUT_DIR = Path("results_object_attribute_values")

MISSING_RATE = 0.10
SEEDS = [42, 43, 44]

MAX_CONTEXT_SIZE = 8192
BAGGING = 8

TARGET_TYPES = {
    "obj_attr__Amount of Containers": "regression",
    "obj_attr__Amount of Goods": "regression",
    "obj_attr__Amount of Handling Units": "regression",
    "obj_attr__Departure Date": "regression",
    "obj_attr__Status": "classification",
    "obj_attr__Weight": "regression",
}

# object_id is dropped to avoid memorizing object identity.
DROP_FEATURE_COLUMNS = ["object_id"]


# ============================================================
# Helpers
# ============================================================

def read_csv_auto(path):
    return pd.read_csv(path, sep=None, engine="python")


def numeric_regression_target(values, target):
    if target == "obj_attr__Departure Date":
        datetimes = pd.to_datetime(
            values,
            dayfirst=True,
            errors="raise",
        )
        # Express dates as days since 2013-01-01 so regression metrics remain
        # interpretable in days rather than nanoseconds.
        return (
            datetimes - pd.Timestamp("2013-01-01")
        ).dt.total_seconds() / 86_400

    return pd.to_numeric(values, errors="raise")


def discover_files():
    input_dir = INPUT_DIR.expanduser().resolve()
    if not input_dir.is_dir():
        return []

    return sorted(
        {
            path.resolve()
            for pattern in INPUT_PATTERNS
            for path in input_dir.glob(pattern)
            if path.is_file()
        }
    )


def find_attribute_value_columns(df):
    """
    Returns columns such as:
        obj_attr__Status

    but not:
        obj_attr__Status__exists
    """
    return [
        c for c in df.columns
        if c.startswith("obj_attr__")
        and not c.endswith("__exists")
    ]


def regression_metrics(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)

    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred) if len(y_true) >= 2 else np.nan
    if not np.isfinite(r2):
        r2 = 1.0

    nonzero = np.abs(y_true) > 1e-12
    mape = (
        np.mean(np.abs((y_true[nonzero] - y_pred[nonzero]) / y_true[nonzero])) * 100
        if nonzero.any()
        else np.nan
    )

    return {
        "RMSE": rmse,
        "MAE": mae,
        "R2": r2,
        "MAPE_percent": mape,
    }


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


def make_model(task_type):
    if task_type == "regression":
        return SAP_RPT_OSS_Regressor(
            max_context_size=MAX_CONTEXT_SIZE,
            bagging=BAGGING,
        )

    return SAP_RPT_OSS_Classifier(
        max_context_size=MAX_CONTEXT_SIZE,
        bagging=BAGGING,
    )


def paired_columns(target):
    return (
        f"{target}__exists"
    )


def prepare_features(df, target):
    """
    Remove:
      - target itself
      - paired __exists flags
      - object_id

    Dropping the paired flags avoids direct leakage that the target attribute
    exists/is applicable on the queried row.
    """
    exists_col = paired_columns(target)

    drop_cols = [
        target,
        exists_col,
        *DROP_FEATURE_COLUMNS,
    ]

    return df.drop(columns=drop_cols, errors="ignore")


# ============================================================
# One experiment
# ============================================================

def run_one(df, file_name, target, seed):
    exists_col = paired_columns(target)

    # Only objects for which this attribute is schema-applicable and observed
    # are eligible for a missing-value experiment.
    eligible = df[target].notna()

    if exists_col in df.columns:
        eligible &= pd.to_numeric(
            df[exists_col], errors="coerce"
        ).eq(1)

    eligible_idx = df.index[eligible].to_numpy()

    if len(eligible_idx) < 2:
        warnings.warn(
            f"{file_name} / {target}: fewer than 2 eligible rows; skipped."
        )
        return None

    rng = np.random.default_rng(seed)

    n_mask = max(1, int(round(len(eligible_idx) * MISSING_RATE)))
    n_mask = min(n_mask, len(eligible_idx) - 1)

    query_idx = rng.choice(
        eligible_idx,
        size=n_mask,
        replace=False,
    )

    context_idx = np.setdiff1d(
        eligible_idx,
        query_idx,
    )

    y = df[target]
    task_type = TARGET_TYPES[target]

    if task_type == "regression":
        y_context = numeric_regression_target(
            y.loc[context_idx], target,
        )
        y_true = numeric_regression_target(
            y.loc[query_idx], target,
        )
    else:
        y_context = y.loc[context_idx].astype(str)
        y_true = y.loc[query_idx].astype(str)

        if y_context.nunique() < 2:
            warnings.warn(
                f"{file_name} / {target} / seed {seed}: "
                f"classification context has <2 classes; skipped."
            )
            return None

    X = prepare_features(df, target)

    X_context = X.loc[context_idx].copy()
    X_query = X.loc[query_idx].copy()

    model = make_model(task_type)
    model.fit(X_context, y_context)
    y_pred = model.predict(X_query)

    metrics = (
        regression_metrics(y_true, y_pred)
        if task_type == "regression"
        else classification_metrics(y_true, y_pred)
    )

    return {
        "file": file_name,
        "target": target,
        "task_type": task_type,
        "seed": seed,
        "missing_rate": MISSING_RATE,
        "n_eligible": len(eligible_idx),
        "n_masked": len(query_idx),
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
        "task_type",
        "seed",
        "missing_rate",
        "n_eligible",
        "n_masked",
    }

    metric_cols = [
        c for c in result_df.columns
        if c not in id_cols
        and pd.api.types.is_numeric_dtype(result_df[c])
    ]

    # Stage 1: average 3 seeds within each file.
    per_file = (
        result_df
        .groupby(
            ["file", "target", "task_type"],
            dropna=False,
        )[metric_cols]
        .mean()
        .reset_index()
    )

    per_file.to_csv(
        OUTPUT_DIR / "per_file_mean_metrics.csv",
        index=False,
    )

    # Stage 2: mean ± SD across files.
    overall = (
        per_file
        .groupby(
            ["target", "task_type"],
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
            f"No files matching {INPUT_PATTERNS!r} found in "
            f"{INPUT_DIR.expanduser().resolve()}."
        )

    results = []

    for file_path in files:
        print(f"\n=== {file_path.name} ===")

        df = read_csv_auto(file_path)
        targets = find_attribute_value_columns(df)

        unconfigured_targets = set(targets) - set(TARGET_TYPES)
        if unconfigured_targets:
            raise KeyError(
                f"Target types are not configured for {file_path.name}: "
                f"{sorted(unconfigured_targets)}"
            )

        print("Targets:", targets)

        for target in targets:
            for seed in SEEDS:
                print(f"  {target} | seed={seed}")

                result = run_one(
                    df,
                    file_path.name,
                    target,
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
