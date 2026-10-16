"""Reconstruct hidden object records from their E2O event context with SAP RPT.

This is an object-node reconstruction experiment, not E2O endpoint prediction.
For each seed, approximately 10% of complete object records are hidden. The
input features for those objects are rebuilt exclusively from their surviving
rows in the matching E2O table. The hidden ``object_level`` rows supply only
the ground-truth object type and attributes used for evaluation.

``object_id`` is used to join and group the two tables, but it is never passed
to SAP RPT. Consequently, prefixes or naming conventions in object IDs cannot
leak the object type.

By default the script evaluates object type, attribute-presence flags, and
observed attribute values. Each target requires a separate SAP RPT fit. Set
the PREDICT_* switches or ATTRIBUTE_VALUE_TARGETS for a smaller experiment.
Targets without sufficient observed values or training classes are listed in
``skipped_targets.csv`` with their reasons; they do not contribute metrics.
"""

from __future__ import annotations

from collections import Counter
from pathlib import Path
import warnings

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    mean_absolute_error,
    mean_squared_error,
    precision_recall_fscore_support,
    r2_score,
)

from sap_rpt_oss import SAP_RPT_OSS_Classifier, SAP_RPT_OSS_Regressor


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

# INPUT_DIR = Path("Manufacturing/manufact_tabular")
INPUT_DIR = Path("container_logistics_tabular")
OBJECT_PATTERNS = ("object_level.csv", "object_level_*.csv")
# OUTPUT_DIR = Path("Manufacturing/results_missing_object_manufact")
OUTPUT_DIR = Path("results_missing_object_logistics")

MISSING_RATE = 0.10
SEEDS = [42, 43, 44]
MAX_CONTEXT_SIZE = 2048 # 8192
BAGGING = 1 # 8

PREDICT_OBJECT_TYPE = True
PREDICT_ATTRIBUTE_PRESENCE = True
PREDICT_ATTRIBUTE_VALUES = True

# Empty means auto-discover all attribute-value columns. Otherwise, only the
# listed columns are evaluated. Presence flags are controlled separately.
ATTRIBUTE_VALUE_TARGETS: list[str] = []

# Numeric targets with few distinct values are normally codes/categories.
# Override ambiguous columns explicitly with "classification" or "regression".
NUMERIC_CLASSIFICATION_MAX_UNIQUE = 20
TASK_TYPE_OVERRIDES: dict[str, str] = {}

# Date regression targets and saved predictions use fractional days since this
# fixed UTC reference. Their MAE and RMSE are therefore measured in days.
DATE_REFERENCE = pd.Timestamp("1970-01-01", tz="UTC")

MAX_SEQUENCE_VALUES = 100
OBJECT_ID = "object_id"
OBJECT_TYPE = "object_type"


# ---------------------------------------------------------------------------
# Input and pairing
# ---------------------------------------------------------------------------

def read_csv_auto(path: Path) -> pd.DataFrame:
    """Read a comma-, semicolon-, or tab-delimited CSV."""
    return pd.read_csv(path, sep=None, engine="python", encoding="utf-8-sig")


def discover_pairs() -> list[tuple[Path, Path]]:
    """Pair object_level[_n].csv with e2o_level[_n].csv."""
    input_dir = INPUT_DIR.expanduser().resolve()
    if not input_dir.is_dir():
        return []

    object_files = {
        path.resolve()
        for pattern in OBJECT_PATTERNS
        for path in input_dir.glob(pattern)
        if path.is_file()
    }
    pairs: list[tuple[Path, Path]] = []
    for object_path in sorted(object_files, key=lambda path: path.name):
        e2o_name = object_path.name.replace("object_level", "e2o_level", 1)
        e2o_path = object_path.with_name(e2o_name)
        if e2o_path.is_file():
            pairs.append((object_path, e2o_path.resolve()))
        else:
            warnings.warn(f"No matching E2O table for {object_path.name}; skipped.")
    return pairs


# ---------------------------------------------------------------------------
# E2O-only behavioral features
# ---------------------------------------------------------------------------

def _joined_values(values: pd.Series, unique: bool = False) -> str | float:
    cleaned = [str(value) for value in values.dropna() if str(value).strip()]
    if unique:
        cleaned = list(dict.fromkeys(cleaned))
    if not cleaned:
        return np.nan
    return ">".join(cleaned[:MAX_SEQUENCE_VALUES])


def build_e2o_context(e2o: pd.DataFrame) -> pd.DataFrame:
    """Build one behavioral-context row per object using only the E2O table."""
    required = {OBJECT_ID, "event_activity"}
    missing = required - set(e2o.columns)
    if missing:
        raise KeyError(f"E2O table is missing required columns: {sorted(missing)}")

    work = e2o.loc[e2o[OBJECT_ID].notna()].copy()
    work[OBJECT_ID] = work[OBJECT_ID].astype(str)
    if "event_timestamp" in work.columns:
        work["_parsed_timestamp"] = pd.to_datetime(
            work["event_timestamp"], errors="coerce", utc=True
        )
        work = work.sort_values(
            [OBJECT_ID, "_parsed_timestamp"], kind="stable", na_position="last"
        )
    else:
        work["_parsed_timestamp"] = pd.NaT

    grouped = work.groupby(OBJECT_ID, sort=False, dropna=False)
    context = pd.DataFrame(index=grouped.size().index)
    context["ctx_relation_rows"] = grouped.size().astype(float)
    if "event_id" in work.columns:
        # Raw event IDs are excluded because they can contain object prefixes.
        context["ctx_unique_events"] = grouped["event_id"].nunique().astype(float)

    context["ctx_unique_activities"] = grouped["event_activity"].nunique().astype(float)
    context["ctx_first_activity"] = grouped["event_activity"].first()
    context["ctx_last_activity"] = grouped["event_activity"].last()
    context["ctx_activity_sequence"] = grouped["event_activity"].agg(_joined_values)
    context["ctx_activity_set"] = grouped["event_activity"].agg(
        lambda values: _joined_values(values, unique=True)
    )

    first_time = grouped["_parsed_timestamp"].min()
    last_time = grouped["_parsed_timestamp"].max()
    context["ctx_first_event_time"] = first_time.astype(str)
    context["ctx_last_event_time"] = last_time.astype(str)
    context["ctx_duration_seconds"] = (last_time - first_time).dt.total_seconds()

    if "event_relation_count" in work.columns:
        relation_count = pd.to_numeric(work["event_relation_count"], errors="coerce")
        relation_groups = relation_count.groupby(work[OBJECT_ID], sort=False)
        context["ctx_event_relation_count_min"] = relation_groups.min()
        context["ctx_event_relation_count_mean"] = relation_groups.mean()
        context["ctx_event_relation_count_max"] = relation_groups.max()

    # These describe surviving events/incident edges. No object-side columns
    # such as object_type or lifecycle statistics are included.
    text_columns = [
        column
        for column in work.columns
        if column == "e2o_qualifier" or column.startswith("event_attr__")
    ]
    for column in text_columns:
        context[f"ctx_values__{column}"] = grouped[column].agg(
            lambda values: _joined_values(values, unique=True)
        )

    context = context.replace({"NaT": np.nan}).dropna(axis=1, how="all")
    context.index = context.index.astype(str)
    context.index.name = OBJECT_ID
    return context


def prepare_object_truth(objects: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    """Return one ground-truth record per object ID."""
    if OBJECT_ID not in objects.columns:
        raise KeyError(f"Object table is missing required column {OBJECT_ID!r}.")
    truth = objects.loc[objects[OBJECT_ID].notna()].copy()
    truth[OBJECT_ID] = truth[OBJECT_ID].astype(str)
    duplicate_rows = int(truth.duplicated(OBJECT_ID, keep="first").sum())
    if duplicate_rows:
        warnings.warn(
            f"Object table contains {duplicate_rows} duplicate object rows; "
            "the first record for each object_id is used."
        )
    truth = truth.drop_duplicates(OBJECT_ID, keep="first").set_index(OBJECT_ID)
    return truth, duplicate_rows


# ---------------------------------------------------------------------------
# Hidden-object sampling and target definitions
# ---------------------------------------------------------------------------

def choose_hidden_objects(truth: pd.DataFrame, seed: int) -> tuple[np.ndarray, int]:
    """Hide ~10% of objects while retaining every queried type in context."""
    if OBJECT_TYPE not in truth.columns:
        raise KeyError(f"Object table is missing required column {OBJECT_TYPE!r}.")
    y = truth[OBJECT_TYPE]
    valid = y.notna()
    counts = y.loc[valid].astype(str).value_counts()
    eligible = valid & y.astype(str).map(counts).fillna(0).ge(2)
    eligible_ids = truth.index[eligible].to_numpy(dtype=str)
    if len(eligible_ids) == 0:
        return np.asarray([], dtype=str), 0

    desired = max(1, int(round(len(eligible_ids) * MISSING_RATE)))
    shuffled = np.random.default_rng(seed).permutation(eligible_ids)
    remaining = Counter(y.loc[valid].astype(str).tolist())
    selected: list[str] = []
    for object_id in shuffled:
        object_type = str(y.loc[object_id])
        if remaining[object_type] <= 1:
            continue
        selected.append(str(object_id))
        remaining[object_type] -= 1
        if len(selected) == desired:
            break
    return np.asarray(selected, dtype=str), len(eligible_ids)


def discover_targets(truth: pd.DataFrame) -> list[str]:
    targets: list[str] = []
    if PREDICT_OBJECT_TYPE and OBJECT_TYPE in truth.columns:
        targets.append(OBJECT_TYPE)

    attributes = [
        column for column in truth.columns if str(column).startswith("obj_attr__")
    ]
    if PREDICT_ATTRIBUTE_PRESENCE:
        targets.extend(column for column in attributes if column.endswith("__exists"))
    if PREDICT_ATTRIBUTE_VALUES:
        values = [column for column in attributes if not column.endswith("__exists")]
        if ATTRIBUTE_VALUE_TARGETS:
            requested = set(ATTRIBUTE_VALUE_TARGETS)
            values = [column for column in values if column in requested]
            absent = requested - set(values)
            if absent:
                warnings.warn(f"Configured attribute targets not found: {sorted(absent)}")
        targets.extend(values)
    return list(dict.fromkeys(targets))


def target_task_type(values: pd.Series, target: str) -> str:
    override = TASK_TYPE_OVERRIDES.get(target)
    if override:
        if override not in {"classification", "regression"}:
            raise ValueError(f"Invalid task override for {target!r}: {override!r}")
        return override
    if "date" in target.lower() and not target.endswith("__exists"):
        return "regression"
    if target == OBJECT_TYPE or target.endswith("__exists"):
        return "classification"

    observed = values.dropna()
    numeric = pd.to_numeric(observed, errors="coerce")
    if not observed.empty and numeric.notna().all():
        if observed.nunique() <= NUMERIC_CLASSIFICATION_MAX_UNIQUE:
            return "classification"
        return "regression"
    return "classification"


def regression_target_values(values: pd.Series, target: str) -> pd.Series:
    """Encode dates as fractional days; preserve ordinary numeric targets."""
    if "date" in target.lower() and not target.endswith("__exists"):
        dates = pd.to_datetime(values, errors="coerce", utc=True, format="mixed")
        return (dates - DATE_REFERENCE).dt.total_seconds() / 86400.0
    return pd.to_numeric(values, errors="coerce")


def eligible_target_values(truth: pd.DataFrame, target: str) -> pd.Series:
    """Select observed ground truth for one target."""
    values = truth[target]
    eligible = values.notna()
    if target.startswith("obj_attr__") and not target.endswith("__exists"):
        exists_column = f"{target}__exists"
        if exists_column in truth.columns:
            eligible &= pd.to_numeric(truth[exists_column], errors="coerce").eq(1)
    return values.loc[eligible]


# ---------------------------------------------------------------------------
# Models and metrics
# ---------------------------------------------------------------------------

def classification_metrics(y_true: pd.Series, y_pred: np.ndarray) -> dict[str, float]:
    truth = pd.Series(y_true).astype(str)
    prediction = pd.Series(y_pred).astype(str)
    p_macro, r_macro, f1_macro, _ = precision_recall_fscore_support(
        truth, prediction, average="macro", zero_division=0
    )
    p_weighted, r_weighted, f1_weighted, _ = precision_recall_fscore_support(
        truth, prediction, average="weighted", zero_division=0
    )
    return {
        "Accuracy": accuracy_score(truth, prediction),
        "Precision_macro": p_macro,
        "Recall_macro": r_macro,
        "F1_macro": f1_macro,
        "Precision_weighted": p_weighted,
        "Recall_weighted": r_weighted,
        "F1_weighted": f1_weighted,
    }


def regression_metrics(y_true: pd.Series, y_pred: np.ndarray) -> dict[str, float]:
    truth = np.asarray(y_true, dtype=float)
    prediction = np.asarray(y_pred, dtype=float)
    nonzero = np.abs(truth) > 1e-12
    return {
        "RMSE": float(np.sqrt(mean_squared_error(truth, prediction))),
        "MAE": float(mean_absolute_error(truth, prediction)),
        "R2": float(r2_score(truth, prediction)) if len(truth) >= 2 else np.nan,
        "MAPE_percent": (
            float(np.mean(np.abs((truth[nonzero] - prediction[nonzero]) / truth[nonzero])) * 100)
            if nonzero.any() else np.nan
        ),
    }


def _query_for_single_row(X: pd.DataFrame) -> tuple[pd.DataFrame, bool]:
    if len(X) != 1:
        return X, False
    return pd.concat([X, X], ignore_index=True), True


def predict_safe(model: object, X: pd.DataFrame) -> np.ndarray:
    query, duplicated = _query_for_single_row(X)
    prediction = np.asarray(model.predict(query))
    return prediction[:1] if duplicated else prediction


def predict_proba_safe(model: object, X: pd.DataFrame) -> np.ndarray:
    query, duplicated = _query_for_single_row(X)
    probabilities = np.asarray(model.predict_proba(query))
    return probabilities[:1] if duplicated else probabilities


def top_k_classes(probabilities: np.ndarray, classes: np.ndarray, k: int) -> np.ndarray:
    k = min(k, probabilities.shape[1])
    indices = np.argpartition(probabilities, -k, axis=1)[:, -k:]
    rows = np.arange(probabilities.shape[0])[:, None]
    order = np.argsort(probabilities[rows, indices], axis=1)[:, ::-1]
    return np.asarray(classes, dtype=str)[indices[rows, order]]


def run_target(
    context_features: pd.DataFrame,
    truth: pd.DataFrame,
    hidden_ids: np.ndarray,
    file_name: str,
    e2o_file_name: str,
    target: str,
    seed: int,
    n_maskable_objects: int,
    skipped_targets: list[dict[str, object]] | None = None,
) -> tuple[dict[str, object], pd.DataFrame] | None:
    values = eligible_target_values(truth, target)
    hidden = set(hidden_ids)
    query_ids = [object_id for object_id in values.index if object_id in hidden]
    context_ids = [object_id for object_id in values.index if object_id not in hidden]

    def skip(reason: str) -> None:
        if skipped_targets is not None:
            skipped_targets.append({
                "object_file": file_name,
                "e2o_file": e2o_file_name,
                "seed": seed,
                "target": target,
                "reason": reason,
                "n_eligible_values": len(values),
                "n_target_context": len(context_ids),
                "n_target_query": len(query_ids),
                "n_context_unique_values": values.loc[context_ids].nunique(),
            })
        warnings.warn(f"{file_name} / {target} / seed {seed}: {reason}; skipped.")

    if not query_ids or len(context_ids) < 2:
        if values.empty:
            skip("no observed eligible values among matched objects")
        elif not query_ids:
            skip("no observed eligible values among hidden objects")
        else:
            skip("fewer than 2 eligible context rows")
        return None

    task_type = target_task_type(values, target)
    X_context = context_features.loc[context_ids].copy()
    X_query = context_features.loc[query_ids].copy()
    if OBJECT_ID in X_context.columns or OBJECT_ID in X_query.columns:
        raise RuntimeError("object_id leakage: identifier found in model features.")

    y_context_raw = values.loc[context_ids]
    y_true_raw = values.loc[query_ids]
    if task_type == "classification":
        y_context = y_context_raw.astype(str)
        y_true = y_true_raw.astype(str)
        if y_context.nunique() < 2:
            skip("context has fewer than 2 classes")
            return None
        model = SAP_RPT_OSS_Classifier(
            max_context_size=MAX_CONTEXT_SIZE, bagging=BAGGING
        )
    else:
        y_context = regression_target_values(y_context_raw, target)
        y_true = regression_target_values(y_true_raw, target)
        valid_context = y_context.notna()
        valid_query = y_true.notna()
        X_context, y_context = X_context.loc[valid_context], y_context.loc[valid_context]
        X_query, y_true = X_query.loc[valid_query], y_true.loc[valid_query]
        query_ids = list(y_true.index)
        if len(y_context) < 2 or len(y_true) == 0:
            skip("insufficient numeric/date rows after conversion")
            return None
        model = SAP_RPT_OSS_Regressor(
            max_context_size=MAX_CONTEXT_SIZE, bagging=BAGGING
        )

    model.fit(X_context, y_context)
    y_pred = predict_safe(model, X_query)
    prediction_rows = pd.DataFrame({
        "object_file": file_name,
        "e2o_file": e2o_file_name,
        "seed": seed,
        "object_id_for_evaluation_only": query_ids,
        "target": target,
        "task_type": task_type,
        "true_value": np.asarray(y_true).astype(str),
        "predicted_value": np.asarray(y_pred).astype(str),
    })

    extra: dict[str, float | int] = {}
    if task_type == "classification":
        metrics = classification_metrics(y_true, y_pred)
        unseen = ~y_true.isin(set(y_context))
        extra["n_query_unseen_classes"] = int(unseen.sum())
        extra["query_unseen_class_rate"] = float(unseen.mean())
        probabilities = predict_proba_safe(model, X_query)
        top5 = top_k_classes(probabilities, model.classes_, 5)
        truth_array = y_true.to_numpy(dtype=str)
        metrics["Hits_at_3"] = float(np.mean([
            value in candidates for value, candidates in zip(truth_array, top5[:, :3])
        ]))
        metrics["Hits_at_5"] = float(np.mean([
            value in candidates for value, candidates in zip(truth_array, top5)
        ]))
        for rank in range(top5.shape[1]):
            prediction_rows[f"top_{rank + 1}_value"] = top5[:, rank]
    else:
        metrics = regression_metrics(y_true, y_pred)
        extra["n_query_unseen_classes"] = 0
        extra["query_unseen_class_rate"] = 0.0

    result = {
        "object_file": file_name,
        "e2o_file": e2o_file_name,
        "seed": seed,
        "missing_rate": MISSING_RATE,
        "target": target,
        "task_type": task_type,
        "n_common_objects": int(len(truth)),
        "n_maskable_objects": int(n_maskable_objects),
        "n_hidden_complete_objects": int(len(hidden_ids)),
        "n_target_context": int(len(y_context)),
        "n_target_query": int(len(y_true)),
        "n_context_features": int(context_features.shape[1]),
        **extra,
        **metrics,
    }
    return result, prediction_rows


# ---------------------------------------------------------------------------
# Output and main
# ---------------------------------------------------------------------------

def save_results(
    results: list[dict[str, object]], predictions: list[pd.DataFrame]
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    per_run = pd.DataFrame(results)
    per_run.to_csv(OUTPUT_DIR / "per_run_metrics.csv", index=False)
    pd.concat(predictions, ignore_index=True).to_csv(
        OUTPUT_DIR / "per_query_predictions.csv", index=False
    )

    metadata = {
        "object_file", "e2o_file", "seed", "missing_rate", "target", "task_type",
        "n_common_objects", "n_maskable_objects", "n_hidden_complete_objects",
        "n_target_context", "n_target_query", "n_context_features",
        "n_query_unseen_classes", "query_unseen_class_rate",
    }
    metric_columns = [
        column for column in per_run.columns
        if column not in metadata and pd.api.types.is_numeric_dtype(per_run[column])
    ]
    per_file = (
        per_run.groupby(["object_file", "e2o_file", "target", "task_type"], dropna=False)[metric_columns]
        .mean().reset_index()
    )
    per_file.to_csv(OUTPUT_DIR / "per_file_mean_metrics.csv", index=False)
    overall = (
        per_file.groupby(["target", "task_type"], dropna=False)[metric_columns]
        .agg(["mean", "std"])
    )
    overall.to_csv(OUTPUT_DIR / "overall_mean_std_across_files.csv")
    return per_run, per_file, overall


def main() -> None:
    pairs = discover_pairs()
    if not pairs:
        raise FileNotFoundError(
            f"No matching object_level/e2o_level pairs found in {INPUT_DIR.resolve()}."
        )

    results: list[dict[str, object]] = []
    predictions: list[pd.DataFrame] = []
    skipped_targets: list[dict[str, object]] = []
    for object_path, e2o_path in pairs:
        print(f"\n=== {object_path.name} + {e2o_path.name} ===")
        object_table = read_csv_auto(object_path)
        e2o_table = read_csv_auto(e2o_path)
        truth, _ = prepare_object_truth(object_table)
        e2o_context = build_e2o_context(e2o_table)

        attribute_values = [
            column for column in truth.columns
            if column.startswith("obj_attr__") and not column.endswith("__exists")
        ]
        n_with_values = int(truth[attribute_values].notna().any(axis=1).sum())

        common_ids = truth.index.intersection(e2o_context.index, sort=False)
        truth = truth.loc[common_ids].copy()
        context_features = e2o_context.loc[common_ids].copy()
        if len(common_ids) < 3:
            warnings.warn(f"{object_path.name}: fewer than 3 matched objects; skipped.")
            continue

        targets = discover_targets(truth)
        presence_targets = [target for target in targets if target.endswith("__exists")]
        value_targets = [target for target in targets if target in attribute_values]
        n_matched_with_values = int(truth[attribute_values].notna().any(axis=1).sum())
        print(
            f"Matched objects: {len(common_ids):,}; E2O context features: "
            f"{context_features.shape[1]:,}; targets: {len(targets):,}"
        )
        print(
            f"  Attribute targets: {len(presence_targets)} presence, "
            f"{len(value_targets)} values; objects with observed attribute values: "
            f"{n_with_values:,} before E2O matching, {n_matched_with_values:,} after"
        )
        if (PREDICT_ATTRIBUTE_PRESENCE or PREDICT_ATTRIBUTE_VALUES) and not attribute_values:
            print("  No obj_attr__ value columns found in the object table.")
        elif PREDICT_ATTRIBUTE_VALUES and not n_matched_with_values:
            print("  Attribute values cannot be evaluated: matched objects have no observed values.")
        for seed in SEEDS:
            hidden_ids, n_maskable = choose_hidden_objects(truth, seed)
            print(f"  seed={seed}: hiding {len(hidden_ids):,} complete object records")
            skipped_before = len(skipped_targets)
            for target in targets:
                run = run_target(
                    context_features=context_features,
                    truth=truth,
                    hidden_ids=hidden_ids,
                    file_name=object_path.name,
                    e2o_file_name=e2o_path.name,
                    target=target,
                    seed=seed,
                    n_maskable_objects=n_maskable,
                    skipped_targets=skipped_targets,
                )
                if run is not None:
                    result, query_predictions = run
                    results.append(result)
                    predictions.append(query_predictions)
            seed_skips = skipped_targets[skipped_before:]
            print(f"    Evaluated {len(targets) - len(seed_skips)}; skipped {len(seed_skips)} targets")
            for reason, count in Counter(row["reason"] for row in seed_skips).items():
                print(f"      {count}: {reason}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    skip_path = OUTPUT_DIR / "skipped_targets.csv"
    pd.DataFrame(skipped_targets, columns=[
        "object_file", "e2o_file", "seed", "target", "reason",
        "n_eligible_values", "n_target_context", "n_target_query", "n_context_unique_values",
    ]).to_csv(skip_path, index=False)
    print(f"\nSkipped-target report: {skip_path.resolve()}")

    if not results:
        print("No valid experiments were run.")
        return

    per_run, per_file, overall = save_results(results, predictions)
    print("\n=== Per-run metrics ===")
    print(per_run.to_string(index=False))
    print("\n=== Per-file mean over seeds ===")
    print(per_file.to_string(index=False))
    print("\n=== Overall mean +/- SD across files ===")
    print(overall)
    print(f"\nSaved to: {OUTPUT_DIR.resolve()}")


if __name__ == "__main__":
    main()
