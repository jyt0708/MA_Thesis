"""
SAP RPT 1-OSS experiment: Missing E2O relation endpoint (object_id)

Task:
    Given an event and the remaining NON-TARGET-DERIVED context,
    predict which object should be connected to the event.

Input:
    A list of e2o_*.csv files with columns such as:
        event_id
        event_activity
        event_timestamp
        object_id
        object_type
        e2o_qualifier
        event_relation_count
        object_event_count
        object_event_position
        previous_object_activity
        next_object_activity

Experiment:
    - 10% missingness
    - seeds [42, 43, 44]
    - target = object_id
    - SAP_RPT_OSS_Classifier
    - classification metrics:
        Accuracy
        Macro / weighted Precision, Recall, F1
        Hits@3
        Hits@5

Important methodological choice
-------------------------------
STRICT_MISSING_RELATION = True (recommended)

In strict mode, the following are removed because they are either:
    (a) direct leakage, or
    (b) derived from the target object / relation itself.

Dropped:
    event_id
        E.g: "weigh_cr2" literally contains target object "cr2".

    object_type
        The endpoint's type is not necessarily known when the full E2O relation
        is missing. Keeping it is useful only for an endpoint-completion variant.

    e2o_qualifier
        The qualifier is part of the E2O relation. If the whole relation is
        missing, the qualifier should also be missing.

    object_event_count
    object_event_position
    previous_object_activity
    next_object_activity
        These were calculated after knowing object_id, so using them to predict
        object_id is circular / target-derived leakage.

Kept in strict mode:
    event_activity
    event_timestamp
    event_relation_count
        For query rows, event_relation_count is reduced by 1 to represent the
        remaining visible E2O relations after one relation is removed.

If STRICT_MISSING_RELATION = False:
    object_type, e2o_qualifier and object lifecycle context are retained.
    This is better described as "E2O endpoint completion" rather than repair of
    a completely missing E2O relation.

Class-support constraint
------------------------
SAP RPT can only predict object IDs that are present in the context labels.
Therefore, rows whose object_id occurs only once are NOT eligible for masking.
The sampler also guarantees at least one occurrence of every masked object_id
remains in the context.

"""

from pathlib import Path
from collections import Counter
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

FILES = []

# Can also be used instead of FILES.
# INPUT_GLOB = "Manufacturing/manufact_tabular/e2o_level*.csv"
INPUT_GLOB = "container_logistics_tabular/e2o_level*.csv"

# OUTPUT_DIR = Path("Manufacturing/results_e2o")
OUTPUT_DIR = Path("logistics_old_results/results_e2o")

TARGET = "object_id"

MISSING_RATE = 0.10
SEEDS = [42, 43, 44]

MAX_CONTEXT_SIZE = 8192
BAGGING = 8

# Recommended main experiment:
STRICT_MISSING_RELATION = True

# Always remove event_id because in this logistics dataset it often directly
# contains the target object ID, e.g. weigh_cr2, reg_co1, load_truck_hu4.
ALWAYS_DROP = ["event_id"]


# ============================================================
# IO
# ============================================================

def read_csv_auto(path):
    """Read comma-, semicolon-, or tab-delimited CSV automatically."""
    return pd.read_csv(path, sep=None, engine="python")


def discover_files():
    paths = [Path(p) for p in FILES]

    if INPUT_GLOB:
        paths.extend(Path().glob(INPUT_GLOB))

    unique = []
    seen = set()

    for p in paths:
        p = p.resolve()

        if p.exists() and p not in seen:
            unique.append(p)
            seen.add(p)

    return unique


# ============================================================
# Leakage-safe feature construction
# ============================================================

def prepare_features(df):
    """
    Build the feature table used to predict object_id.
    """
    drop_cols = [TARGET, *ALWAYS_DROP]

    if STRICT_MISSING_RELATION:
        drop_cols.extend([
            "object_type",
            "e2o_qualifier",
            "object_event_count",
            "object_event_position",
            "previous_object_activity",
            "next_object_activity",
        ])

    X = df.drop(
        columns=drop_cols,
        errors="ignore",
    ).copy()

    return X


# ============================================================
# Constrained random masking
# ============================================================

def choose_query_indices(df, seed):
    """
    Select ~10% of E2O rows for masking, while guaranteeing that every
    masked object_id still occurs at least once in the remaining context.

    Objects appearing only once are ineligible because SAP RPT cannot predict
    a class it never sees in its context labels.
    """

    y = df[TARGET]

    valid = y.notna()
    counts = y[valid].value_counts()

    # Only rows from object IDs with at least two occurrences are maskable.
    eligible_mask = valid & y.map(counts).fillna(0).ge(2)
    eligible_idx = df.index[eligible_mask].to_numpy()

    if len(eligible_idx) == 0:
        return np.asarray([], dtype=int), 0

    desired_n = max(
        1,
        int(round(len(eligible_idx) * MISSING_RATE)),
    )

    rng = np.random.default_rng(seed)
    shuffled = rng.permutation(eligible_idx)

    # Track how many instances of each object are still available for context.
    remaining_counts = Counter(y[valid].astype(str).tolist())

    selected = []

    for idx in shuffled:
        cls = str(y.loc[idx])

        # Never remove the final context instance of an object ID.
        if remaining_counts[cls] <= 1:
            continue

        selected.append(idx)
        remaining_counts[cls] -= 1

        if len(selected) >= desired_n:
            break

    return np.asarray(selected, dtype=int), len(eligible_idx)


# ============================================================
# Metrics
# ============================================================

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


def hits_at_k(y_true, probabilities, classes, k):
    """
    Fraction of query rows where the true object ID is among the top-k
    SAP RPT class probabilities.
    """
    y_true = np.asarray(y_true, dtype=str)
    classes = np.asarray(classes, dtype=str)

    k = min(k, probabilities.shape[1])

    topk_idx = np.argpartition(
        probabilities,
        kth=probabilities.shape[1] - k,
        axis=1,
    )[:, -k:]

    topk_classes = classes[topk_idx]

    return float(
        np.mean([
            truth in candidates
            for truth, candidates in zip(y_true, topk_classes)
        ])
    )


# ============================================================
# One run
# ============================================================

def run_one(df, file_name, seed):
    if TARGET not in df.columns:
        warnings.warn(
            f"{file_name}: target column {TARGET!r} not found; skipped."
        )
        return None

    query_idx, n_eligible = choose_query_indices(
        df,
        seed,
    )

    if len(query_idx) == 0:
        warnings.warn(
            f"{file_name} / seed {seed}: no eligible rows to mask; skipped."
        )
        return None

    all_valid_idx = df.index[
        df[TARGET].notna()
    ].to_numpy()

    context_idx = np.setdiff1d(
        all_valid_idx,
        query_idx,
    )

    y_context = (
        df.loc[context_idx, TARGET]
        .astype(str)
    )

    y_true = (
        df.loc[query_idx, TARGET]
        .astype(str)
    )

    # Defensive check: all query classes must be visible in context.
    unseen = set(y_true) - set(y_context)

    if unseen:
        raise RuntimeError(
            "Masking bug: query contains object IDs absent from context: "
            f"{sorted(unseen)[:10]}"
        )

    X = prepare_features(df)

    X_context = X.loc[
        context_idx
    ].copy()

    X_query = X.loc[
        query_idx
    ].copy()

    # In a strict missing-relation setting, one E2O relation is removed.
    # Therefore the number of remaining visible relations for the event
    # should be one smaller than in the complete log.
    if (
        STRICT_MISSING_RELATION
        and "event_relation_count" in X_query.columns
    ):
        rel_count = pd.to_numeric(
            X_query["event_relation_count"],
            errors="coerce",
        )

        X_query["event_relation_count"] = (
            rel_count - 1
        ).clip(lower=0)

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

    # Current SAP RPT 1-OSS exposes predict_proba().
    probabilities = model.predict_proba(
        X_query
    )

    metrics["Hits_at_3"] = hits_at_k(
        y_true,
        probabilities,
        model.classes_,
        k=3,
    )

    metrics["Hits_at_5"] = hits_at_k(
        y_true,
        probabilities,
        model.classes_,
        k=5,
    )

    return {
        "file": file_name,
        "seed": seed,
        "missing_rate": MISSING_RATE,
        "strict_missing_relation": STRICT_MISSING_RELATION,
        "n_total_relations": int(df[TARGET].notna().sum()),
        "n_eligible_relations": int(n_eligible),
        "n_masked": int(len(query_idx)),
        "n_context": int(len(context_idx)),
        "n_object_id_classes_context": int(y_context.nunique()),
        **metrics,
    }


# ============================================================
# Aggregation
# ============================================================

def save_results(results):
    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    result_df = pd.DataFrame(results)

    # --------------------------------------------------------
    # 1. Individual seed runs
    # --------------------------------------------------------

    result_df.to_csv(
        OUTPUT_DIR / "per_run_metrics.csv",
        index=False,
    )

    metadata_cols = {
        "file",
        "seed",
        "missing_rate",
        "strict_missing_relation",
        "n_total_relations",
        "n_eligible_relations",
        "n_masked",
        "n_context",
        "n_object_id_classes_context",
    }

    metric_cols = [
        c
        for c in result_df.columns
        if c not in metadata_cols
        and pd.api.types.is_numeric_dtype(
            result_df[c]
        )
    ]

    # --------------------------------------------------------
    # 2. Average the 3 seeds within each OCEL file
    # --------------------------------------------------------

    per_file = (
        result_df
        .groupby(
            ["file"],
            dropna=False,
        )[metric_cols]
        .mean()
        .reset_index()
    )

    per_file.to_csv(
        OUTPUT_DIR / "per_file_mean_metrics.csv",
        index=False,
    )

    # --------------------------------------------------------
    # 3. Mean ± SD across OCEL files
    # --------------------------------------------------------

    overall = (
        per_file[metric_cols]
        .agg(["mean", "std"])
        .T
        .reset_index()
        .rename(
            columns={
                "index": "metric",
                "mean": "mean_across_files",
                "std": "std_across_files",
            }
        )
    )

    overall.to_csv(
        OUTPUT_DIR / "overall_mean_std_across_files.csv",
        index=False,
    )

    return result_df, per_file, overall


# ============================================================
# Main
# ============================================================

def main():
    files = discover_files()

    if not files:
        raise FileNotFoundError(
            "No E2O files found."
        )

    results = []

    for file_path in files:
        print(
            f"\n=== {file_path.name} ==="
        )

        df = read_csv_auto(
            file_path
        )

        print(
            f"Rows: {len(df):,}"
        )

        print(
            f"Unique object IDs: "
            f"{df[TARGET].nunique(dropna=True):,}"
        )

        for seed in SEEDS:
            print(
                f"  object_id | seed={seed}"
            )

            result = run_one(
                df=df,
                file_name=file_path.name,
                seed=seed,
            )

            if result is not None:
                results.append(result)

    if not results:
        print(
            "\nNo valid experiments were run."
        )
        return

    result_df, per_file, overall = (
        save_results(results)
    )

    print(
        "\n=== Per-run metrics ==="
    )
    print(
        result_df.to_string(
            index=False
        )
    )

    print(
        "\n=== Per-file mean over 3 seeds ==="
    )
    print(
        per_file.to_string(
            index=False
        )
    )

    print(
        "\n=== Overall mean ± SD across files ==="
    )
    print(
        overall.to_string(
            index=False
        )
    )

    print(
        f"\nSaved to: "
        f"{OUTPUT_DIR.resolve()}"
    )


if __name__ == "__main__":
    main()
