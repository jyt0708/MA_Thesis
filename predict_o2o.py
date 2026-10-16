"""
SAP RPT 1-OSS experiment: Missing O2O target endpoint

Task
----
Given a source object and remaining NON-TARGET-DERIVED context,
predict which target object should be connected to it.

Input files:
    container_logistics_tabular/e2o_level_*.csv
    container_logistics_tabular/e2o_level.csv

Expected columns:
    source_object_id
    source_object_type
    target_object_id
    target_object_type
    o2o_qualifier
    source_event_count
    target_event_count
    shared_event_count
    source_o2o_degree
    target_o2o_degree

Experiment:
    - target = target_object_id
    - 10% missingness
    - seeds = [42, 43, 44]
    - SAP_RPT_OSS_Classifier
    - metrics:
        Accuracy
        Macro Precision / Recall / F1
        Weighted Precision / Recall / F1
        Hits@3
        Hits@5

Leakage policy
--------------
The following columns are hidden because they reveal or are derived from the
target object / O2O relation:

    target_object_type
        Direct metadata about the object being predicted.

    o2o_qualifier
        Part of the missing O2O relation itself.

    target_event_count
        Computed using the true target object.

    target_o2o_degree
        Computed using the true target object.

    shared_event_count
        Requires knowing both source and target objects.

The following are kept:

    source_object_id
    source_object_type
    source_event_count
    source_o2o_degree

Since one O2O relation is treated as missing, source_o2o_degree is reduced by 1
for query rows.

Class-support constraint
------------------------
SAP RPT can only predict a target_object_id if that class is present in the
context labels. Therefore:
    - target IDs occurring only once are not eligible for masking
    - for every masked target ID, at least one occurrence remains in context

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

FILES = [
    # "Manufacturing/manufact_tabular/o2o_level.csv" 
    "container_logistics_tabular/o2o_level.csv",
]

INPUT_GLOB = "container_logistics_tabular/o2o_level_*.csv" # "Manufacturing/manufact_tabular/o2o_level_*.csv"

OUTPUT_DIR = Path("results_o2o") # Path("Manufacturing/results_o2o")

TARGET = "target_object_id"

MISSING_RATE = 0.10
SEEDS = [42, 43, 44]


MAX_CONTEXT_SIZE = 8192
BAGGING = 8


# ============================================================
# IO
# ============================================================

def read_csv_auto(path):
    """Read comma-, semicolon-, or tab-delimited CSV automatically."""
    return pd.read_csv(
        path,
        sep=None,
        engine="python",
    )


def discover_files():
    paths = [Path(p) for p in FILES]

    if INPUT_GLOB:
        paths.extend(
            Path().glob(INPUT_GLOB)
        )
    print(f"paths: {paths}")

    result = []
    seen = set()

    for p in paths:
        p = p.resolve()

        if p.exists() and p not in seen:
            result.append(p)
            seen.add(p)

    return result


# ============================================================
# Leakage-safe feature construction
# ============================================================

def prepare_features(df):
    """
    Keep source-side information, remove target-side information.
    """

    drop_cols = [
        TARGET,

        # Explicit target metadata
        "target_object_type",

        # Relation metadata
        "o2o_qualifier",

        # Features derived from the true target object
        "target_event_count",
        "target_o2o_degree",
        "shared_event_count",
    ]

    X = df.drop(
        columns=drop_cols,
        errors="ignore",
    ).copy()

    return X


# ============================================================
# Constrained 10% masking
# ============================================================

def choose_query_indices(df, seed):
    """
    Randomly mask approximately 10% of eligible O2O rows.

    A row is eligible only if its target_object_id occurs at least twice,
    because at least one instance of that target class must remain in
    SAP RPT's context.
    """

    y = df[TARGET]

    valid = y.notna()

    counts = (
        y[valid]
        .astype(str)
        .value_counts()
    )

    eligible_mask = (
        valid
        & y.astype(str).map(counts).fillna(0).ge(2)
    )

    eligible_idx = df.index[
        eligible_mask
    ].to_numpy()

    if len(eligible_idx) == 0:
        return (
            np.asarray([], dtype=int),
            0,
        )

    desired_n = max(
        1,
        int(round(
            len(eligible_idx)
            * MISSING_RATE
        )),
    )

    rng = np.random.default_rng(seed)

    shuffled = rng.permutation(
        eligible_idx
    )

    # Number of context examples still available for each class.
    remaining_counts = Counter(
        y[valid].astype(str).tolist()
    )

    selected = []

    for idx in shuffled:

        cls = str(
            y.loc[idx]
        )

        # Do not remove the last remaining context example.
        if remaining_counts[cls] <= 1:
            continue

        selected.append(idx)

        remaining_counts[cls] -= 1

        if len(selected) >= desired_n:
            break

    return (
        np.asarray(selected, dtype=int),
        len(eligible_idx),
    )


# ============================================================
# Metrics
# ============================================================

def classification_metrics(
    y_true,
    y_pred,
):
    y_true = pd.Series(
        y_true
    ).astype(str)

    y_pred = pd.Series(
        y_pred
    ).astype(str)

    (
        p_macro,
        r_macro,
        f1_macro,
        _
    ) = precision_recall_fscore_support(
        y_true,
        y_pred,
        average="macro",
        zero_division=0,
    )

    (
        p_weighted,
        r_weighted,
        f1_weighted,
        _
    ) = precision_recall_fscore_support(
        y_true,
        y_pred,
        average="weighted",
        zero_division=0,
    )

    return {
        "Accuracy":
            accuracy_score(
                y_true,
                y_pred,
            ),

        "Precision_macro":
            p_macro,

        "Recall_macro":
            r_macro,

        "F1_macro":
            f1_macro,

        "Precision_weighted":
            p_weighted,

        "Recall_weighted":
            r_weighted,

        "F1_weighted":
            f1_weighted,
    }


def hits_at_k(
    y_true,
    probabilities,
    classes,
    k,
):
    """
    Fraction of query rows for which the true target object is among
    SAP RPT's top-k predicted classes.
    """

    y_true = np.asarray(
        y_true,
        dtype=str,
    )

    classes = np.asarray(
        classes,
        dtype=str,
    )

    k = min(
        k,
        probabilities.shape[1],
    )

    if k <= 0:
        return np.nan

    # Largest k class-probability indices
    topk_idx = np.argpartition(
        probabilities,
        kth=probabilities.shape[1] - k,
        axis=1,
    )[:, -k:]

    topk_classes = classes[
        topk_idx
    ]

    return float(
        np.mean([
            truth in candidates
            for truth, candidates
            in zip(
                y_true,
                topk_classes,
            )
        ])
    )


# ============================================================
# One run
# ============================================================

def run_one(
    df,
    file_name,
    seed,
):
    if TARGET not in df.columns:
        warnings.warn(
            f"{file_name}: "
            f"{TARGET!r} not found; skipped."
        )
        return None

    query_idx, n_eligible = (
        choose_query_indices(
            df,
            seed,
        )
    )

    if len(query_idx) == 0:
        warnings.warn(
            f"{file_name} / seed {seed}: "
            f"no eligible rows to mask; skipped."
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
        df.loc[
            context_idx,
            TARGET,
        ]
        .astype(str)
    )

    y_true = (
        df.loc[
            query_idx,
            TARGET,
        ]
        .astype(str)
    )

    # Defensive check:
    # every query class must still be represented in context.
    unseen = (
        set(y_true)
        - set(y_context)
    )

    if unseen:
        raise RuntimeError(
            "Masking bug: query contains "
            "target_object_id classes not "
            "present in context: "
            f"{sorted(unseen)[:10]}"
        )

    # --------------------------------------------------------
    # Leakage-safe X
    # --------------------------------------------------------

    X = prepare_features(df)

    X_context = X.loc[
        context_idx
    ].copy()

    X_query = X.loc[
        query_idx
    ].copy()

    # --------------------------------------------------------
    # Simulate removal of one O2O relation
    # --------------------------------------------------------

    # source_o2o_degree in the complete table counted the relation
    # that is now missing. Therefore query rows should see one less.
    if (
        "source_o2o_degree"
        in X_query.columns
    ):
        degree = pd.to_numeric(
            X_query[
                "source_o2o_degree"
            ],
            errors="coerce",
        )

        X_query[
            "source_o2o_degree"
        ] = (
            degree - 1
        ).clip(lower=0)

    # --------------------------------------------------------
    # SAP RPT
    # --------------------------------------------------------

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

    probabilities = (
        model.predict_proba(
            X_query
        )
    )

    metrics = classification_metrics(
        y_true,
        y_pred,
    )

    metrics[
        "Hits_at_3"
    ] = hits_at_k(
        y_true,
        probabilities,
        model.classes_,
        3,
    )

    metrics[
        "Hits_at_5"
    ] = hits_at_k(
        y_true,
        probabilities,
        model.classes_,
        5,
    )

    return {
        "file":
            file_name,

        "seed":
            seed,

        "missing_rate":
            MISSING_RATE,

        "n_total_relations":
            int(
                df[TARGET]
                .notna()
                .sum()
            ),

        "n_eligible_relations":
            int(n_eligible),

        "n_masked":
            int(
                len(query_idx)
            ),

        "n_context":
            int(
                len(context_idx)
            ),

        "n_target_classes_context":
            int(
                y_context.nunique()
            ),

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

    result_df = pd.DataFrame(
        results
    )

    # --------------------------------------------------------
    # 1. Every file × seed run
    # --------------------------------------------------------

    result_df.to_csv(
        OUTPUT_DIR
        / "per_run_metrics.csv",
        index=False,
    )

    metadata_cols = {
        "file",
        "seed",
        "missing_rate",
        "n_total_relations",
        "n_eligible_relations",
        "n_masked",
        "n_context",
        "n_target_classes_context",
    }

    metric_cols = [
        c
        for c in result_df.columns
        if (
            c not in metadata_cols
            and pd.api.types.is_numeric_dtype(
                result_df[c]
            )
        )
    ]

    # --------------------------------------------------------
    # 2. Average 3 seeds inside each OCEL
    # --------------------------------------------------------

    per_file = (
        result_df
        .groupby(
            "file",
            dropna=False,
        )[metric_cols]
        .mean()
        .reset_index()
    )

    per_file.to_csv(
        OUTPUT_DIR
        / "per_file_mean_metrics.csv",
        index=False,
    )

    # --------------------------------------------------------
    # 3. Mean ± SD across OCEL files
    # --------------------------------------------------------

    overall = (
        per_file[
            metric_cols
        ]
        .agg([
            "mean",
            "std",
        ])
        .T
        .reset_index()
        .rename(
            columns={
                "index":
                    "metric",

                "mean":
                    "mean_across_files",

                "std":
                    "std_across_files",
            }
        )
    )

    overall.to_csv(
        OUTPUT_DIR
        / "overall_mean_std_across_files.csv",
        index=False,
    )

    return (
        result_df,
        per_file,
        overall,
    )


# ============================================================
# Main
# ============================================================

def main():

    files = discover_files()

    if not files:
        raise FileNotFoundError(
            "No O2O files found. "
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
            f"Unique target object IDs: "
            f"{df[TARGET].nunique(dropna=True):,}"
        )

        # Diagnostic:
        target_counts = (
            df[TARGET]
            .value_counts()
        )

        print(
            f"Target IDs occurring once: "
            f"{int((target_counts == 1).sum()):,}"
        )

        print(
            f"Target IDs occurring >=2 times: "
            f"{int((target_counts >= 2).sum()):,}"
        )

        for seed in SEEDS:

            print(
                f"  target_object_id "
                f"| seed={seed}"
            )

            result = run_one(
                df=df,
                file_name=file_path.name,
                seed=seed,
            )

            if result is not None:
                results.append(
                    result
                )

    if not results:
        print(
            "\nNo valid experiments were run."
        )
        return

    (
        result_df,
        per_file,
        overall,
    ) = save_results(
        results
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
