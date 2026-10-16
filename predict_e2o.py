"""Evaluate pretrained SAP RPT on E2O endpoint candidate ranking.

The script performs two separate steps:

1. It converts an E2O relation view into a candidate-ranking table and writes
   ``e2o_candidate_rank.csv`` without modifying the input CSV.
2. It evaluates the pretrained ``SAP_RPT_OSS_Classifier`` as a binary link
   scorer. For each held-out E2O relation, candidate objects are ranked by the
   predicted probability of ``is_linked == 1``.

The candidate ID, event ID, and query ID are metadata only. They are removed
from the SAP RPT feature table, which prevents identifier memorisation.


Useful outputs in ``results_e2o_candidate_ranking/``:

    e2o_candidate_rank.csv
    per_run_metrics.csv
    per_query_ranking.csv
    overall_mean_std.csv

"""

from __future__ import annotations

import gc
import json
import warnings
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    precision_score,
    recall_score,
)


# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------

DEFAULT_INPUT = Path("container_logistics_tabular_new/e2o_level_*.csv")
DEFAULT_OUTPUT_DIR = Path("results_e2o_candidate_ranking")
DEFAULT_SEEDS = (42, 43, 44)

MISSING_RATE = 0.10
CANDIDATES_PER_QUERY = 20
HARD_NEGATIVE_FRACTION = 0.50
CANDIDATE_GENERATION_SEED = 2026
MIN_OBJECT_RELATIONS = 2

MAX_CONTEXT_SIZE = 8192
BAGGING = 8
PREDICTION_BATCH_SIZE = 2048

REQUIRED_COLUMNS = (
    "event_id",
    "event_activity",
    "event_timestamp",
    "object_id",
    "object_type",
)

# These are event-level rather than endpoint-level columns in the supplied
# view. Identifier-like and free-form data fields are intentionally excluded:
# they may contain endpoint IDs or instance-specific strings.
SAFE_EVENT_ATTRIBUTE_COLUMNS = (
    "event_attr__cpee:lifecycle:transition",
    "event_attr__lifecycle:transition",
    "event_attr__sub:root",
)

# Metadata is needed to reconstruct rankings but must never be passed to SAP
# RPT as a feature.
MODEL_METADATA_COLUMNS = {
    "query_id",
    "source_row_index",
    "event_id",
    "candidate_object_id",
    "is_linked",
}


@dataclass(frozen=True)
class HistoryEntry:
    timestamp: pd.Timestamp | None
    event_id: str
    activity: str


def read_csv_auto(path: Path) -> pd.DataFrame:
    """Read E2O CSV matches and attach attributes from paired object views."""
    has_wildcard = any(character in path.name for character in "*?[")
    input_paths = sorted(path.parent.glob(path.name)) if has_wildcard else [path]
    input_paths = [input_path for input_path in input_paths if input_path.is_file()]
    if not input_paths:
        raise FileNotFoundError(f"No input CSV matches: {path}")

    frames: list[pd.DataFrame] = []
    for input_path in input_paths:
        df = pd.read_csv(input_path, sep=None, engine="python")
        df.columns = [str(column).strip() for column in df.columns]
        missing = [column for column in REQUIRED_COLUMNS if column not in df.columns]
        if missing:
            raise KeyError(f"Missing required E2O columns in {input_path}: {missing}")

        if not input_path.name.startswith("e2o_level_"):
            raise ValueError(
                f"Cannot infer the corresponding object view for {input_path.name!r}; "
                "expected an e2o_level_*.csv filename."
            )
        object_path = input_path.with_name(
            input_path.name.replace("e2o_level_", "object_level_", 1)
        )
        if not object_path.is_file():
            raise FileNotFoundError(
                f"Corresponding object-level CSV does not exist: {object_path}"
            )

        objects = pd.read_csv(object_path, sep=None, engine="python")
        objects.columns = [str(column).strip() for column in objects.columns]
        if "object_id" not in objects.columns:
            raise KeyError(f"Missing 'object_id' column in {object_path}")
        object_attribute_columns = [
            column for column in objects.columns if column.startswith("obj_attr__")
        ]
        if objects["object_id"].isna().any():
            raise ValueError(f"Missing object IDs in {object_path}")
        if objects["object_id"].duplicated().any():
            raise ValueError(f"Duplicate object IDs in {object_path}")

        overlapping = sorted(set(df.columns).intersection(object_attribute_columns))
        if overlapping:
            raise ValueError(
                f"E2O input {input_path} already contains object attributes: {overlapping}"
            )

        df["object_id"] = df["object_id"].astype("string")
        objects["object_id"] = objects["object_id"].astype("string")
        df = df.merge(
            objects[["object_id", *object_attribute_columns]],
            on="object_id",
            how="left",
            validate="many_to_one",
        )
        frames.append(df)

    return pd.concat(frames, ignore_index=True, sort=False)


def scalar(value: object) -> object:
    """Convert nested or NumPy values to stable CSV-compatible scalars."""
    if isinstance(value, (dict, list, tuple, set)):
        return json.dumps(value, ensure_ascii=False, sort_keys=True)
    if isinstance(value, np.generic):
        return value.item()
    return value


def clean_relations(df: pd.DataFrame) -> pd.DataFrame:
    """Keep valid E2O rows and remove duplicate event-object pairs."""
    relations = df.loc[
        df["event_id"].notna() & df["object_id"].notna()
    ].copy()
    relations["event_id"] = relations["event_id"].astype(str)
    relations["object_id"] = relations["object_id"].astype(str)
    relations["event_activity"] = relations["event_activity"].fillna("__missing__").astype(str)
    relations["object_type"] = relations["object_type"].fillna("__missing__").astype(str)
    relations["source_row_index"] = relations.index.astype(int)

    duplicated = relations.duplicated(["event_id", "object_id"], keep="first")
    if duplicated.any():
        warnings.warn(
            f"Removed {int(duplicated.sum())} duplicate event-object relation rows."
        )
        relations = relations.loc[~duplicated].copy()

    relations["_event_time"] = pd.to_datetime(
        relations["event_timestamp"], errors="coerce", utc=True
    )
    return relations.reset_index(drop=True)


def validate_event_level_columns(
    relations: pd.DataFrame,
    columns: Sequence[str],
) -> list[str]:
    """Keep only columns whose non-null value is consistent within each event."""
    safe: list[str] = []
    for column in columns:
        if column not in relations.columns:
            continue
        maximum_values_per_event = (
            relations.groupby("event_id", dropna=False)[column]
            .nunique(dropna=True)
            .max()
        )
        if pd.isna(maximum_values_per_event) or maximum_values_per_event <= 1:
            safe.append(column)
        else:
            warnings.warn(
                f"Dropped {column!r}: it is not constant within event_id and "
                "may contain endpoint-specific information."
            )
    return safe


def mode_or_missing(values: pd.Series) -> str:
    observed = values.dropna().astype(str)
    if observed.empty:
        return "__missing__"
    modes = observed.mode()
    return str(modes.iloc[0] if not modes.empty else observed.iloc[0])


def build_object_histories(
    relations: pd.DataFrame,
) -> tuple[
    dict[str, list[HistoryEntry]],
    dict[str, str],
    dict[str, Counter],
]:
    histories: dict[str, list[HistoryEntry]] = defaultdict(list)
    activity_counts: dict[str, Counter] = defaultdict(Counter)

    history_columns = ["object_id", "event_id", "event_activity", "_event_time"]
    for object_id, event_id, event_activity, raw_timestamp in relations[
        history_columns
    ].itertuples(index=False, name=None):
        timestamp = raw_timestamp if not pd.isna(raw_timestamp) else None
        entry = HistoryEntry(
            timestamp=timestamp,
            event_id=str(event_id),
            activity=str(event_activity),
        )
        object_key = str(object_id)
        histories[object_key].append(entry)
        activity_counts[object_key][entry.activity] += 1

    for entries in histories.values():
        entries.sort(
            key=lambda item: (
                item.timestamp is None,
                item.timestamp if item.timestamp is not None else pd.Timestamp.max.tz_localize("UTC"),
                item.event_id,
            )
        )

    object_types = {
        str(object_id): mode_or_missing(group["object_type"])
        for object_id, group in relations.groupby("object_id", sort=False)
    }
    return dict(histories), object_types, dict(activity_counts)


def iso_timestamp(timestamp: pd.Timestamp | None) -> object:
    if timestamp is None or pd.isna(timestamp):
        return np.nan
    return timestamp.isoformat()


def visible_candidate_features(
    candidate_id: str,
    event_id: str,
    event_activity: str,
    event_time: pd.Timestamp | None,
    remove_query_relation: bool,
    histories: dict[str, list[HistoryEntry]],
    object_types: dict[str, str],
) -> dict[str, object]:
    """Compute candidate features after optionally hiding the positive edge."""
    entries = histories[candidate_id]
    if remove_query_relation:
        visible = [entry for entry in entries if entry.event_id != event_id]
    else:
        visible = entries

    unique_event_count = len({entry.event_id for entry in visible})
    unique_activities = len({entry.activity for entry in visible})
    same_activity_count = sum(entry.activity == event_activity for entry in visible)

    timed = [entry for entry in visible if entry.timestamp is not None]
    first_time = timed[0].timestamp if timed else None
    last_time = timed[-1].timestamp if timed else None

    previous: HistoryEntry | None = None
    following: HistoryEntry | None = None
    if event_time is not None:
        for entry in timed:
            if entry.timestamp < event_time:
                previous = entry
            elif entry.timestamp > event_time:
                following = entry
                break

    seconds_since_previous = (
        float((event_time - previous.timestamp).total_seconds())
        if event_time is not None and previous is not None and previous.timestamp is not None
        else np.nan
    )
    seconds_until_next = (
        float((following.timestamp - event_time).total_seconds())
        if event_time is not None and following is not None and following.timestamp is not None
        else np.nan
    )

    return {
        "candidate_object_type": object_types.get(candidate_id, "__missing__"),
        "candidate_event_count_visible": unique_event_count,
        "candidate_unique_activity_count_visible": unique_activities,
        "candidate_same_activity_count_visible": same_activity_count,
        "candidate_first_event_time_visible": iso_timestamp(first_time),
        "candidate_last_event_time_visible": iso_timestamp(last_time),
        "candidate_previous_activity": previous.activity if previous else "__none__",
        "candidate_next_activity": following.activity if following else "__none__",
        "seconds_since_candidate_previous_event": seconds_since_previous,
        "seconds_until_candidate_next_event": seconds_until_next,
        "candidate_has_previous_event": int(previous is not None),
        "candidate_has_next_event": int(following is not None),
    }


def sample_negatives(
    positive_id: str,
    positive_type: str,
    connected_ids: set[str],
    eligible_ids: Sequence[str],
    eligible_ids_by_type: dict[str, list[str]],
    n_negatives: int,
    hard_fraction: float,
    rng: np.random.Generator,
) -> list[str]:
    excluded = set(connected_ids)
    excluded.add(positive_id)

    same_type_pool = [
        candidate
        for candidate in eligible_ids_by_type.get(positive_type, [])
        if candidate not in excluded
    ]
    desired_hard = min(n_negatives, int(round(n_negatives * hard_fraction)))
    hard_count = min(desired_hard, len(same_type_pool))
    selected: list[str] = []
    if hard_count:
        selected.extend(
            rng.choice(same_type_pool, size=hard_count, replace=False).tolist()
        )

    remaining_pool = [
        candidate
        for candidate in eligible_ids
        if candidate not in excluded and candidate not in selected
    ]
    remaining_count = min(n_negatives - len(selected), len(remaining_pool))
    if remaining_count:
        selected.extend(
            rng.choice(remaining_pool, size=remaining_count, replace=False).tolist()
        )
    return selected


def build_candidate_table(
    raw_df: pd.DataFrame,
    candidates_per_query: int,
    hard_negative_fraction: float,
    seed: int,
    min_object_relations: int,
    max_positive_relations: int | None,
) -> tuple[pd.DataFrame, dict[str, object]]:
    """Create one positive and sampled negative rows for every eligible edge."""
    relations = clean_relations(raw_df)
    object_attribute_columns = sorted(
        column for column in relations.columns if column.startswith("obj_attr__")
    )
    object_attributes = relations[
        ["object_id", *object_attribute_columns]
    ].drop_duplicates()
    if object_attributes["object_id"].duplicated().any():
        conflicting_ids = (
            object_attributes.loc[
                object_attributes["object_id"].duplicated(keep=False), "object_id"
            ]
            .drop_duplicates()
            .head()
            .tolist()
        )
        raise ValueError(
            "Conflicting object attributes across input views for object IDs: "
            f"{conflicting_ids}"
        )
    object_attributes = object_attributes.rename(
        columns={
            "object_id": "candidate_object_id",
            **{
                column: f"candidate_{column}"
                for column in object_attribute_columns
            },
        }
    )
    safe_event_attributes = validate_event_level_columns(
        relations, SAFE_EVENT_ATTRIBUTE_COLUMNS
    )
    histories, object_types, _ = build_object_histories(relations)

    relation_counts = relations["object_id"].value_counts()
    eligible_ids = sorted(
        str(object_id)
        for object_id, count in relation_counts.items()
        if int(count) >= min_object_relations
    )
    eligible_id_set = set(eligible_ids)
    eligible_ids_by_type: dict[str, list[str]] = defaultdict(list)
    for object_id in eligible_ids:
        eligible_ids_by_type[object_types[object_id]].append(object_id)

    connected_by_event = {
        str(event_id): set(group["object_id"].astype(str))
        for event_id, group in relations.groupby("event_id", sort=False)
    }
    event_relation_counts = {
        event_id: len(object_ids)
        for event_id, object_ids in connected_by_event.items()
    }

    positives = relations.loc[relations["object_id"].isin(eligible_id_set)].copy()
    if max_positive_relations is not None and len(positives) > max_positive_relations:
        positives = positives.sample(n=max_positive_relations, random_state=seed)
        positives = positives.sort_values("source_row_index")

    rng = np.random.default_rng(seed)
    candidate_rows: list[dict[str, object]] = []
    skipped_small_pools = 0

    for query_number, (_, row) in enumerate(positives.iterrows()):
        event_id = str(row["event_id"])
        positive_id = str(row["object_id"])
        positive_type = object_types[positive_id]
        negatives = sample_negatives(
            positive_id=positive_id,
            positive_type=positive_type,
            connected_ids=connected_by_event[event_id],
            eligible_ids=eligible_ids,
            eligible_ids_by_type=dict(eligible_ids_by_type),
            n_negatives=candidates_per_query - 1,
            hard_fraction=hard_negative_fraction,
            rng=rng,
        )
        if not negatives:
            skipped_small_pools += 1
            continue

        query_id = f"q{query_number:08d}"
        candidates = [positive_id, *negatives]
        candidates = [candidates[index] for index in rng.permutation(len(candidates))]
        event_time = row["_event_time"] if not pd.isna(row["_event_time"]) else None

        base = {
            "query_id": query_id,
            "source_row_index": int(row["source_row_index"]),
            "event_id": event_id,
            "event_activity": str(row["event_activity"]),
            "event_timestamp": scalar(row["event_timestamp"]),
            "event_relation_count_visible": max(
                0, event_relation_counts[event_id] - 1
            ),
        }
        for column in safe_event_attributes:
            base[column] = scalar(row[column])

        for candidate_id in candidates:
            is_positive = candidate_id == positive_id
            candidate_features = visible_candidate_features(
                candidate_id=candidate_id,
                event_id=event_id,
                event_activity=str(row["event_activity"]),
                event_time=event_time,
                remove_query_relation=is_positive,
                histories=histories,
                object_types=object_types,
            )
            candidate_rows.append(
                {
                    **base,
                    "candidate_object_id": candidate_id,
                    **candidate_features,
                    "is_linked": int(is_positive),
                }
            )

    candidate_df = pd.DataFrame(candidate_rows)
    if candidate_df.empty:
        raise ValueError("No candidate-ranking rows could be generated.")
    candidate_df = candidate_df.merge(
        object_attributes,
        on="candidate_object_id",
        how="left",
        validate="many_to_one",
    )

    diagnostics = {
        "input_relations": int(len(relations)),
        "eligible_positive_relations": int(len(positives)),
        "generated_queries": int(candidate_df["query_id"].nunique()),
        "candidate_rows": int(len(candidate_df)),
        "eligible_candidate_objects": int(len(eligible_ids)),
        "safe_event_attribute_columns": safe_event_attributes,
        "candidate_object_attribute_columns": [
            f"candidate_{column}" for column in object_attribute_columns
        ],
        "skipped_queries_without_negatives": int(skipped_small_pools),
    }
    validate_candidate_table(candidate_df, connected_by_event)
    return candidate_df, diagnostics


def validate_candidate_table(
    candidate_df: pd.DataFrame,
    connected_by_event: dict[str, set[str]],
) -> None:
    required = {
        "query_id",
        "event_id",
        "candidate_object_id",
        "is_linked",
    }
    missing = required - set(candidate_df.columns)
    if missing:
        raise ValueError(f"Candidate table is missing columns: {sorted(missing)}")

    positives_per_query = candidate_df.groupby("query_id")["is_linked"].sum()
    if not positives_per_query.eq(1).all():
        bad = positives_per_query.loc[~positives_per_query.eq(1)].head().to_dict()
        raise ValueError(f"Every query must contain exactly one positive: {bad}")

    candidate_counts = candidate_df.groupby("query_id").size()
    if (candidate_counts < 2).any():
        raise ValueError("Every query must contain at least two candidates.")

    duplicates = candidate_df.duplicated(
        ["query_id", "candidate_object_id"], keep=False
    )
    if duplicates.any():
        raise ValueError("A query contains duplicate candidate object IDs.")

    negatives = candidate_df.loc[candidate_df["is_linked"].eq(0)]
    leaked_negative = negatives.apply(
        lambda row: str(row["candidate_object_id"])
        in connected_by_event[str(row["event_id"])],
        axis=1,
    )
    if leaked_negative.any():
        raise ValueError(
            "A negative candidate is actually connected to its query event."
        )


def model_features(candidate_df: pd.DataFrame) -> pd.DataFrame:
    X = candidate_df.drop(columns=MODEL_METADATA_COLUMNS, errors="ignore").copy()
    unnamed = [column for column in X.columns if str(column).startswith("Unnamed:")]
    X = X.drop(columns=unnamed, errors="ignore")
    X = X.dropna(axis=1, how="all")
    if X.empty:
        raise ValueError("No model features remain after metadata removal.")
    forbidden = MODEL_METADATA_COLUMNS.intersection(X.columns)
    if forbidden:
        raise RuntimeError(f"Identifier metadata leaked into features: {forbidden}")
    return X


def choose_evaluation_events(
    candidate_df: pd.DataFrame,
    missing_rate: float,
    seed: int,
) -> set[str]:
    """Split by event ID so the same event never appears in context and query."""
    event_ids = np.asarray(sorted(candidate_df["event_id"].astype(str).unique()))
    if len(event_ids) < 2:
        raise ValueError("At least two event IDs are required for evaluation.")
    n_query = max(1, int(round(len(event_ids) * missing_rate)))
    n_query = min(n_query, len(event_ids) - 1)
    rng = np.random.default_rng(seed)
    return set(rng.choice(event_ids, size=n_query, replace=False).tolist())


def balanced_context_indices(
    context_df: pd.DataFrame,
    max_context_size: int,
    seed: int,
) -> np.ndarray:
    """Sample one positive and one negative per selected context query."""
    rng = np.random.default_rng(seed)
    grouped = list(context_df.groupby("query_id", sort=False).groups.items())
    if not grouped:
        raise ValueError("No context query groups are available.")
    max_groups = max_context_size // 2
    if len(grouped) > max_groups:
        selected_positions = rng.choice(len(grouped), size=max_groups, replace=False)
        grouped = [grouped[position] for position in selected_positions]

    selected: list[int] = []
    for _, group_indices in grouped:
        group = context_df.loc[group_indices]
        positive_indices = group.index[group["is_linked"].eq(1)].to_numpy()
        negative_indices = group.index[group["is_linked"].eq(0)].to_numpy()
        if not len(positive_indices) or not len(negative_indices):
            continue
        selected.append(int(rng.choice(positive_indices)))
        selected.append(int(rng.choice(negative_indices)))

    if len(selected) < 2:
        raise ValueError("Balanced context sampling produced too few rows.")
    return rng.permutation(np.asarray(selected, dtype=int))


def positive_probabilities(model: object, probabilities: np.ndarray) -> np.ndarray:
    probabilities = np.asarray(probabilities)
    if probabilities.ndim == 1:
        return probabilities.astype(float)
    classes = np.asarray(getattr(model, "classes_", []))
    if classes.size == probabilities.shape[1]:
        numeric = pd.to_numeric(pd.Series(classes), errors="coerce").to_numpy()
        matches = np.flatnonzero(numeric == 1)
        if len(matches) == 1:
            return probabilities[:, int(matches[0])].astype(float)
    if probabilities.shape[1] == 2:
        return probabilities[:, 1].astype(float)
    raise ValueError("Could not locate positive-class probability for is_linked=1.")


def score_in_batches(
    model: object,
    X: pd.DataFrame,
    batch_size: int,
) -> tuple[np.ndarray, np.ndarray]:
    predictions: list[np.ndarray] = []
    scores: list[np.ndarray] = []
    for start in range(0, len(X), batch_size):
        batch = X.iloc[start : start + batch_size]
        raw_predictions = np.asarray(model.predict(batch)).reshape(-1)
        numeric_predictions = pd.to_numeric(
            pd.Series(raw_predictions), errors="raise"
        ).astype(int)
        probabilities = np.asarray(model.predict_proba(batch))
        predictions.append(numeric_predictions.to_numpy())
        scores.append(positive_probabilities(model, probabilities))
    return np.concatenate(predictions), np.concatenate(scores)


def harmonic_number(n: int) -> float:
    return sum(1.0 / rank for rank in range(1, n + 1))


def ranking_results(scored: pd.DataFrame, seed: int) -> tuple[pd.DataFrame, dict[str, float]]:
    rows: list[dict[str, object]] = []
    for query_id, group in scored.groupby("query_id", sort=False):
        ordered = group.sort_values(
            ["link_probability", "candidate_object_id"],
            ascending=[False, True],
            kind="mergesort",
        ).reset_index(drop=True)
        positive_positions = np.flatnonzero(ordered["is_linked"].to_numpy() == 1)
        if len(positive_positions) != 1:
            raise ValueError(f"Query {query_id} does not have exactly one positive.")
        rank = int(positive_positions[0] + 1)
        candidate_count = int(len(ordered))
        true_row = ordered.iloc[int(positive_positions[0])]
        rows.append(
            {
                "seed": seed,
                "query_id": query_id,
                "event_id": true_row["event_id"],
                "true_candidate_object_id": true_row["candidate_object_id"],
                "predicted_candidate_object_id": ordered.iloc[0]["candidate_object_id"],
                "true_candidate_rank": rank,
                "candidate_count": candidate_count,
                "reciprocal_rank": 1.0 / rank,
                "hit_at_1": float(rank <= 1),
                "hit_at_3": float(rank <= 3),
                "hit_at_5": float(rank <= 5),
                "random_hit_at_1": 1.0 / candidate_count,
                "random_hit_at_3": min(3, candidate_count) / candidate_count,
                "random_hit_at_5": min(5, candidate_count) / candidate_count,
                "random_mrr": harmonic_number(candidate_count) / candidate_count,
            }
        )

    per_query = pd.DataFrame(rows)
    metrics = {
        "n_queries": float(len(per_query)),
        "mean_candidate_count": float(per_query["candidate_count"].mean()),
        "Hits_at_1": float(per_query["hit_at_1"].mean()),
        "Hits_at_3": float(per_query["hit_at_3"].mean()),
        "Hits_at_5": float(per_query["hit_at_5"].mean()),
        "MRR": float(per_query["reciprocal_rank"].mean()),
        "mean_true_rank": float(per_query["true_candidate_rank"].mean()),
        "random_Hits_at_1": float(per_query["random_hit_at_1"].mean()),
        "random_Hits_at_3": float(per_query["random_hit_at_3"].mean()),
        "random_Hits_at_5": float(per_query["random_hit_at_5"].mean()),
        "random_MRR": float(per_query["random_mrr"].mean()),
    }
    return per_query, metrics


def binary_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    return {
        "binary_accuracy": float(accuracy_score(y_true, y_pred)),
        "binary_balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "binary_precision": float(
            precision_score(y_true, y_pred, zero_division=0)
        ),
        "binary_recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "binary_f1": float(f1_score(y_true, y_pred, zero_division=0)),
    }


def release_model(model: object) -> None:
    del model
    gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except ImportError:
        pass


def evaluate_pretrained(
    candidate_df: pd.DataFrame,
    seeds: Iterable[int],
    missing_rate: float,
    max_context_size: int,
    bagging: int,
    prediction_batch_size: int,
    output_dir: Path,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    try:
        from sap_rpt_oss import SAP_RPT_OSS_Classifier
    except ImportError as exc:
        raise SystemExit(
            "sap_rpt_oss is not installed. Install SAP RPT 1-OSS or rerun "
            "with --build-only to generate the candidate table without evaluation."
        ) from exc

    all_features = model_features(candidate_df)
    run_records: list[dict[str, object]] = []
    query_records: list[pd.DataFrame] = []

    for seed in seeds:
        evaluation_events = choose_evaluation_events(
            candidate_df, missing_rate=missing_rate, seed=int(seed)
        )
        query_mask = candidate_df["event_id"].astype(str).isin(evaluation_events)
        query_df = candidate_df.loc[query_mask].copy()
        context_df = candidate_df.loc[~query_mask].copy()

        context_indices = balanced_context_indices(
            context_df, max_context_size=max_context_size, seed=int(seed)
        )
        X_context = all_features.loc[context_indices].reset_index(drop=True)
        y_context = (
            candidate_df.loc[context_indices, "is_linked"].astype(int).reset_index(drop=True)
        )
        X_query = all_features.loc[query_df.index]
        y_query = query_df["is_linked"].astype(int).to_numpy()

        print(
            f"seed={seed}: context_rows={len(X_context):,}, "
            f"query_rows={len(X_query):,}, "
            f"query_groups={query_df['query_id'].nunique():,}"
        )
        model = SAP_RPT_OSS_Classifier(
            max_context_size=max_context_size,
            bagging=bagging,
        )
        model.fit(X_context, y_context)
        y_pred, link_scores = score_in_batches(
            model, X_query, batch_size=prediction_batch_size
        )

        scored = query_df[
            ["query_id", "event_id", "candidate_object_id", "is_linked"]
        ].copy()
        scored["link_probability"] = link_scores
        per_query, rank_metrics = ranking_results(scored, seed=int(seed))
        query_records.append(per_query)

        run_records.append(
            {
                "seed": int(seed),
                "missing_rate": float(missing_rate),
                "n_context_rows": int(len(X_context)),
                "n_query_candidate_rows": int(len(X_query)),
                **binary_metrics(y_query, y_pred),
                **rank_metrics,
            }
        )
        release_model(model)

    per_run = pd.DataFrame(run_records)
    per_query_all = pd.concat(query_records, ignore_index=True)
    metric_columns = [
        column
        for column in per_run.columns
        if column not in {"seed", "missing_rate"}
        and pd.api.types.is_numeric_dtype(per_run[column])
    ]
    overall = (
        per_run[metric_columns]
        .agg(["mean", "std"])
        .T.reset_index()
        .rename(columns={"index": "metric", "mean": "mean_across_seeds", "std": "std_across_seeds"})
    )

    per_run.to_csv(output_dir / "per_run_metrics.csv", index=False)
    per_query_all.to_csv(output_dir / "per_query_ranking.csv", index=False)
    overall.to_csv(output_dir / "overall_mean_std.csv", index=False)
    return per_run, per_query_all, overall


def main() -> None:
    input_path = DEFAULT_INPUT
    output_dir = DEFAULT_OUTPUT_DIR
    output_dir.mkdir(parents=True, exist_ok=True)
    candidate_path = output_dir / "e2o_candidate_rank.csv"

    raw_df = read_csv_auto(input_path)
    print(f"Loaded {len(raw_df):,} E2O rows from {input_path}")
    candidate_df, diagnostics = build_candidate_table(
        raw_df=raw_df,
        candidates_per_query=CANDIDATES_PER_QUERY,
        hard_negative_fraction=HARD_NEGATIVE_FRACTION,
        seed=CANDIDATE_GENERATION_SEED,
        min_object_relations=MIN_OBJECT_RELATIONS,
        max_positive_relations=None,
    )
    candidate_df.to_csv(candidate_path, index=False)
    print(f"Saved candidate table: {candidate_path}")
    print(json.dumps(diagnostics, indent=2, ensure_ascii=False))

    per_run, _, overall = evaluate_pretrained(
        candidate_df=candidate_df,
        seeds=DEFAULT_SEEDS,
        missing_rate=MISSING_RATE,
        max_context_size=MAX_CONTEXT_SIZE,
        bagging=BAGGING,
        prediction_batch_size=PREDICTION_BATCH_SIZE,
        output_dir=output_dir,
    )
    print("\nPer-run ranking metrics:")
    print(per_run.to_string(index=False))
    print("\nOverall mean ± SD across seeds:")
    print(overall.to_string(index=False))
    print(f"\nSaved all outputs to: {output_dir}")


if __name__ == "__main__":
    main()
