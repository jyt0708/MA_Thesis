"""Predict missing O2O endpoints with SAP RPT, restricted to DVIDEO candidates.

Only ContainerLogistics 1, 2 and 3 are being predicted. 
Use the already-masked OCELs and CSV tables; do not mask again. 
Original OCELs are read ONLY for evaluation after queries/predictions have been constructed. 
DVIDEO and its first-ten candidate limit are unchanged. 
Every predicted endpoint must belong to its candidate set.

Examples (from the Logistics directory):
    python predict_o2o_new.py --prepare-only
    python predict_o2o_new.py
    python predict_o2o_new.py --log-ids 2 --max-context-size 8192 --bagging 8

No model is loaded in prepare-only mode. Results then go into a preparation/subdirectory, 
so they cannot be mistaken for model predictions. 
Normal runs use SAP_RPT_OSS_Classifier and may download its pretrained checkpoints.

Prepare-only: Reads the masked OCELs and their CSV tables, builds source features, and 
runs DVIDEO to identify candidate missing O2O relations.

"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from dataclasses import dataclass
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "coopis-dvideo" / "detection"))

from detection.missing_o2o_relation import detect_missing_o2o_relation
from detection.missing_object import load_cardinality_constraints
from importer.ocel_importer import import_ocel_raw

LOGS = {1: "ContainerLogistics_1.json", 2: "ContainerLogistics_2.jsonocel", 3: "ContainerLogistics_3.jsonocel"}
TARGET = "target_object_id"


def prepare_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Remove target-leaking features.
    """
    return df.drop(columns=[
        TARGET,
        "target_object_type",
        "o2o_qualifier",
        "target_event_count",
        "target_o2o_degree",
        "shared_event_count",
    ], errors="ignore").copy()


@dataclass
class Prepared:
    raw: dict
    raw_to_table: dict[str, str]
    table_to_raw: dict[str, str]
    context: pd.DataFrame
    queries: pd.DataFrame
    records: list[dict]


def build_id_maps(raw: dict) -> tuple[dict[str, str], dict[str, str]]:
    """
    Map original OCEL IDs to CSV IDs.
    Original OCEL and its tabular version use different IDs. 
    """
    counts = Counter()
    mapping = {}
    
    for obj in raw["objects"]:
        oid, kind = str(obj["id"]), str(obj["type"])
        if oid in mapping:
            raise ValueError(f"Duplicate OCEL object ID: {oid}")
        counts[kind] += 1
        mapping[oid] = f"{kind}_{counts[kind]}"
        
    reverse = {value: key for key, value in mapping.items()}
    
    if len(reverse) != len(mapping):
        raise ValueError("Object ID renaming is ambiguous.")
    
    return mapping, reverse


def build_source_features(raw: dict, mapping: dict,
                          objects: pd.DataFrame) -> tuple[dict, Counter, dict]:
    """
    Return object types, O2O degrees, and distinct event counts for queries.

    Input CSV tables are assumed to match the masked OCEL.
    """
    types = {mapping[o["id"]]: o["type"] for o in raw["objects"]}
    retained = set(objects["object_id"])

    degree = Counter()
    for obj in raw["objects"]:
        source = mapping[obj["id"]]
        for relation in obj.get("relationships", []):
            target = mapping.get(relation["objectId"])
            if source in retained and target in retained:
                degree[source] += 1
                degree[target] += 1

    event_ids = defaultdict(set)
    for event in raw["events"]:
        for relation in event.get("relationships", []):
            oid = mapping.get(relation["objectId"])
            if oid is not None:
                event_ids[oid].add(event["id"])
    event_counts = {oid: len(event_ids[oid]) for oid in types}
    return types, degree, event_counts


def prepare(masked_path: Path, o2o_csv: Path, object_csv: Path, constraints_path: Path) -> Prepared:
    """
    Load the masked OCEL and its CSV tables. Detect potentially missing
    O2O relations, and prepare candidate lists and model input rows for prediction
    """
    raw = json.loads(masked_path.read_text(encoding="utf-8-sig"))
    mapping, reverse = build_id_maps(raw)
    context = pd.read_csv(o2o_csv, keep_default_na=False)
    objects = pd.read_csv(object_csv, keep_default_na=False)
    types, degree, event_counts = build_source_features(raw, mapping, objects)
    
    # Detect missing o2o relation using DVIDEO
    detections = detect_missing_o2o_relation(
        import_ocel_raw(masked_path), load_cardinality_constraints(constraints_path))
    
    retained = set(objects["object_id"])
    context_labels = set(context[TARGET])
    records, queries = [], []
    
    for detected in detections.to_dict("records"):
        source_raw = str(detected["ocel:oid"])
        source = mapping[source_raw]
        raw_candidates = sorted(set(detected["candidates"]))
        existing = set(detected["actual_targets"])
        
        candidates = sorted({mapping[c] for c in raw_candidates
                             if c in mapping and c not in existing})
        
        # Candidate IDs that already appear in the context
        supported = [c for c in candidates if c in context_labels]
        status = "ready"
        
        if source not in retained:
            # Source is absent from the object CSV
            status = "source_filtered_from_tables"
        elif not candidates:
            # No candidates remain after filtering
            status = "no_new_candidates"
        elif not supported:
            # Candidates exist, but none appears in the context
            status = "no_supported_candidates"
            
        records.append({
            "source_id_raw": source_raw, "source_object_id": source,
            "source_object_type": types[source], "qualifier": detected["qualifier"],
            "expected_target_type": detected["expected_target_type"],
            "min_expected": int(detected["min_expected"]),
            "actual_count": int(detected["actual_count"]),
            "required_additions": int(detected["min_expected"] - detected["actual_count"]),
            "dvideo_candidates_raw": raw_candidates,
            "candidate_target_ids": candidates, "supported_candidate_ids": supported,
            "actual_targets_raw": sorted(existing), "status": status,
            "predicted_target_ids": [], "predicted_target_ids_raw": [],
            "ranked_target_ids": [], "ranked_target_ids_raw": [],
            "ranked_probabilities": [], "allowed_probability_mass": None,
        })
        
        # One query is created per detection
        queries.append({"source_object_id": source, "source_object_type": types[source],
                        "relation_origin": "explicit", "source_event_count": event_counts[source],
                        "source_o2o_degree": degree[source]})
        
    # DVIDEO supplies new query rows, not the original hidden relation rows.
    feature_columns = prepare_features(context).columns
    query_features = pd.DataFrame(queries).reindex(columns=feature_columns)
    
    return Prepared(raw, mapping, reverse, context, query_features, records)


def select_context(context: pd.DataFrame, required_labels: set[str], limit: int, seed: int) -> pd.DataFrame:
    """
    Keep at least one row per supported candidate label within RPT's context.

    Select using candidates and observed labels only.
    """
    if limit < 1:
        raise ValueError("max_context_size must be positive")
    
    if len(context) <= limit:
        return context.reset_index(drop=True)
    
    groups = context.groupby(TARGET, sort=False).indices
    required = sorted(required_labels & set(groups))
    
    if len(required) > limit:
        raise ValueError(f"{len(required)} candidate classes exceed context budget {limit}; increase --max-context-size.")
    
    rng = np.random.default_rng(seed)
    selected = [int(rng.choice(groups[label])) for label in required]
    remaining = np.setdiff1d(np.arange(len(context)), selected)
    selected.extend(rng.choice(remaining, size=limit - len(selected), replace=False).tolist())
    rng.shuffle(selected)
    return context.iloc[selected].reset_index(drop=True)


def constrain_scores(probabilities: np.ndarray, classes: list[str], candidates: list[str],
                     required: int) -> dict:
    """
    Restrict model probabilities to allowed candidates, rank them, and select the 
    required number of targets.
    """
    values = np.asarray(probabilities, dtype=float)
    
    if values.shape != (len(classes),) or len(set(classes)) != len(classes):
        raise ValueError("Probability columns must match unique model.classes_ labels.")
    
    if required < 1:
        raise ValueError("A deficient source must require at least one addition.")
    
    allowed = set(candidates)
    if not allowed:
        return {"status": "no_new_candidates", "ranked": [], "scores": [], "selected": [], "mass": 0.0}
    
    # Only keep DVIDEO's candidates
    eligible = [i for i, label in enumerate(classes) if label in allowed]
    
    # Discard unusable probabilitiess
    positive = [i for i in eligible if np.isfinite(values[i]) and values[i] > 0]
    positive.sort(key=lambda i: (-values[i], classes[i]))
    ranked = [classes[i] for i in positive]
    status = "predicted" if len(ranked) >= required else "partially_predicted"
    
    if not eligible:
        status = "no_supported_candidates"
    elif not positive:
        status = "no_usable_candidate_probability"
    
    # required = min_expected - actual_count
    return {"status": status, "ranked": ranked, "scores": values[positive].tolist(),
            "selected": ranked[:required], "mass": float(values[positive].sum())}


def predict(prepared: Prepared, model, max_context_size: int, seed: int) -> dict:
    """
    Use SAP RPT to score candidate target objects.
    Skipped in preparation-only mode.
    """
    ready = [i for i, record in enumerate(prepared.records) if record["status"] == "ready"]
    
    # Only keep supported candidates
    needed = {c for i in ready for c in prepared.records[i]["supported_candidate_ids"]}
    context = select_context(prepared.context, needed, max_context_size, seed)
    stats = {"context_rows_available": len(prepared.context), "context_rows_used": len(context),
             "context_classes": int(context[TARGET].nunique())}
    
    if not ready:
        return stats
    
    X_context = prepare_features(context)
    X_query = prepare_features(prepared.queries.iloc[ready]).reindex(columns=X_context.columns)
    
    model.fit(X_context, context[TARGET].astype(str))
    classes = [str(label) for label in model.classes_]
    
    if not needed <= set(classes):
        raise ValueError("Fitted model lost supported candidate labels.")
     
    # Predict probabilities over target IDs
    probabilities = np.asarray(model.predict_proba(X_query))
    
    if probabilities.shape != (len(ready), len(classes)):
        raise ValueError("Unexpected predict_proba shape.")
    
    for index, row in zip(ready, probabilities):
        record = prepared.records[index]
        selection = constrain_scores(row, classes, record["candidate_target_ids"], record["required_additions"])
        record.update({
            "status": selection["status"], "predicted_target_ids": selection["selected"],
            "predicted_target_ids_raw": [prepared.table_to_raw[c] for c in selection["selected"]],
            "ranked_target_ids": selection["ranked"],
            "ranked_target_ids_raw": [prepared.table_to_raw[c] for c in selection["ranked"]],
            "ranked_probabilities": selection["scores"], "allowed_probability_mass": selection["mass"],
        })
        assert set(record["predicted_target_ids_raw"]) <= set(record["dvideo_candidates_raw"])
    
    return stats


def edge_counts(raw: dict) -> Counter:
    return Counter((obj["id"], relation["qualifier"], relation["objectId"])
                   for obj in raw["objects"] for relation in obj.get("relationships", []))


def evaluate(prepared: Prepared, original_path: Path, inference_done: bool) -> tuple[list[dict], dict]:
    """
    Compare predictions to removed edges, without using truth to form queries.

    Multi-edge groups use set/multiset edge recovery. Endpoint classification
    metrics are reported only for groups with exactly one removed edge and one
    required addition; other groups have no unambiguous row-to-row assignment.
    """
    original = json.loads(original_path.read_text(encoding="utf-8-sig"))
    if {k: v for k, v in original.items() if k != "objects"} != {
            k: v for k, v in prepared.raw.items() if k != "objects"}:
        raise ValueError("Original and masked logs differ outside object relationships.")
    original_objects = {o["id"]: {k: v for k, v in o.items() if k != "relationships"}
                        for o in original["objects"]}
    masked_objects = {o["id"]: {k: v for k, v in o.items() if k != "relationships"}
                      for o in prepared.raw["objects"]}
    if original_objects != masked_objects:
        raise ValueError("Original and masked object metadata differ.")
    before, after = edge_counts(original), edge_counts(prepared.raw)
    if after - before:
        raise ValueError("Masked log contains edges absent from original.")
    removed = before - after
    types = {o["id"]: o["type"] for o in original["objects"]}
    records = {(r["source_id_raw"], r["qualifier"], r["expected_target_type"]): r
               for r in prepared.records}
    group_counts = Counter()
    for (source, qualifier, target), count in removed.items():
        group_counts[(source, qualifier, types.get(target))] += count
    proposed = Counter((r["source_id_raw"], r["qualifier"], target)
                       for r in prepared.records for target in r["predicted_target_ids_raw"])
    matched = proposed & removed
    details, single_true, single_pred = [], [], []
    for (source, qualifier, target), count in sorted(removed.items()):
        key = (source, qualifier, types.get(target))
        record = records.get(key)
        ranked = record["ranked_target_ids_raw"] if record else []
        mapped = prepared.raw_to_table.get(target)
        row = {"source_id_raw": source, "qualifier": qualifier, "true_target_id_raw": target,
               "removed_count": count, "detected": record is not None,
               "in_dvideo_candidates": bool(record and target in record["dvideo_candidates_raw"]),
               "in_supported_candidates": bool(record and mapped in record["supported_candidate_ids"]),
               "status": record["status"] if record else "not_detected",
               "recovered_count": matched[(source, qualifier, target)],
               "hit_at_3": target in ranked[:3], "hit_at_5": target in ranked[:5]}
        details.append(row)
        if (record and group_counts[key] == 1 and record["required_additions"] == 1
                and len(record["predicted_target_ids_raw"]) == 1):
            single_true.append(target)
            single_pred.append(record["predicted_target_ids_raw"][0])
    total = sum(removed.values())
    ratio = lambda n, d: n / d if d else None
    summary = {
        "original_o2o_rows": sum(before.values()), "masked_o2o_rows": sum(after.values()),
        "removed_o2o_rows": total,
        "removed_edges_detected": sum(r["removed_count"] for r in details if r["detected"]),
        "dvideo_candidate_recall": ratio(sum(r["removed_count"] for r in details if r["in_dvideo_candidates"]), total),
        "supported_candidate_recall": ratio(sum(r["removed_count"] for r in details if r["in_supported_candidates"]), total),
    }
    if inference_done:
        recovered = sum(matched.values())
        summary.update({"predicted_edges": sum(proposed.values()), "recovered_edges": recovered,
                        "repair_precision": ratio(recovered, sum(proposed.values())),
                        "repair_recall_all_removed": ratio(recovered, total),
                        "hits_at_3_all_removed": ratio(sum(r["removed_count"] for r in details if r["hit_at_3"]), total),
                        "hits_at_5_all_removed": ratio(sum(r["removed_count"] for r in details if r["hit_at_5"]), total),
                        "single_endpoint_evaluated": len(single_true)})
        if single_true:
            from sklearn.metrics import accuracy_score, precision_recall_fscore_support
            summary["single_endpoint_accuracy"] = float(accuracy_score(single_true, single_pred))
            for average in ("macro", "weighted"):
                p, r, f, _ = precision_recall_fscore_support(single_true, single_pred, average=average, zero_division=0)
                summary.update({f"single_endpoint_precision_{average}": float(p),
                                f"single_endpoint_recall_{average}": float(r),
                                f"single_endpoint_f1_{average}": float(f)})
    return details, summary


def write_records(path: Path, records: list[dict], empty_columns: list[str]) -> None:
    frame = pd.DataFrame(records) if records else pd.DataFrame(columns=empty_columns)
    for column in frame.columns:
        frame[column] = frame[column].map(lambda v: json.dumps(v, ensure_ascii=False) if isinstance(v, (list, dict)) else v)
    frame.to_csv(path, index=False)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--log-ids", nargs="+", type=int, choices=LOGS, default=[1, 2])
    parser.add_argument("--masked-dir", type=Path, default=ROOT / "masked_o2o_relations")
    parser.add_argument("--tabular-dir", type=Path, default=ROOT / "container_logistics_tabular_o2o_masked")
    parser.add_argument("--constraints-dir", type=Path, default=ROOT / "container_logistics_constraints")
    parser.add_argument("--original-dir", type=Path, default=ROOT)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "results_o2o_new")
    parser.add_argument("--max-context-size", type=int, default=2048)
    parser.add_argument("--bagging", type=int, default=1)
    parser.add_argument("--query-batch-size", type=int, default=64)
    parser.add_argument("--seed", type=int, default=42, help="RPT/context selection seed; no new masking.")
    parser.add_argument("--prepare-only", action="store_true", help="Run real DVIDEO and prepare features without loading RPT.")
    parser.add_argument("--no-evaluate", action="store_true", help="Do not read original logs or compute ground-truth metrics.")
    args = parser.parse_args(argv)
    
    if min(args.max_context_size, args.bagging, args.query_batch_size) < 1:
        parser.error("Context size, bagging, and query batch size must be positive.")
    output = args.output_dir / "preparation" if args.prepare_only else args.output_dir
    output.mkdir(parents=True, exist_ok=True)
    summaries, failures = [], 0
    model = None
    
    for log_id in dict.fromkeys(args.log_ids):
        name = Path(LOGS[log_id])
        try:
            prepared = prepare(args.masked_dir / f"{name.stem}_masked{name.suffix}",
                               args.tabular_dir / f"o2o_level_{log_id}.csv",
                               args.tabular_dir / f"object_level_{log_id}.csv",
                               args.constraints_dir / f"cardinality_constraints_{name.stem}.yaml")
            ready = [r for r in prepared.records if r["status"] == "ready"]
            summary = {"log_id": log_id, "input": name.name, "prepare_only": args.prepare_only,
                       "seed": args.seed, "max_context_size": args.max_context_size,
                       "bagging": args.bagging, "query_batch_size": args.query_batch_size,
                       "dvideo_source_rule_groups": len(prepared.records), "ready_groups": len(ready),
                       "context_rows_available": len(prepared.context)}
            
            if args.prepare_only:
                needed = {c for r in ready for c in r["supported_candidate_ids"]}
                selected = select_context(prepared.context, needed, args.max_context_size, args.seed)
                summary["context_rows_used"] = len(selected)
            else:
                if ready and model is None:
                    from sap_rpt_oss import SAP_RPT_OSS_Classifier
                    model = SAP_RPT_OSS_Classifier(max_context_size=args.max_context_size,
                                                 bagging=args.bagging, test_chunk_size=args.query_batch_size)
                    model.seed = args.seed
                    model.tokenizer.random_seed = args.seed
                summary.update(predict(prepared, model, args.max_context_size, args.seed))
            
            prefix = f"ContainerLogistics_{log_id}"
            details = None
            
            if not args.no_evaluate:
                details, metrics = evaluate(prepared, args.original_dir / name, not args.prepare_only)
                summary.update(metrics)
            
            write_records(output / f"{prefix}_candidate_predictions.csv", prepared.records,
                          ["source_id_raw", "qualifier", "status", "predicted_target_ids"])
            prepared.queries.to_csv(output / f"{prefix}_query_features.csv", index=False)
            
            if details is not None:
                write_records(output / f"{prefix}_removed_edge_evaluation.csv", details,
                              ["source_id_raw", "qualifier", "true_target_id_raw", "status"])
            
            summary["statuses"] = dict(Counter(r["status"] for r in prepared.records))
            (output / f"{prefix}_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
            summaries.append(summary)
            print(json.dumps(summary, indent=2))
        except Exception as error:
            failures += 1
            print(f"ERROR {name.name}: {type(error).__name__}: {error}", file=sys.stderr)
    
    if summaries:
        write_records(output / "per_file_metrics.csv", summaries, ["log_id"])
    
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
