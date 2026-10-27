r"""Evaluate DVIDEO missing-O2O alerts and candidate edges against masked logs.

Run from the repository root:
    .\venv\Scripts\python.exe -X utf8 evaluate_dvideo_o2o.py
    .\venv\Scripts\python.exe -X utf8 evaluate_dvideo_o2o.py --logs 1 2 --overwrite

Ground truth is original minus masked relationship rows. Candidate metrics use
unique directed (source, qualifier, target) triples, not confirmed repairs.
Writes summary.csv (TP/FP/FN) and errors.csv (FP/FN identities and reasons).
No LLM or external API is used.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
from functools import partial
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent
MEASUREMENTS = ("source_alerts", "candidate_edges")
ERROR_FIELDS = ["log", "category", "source_id", "source_type", "qualifier",
                "target_id", "target_type", "removed_target_ids", "min_outgoing",
                "actual_outgoing", "reason", "explanation"]


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def index_log(log: dict) -> tuple[dict, Counter]:
    if not isinstance(log, dict) or not isinstance(log.get("objects"), list):
        raise ValueError("Expected OCEL 2.0 with an objects list.")
    objects, edges = {}, Counter()
    for obj in log["objects"]:
        if not isinstance(obj, dict):
            raise ValueError("Each object must be a dictionary.")
        oid, kind = obj.get("id"), obj.get("type")
        if not isinstance(oid, str) or not oid.strip() or oid in objects:
            raise ValueError(f"Invalid or duplicate object ID: {oid!r}")
        if not isinstance(kind, str) or not kind.strip():
            raise ValueError(f"Invalid object type for {oid!r}")
        objects[oid] = obj
        relationships = obj.get("relationships", [])
        if not isinstance(relationships, list):
            raise ValueError(f"Invalid relationships list for {oid!r}")
        for rel in relationships:
            if not isinstance(rel, dict) or not isinstance(rel.get("objectId"), str):
                raise ValueError(f"Invalid relationship for {oid!r}")
            q = rel.get("qualifier")
            if q is not None and not isinstance(q, str):
                raise ValueError(f"Invalid qualifier for {oid!r}")
            edges[(oid, q, rel["objectId"])] += 1
    return objects, edges


def ground_truth(original: dict, masked: dict) -> tuple[dict, Counter, Counter, Counter]:
    """Validate a deletion-only pair and retain the multiplicity of removed rows."""
    objects, original_edges = index_log(original)
    masked_objects, masked_edges = index_log(masked)
    if objects.keys() != masked_objects.keys():
        raise ValueError("Original and masked object IDs differ.")
    for oid, obj in objects.items():
        metadata = lambda row: {k: v for k, v in row.items() if k != "relationships"}
        if metadata(obj) != metadata(masked_objects[oid]):
            raise ValueError(f"Object metadata changed for {oid!r}.")
        # Check entire rows as well as triples, so edited relationship metadata
        # cannot silently masquerade as a deletion-only experiment.
        rows = lambda row: Counter(json.dumps(r, sort_keys=True) for r in row.get("relationships", []))
        if rows(masked_objects[oid]) - rows(obj):
            raise ValueError(f"Masked log adds or modifies relationships for {oid!r}.")
    if {k: v for k, v in original.items() if k != "objects"} != {
        k: v for k, v in masked.items() if k != "objects"
    }:
        raise ValueError("Content outside object relationships changed (for example events).")
    return objects, original_edges, masked_edges, original_edges - masked_edges


def metrics(truth: set, predicted: set) -> dict:
    return {
        "tp": len(truth & predicted),
        "fp": len(predicted - truth),
        "fn": len(truth - predicted),
    }


def score_detections(objects: dict, original: Counter, masked: Counter,
                     removed: Counter, detections: list[dict], constraints: list[dict]) -> dict:
    """Score native alerts separately from the candidate triples they contain."""
    types = {oid: obj["type"] for oid, obj in objects.items()}
    group = lambda edge: (edge[0], edge[1], types.get(edge[2]))
    truth = set(removed)
    truth_groups = {group(edge) for edge in truth}
    alerts = {(r["ocel:oid"], r["qualifier"], r["expected_target_type"]) for r in detections}
    candidates = {(r["ocel:oid"], r["qualifier"], target)
                  for r in detections for target in r["candidates"]}
    rules = {(r["source_type"], r["qualifier"], r["target_type"]): r for r in constraints}
    actual_counts = Counter()
    for edge, count in masked.items():
        if edge[2] in types:
            actual_counts[group(edge)] += count

    removed_targets = defaultdict(set)
    for edge in truth:
        removed_targets[group(edge)].add(edge[2])

    def missed_reason(key):
        source, qualifier, target_type = key
        if target_type is None:
            return "target_object_missing", "The removed relation points to an object absent from the log."
        rule = rules.get((types[source], qualifier, target_type))
        if rule is None:
            if qualifier is None or not qualifier.strip():
                return "blank_qualifier", "The removed relation has a blank qualifier and no matching constraint."
            return "no_matching_constraint", "No constraint covers this source type, qualifier, and target type."
        minimum = rule.get("min_outgoing")
        if minimum is None or minimum <= 0:
            return "outgoing_minimum_disabled", "min_outgoing is unset or zero, so no outgoing deficit can be detected."
        if actual_counts[key] >= minimum:
            return "remaining_count_satisfies_minimum", "The remaining relationship count still meets min_outgoing."
        if key not in alerts:
            return "no_alert_after_candidate_filters", (
                "The count is below min_outgoing, but DVIDEO returned no alert. "
                "Its incoming-capacity and event-cooccurrence filters can leave no candidates; "
                "the detector does not report which filter excluded them.")
        return "target_not_in_returned_candidates", (
            "DVIDEO flagged this source group, but did not return the removed target. "
            "Candidates are filtered by incoming capacity and event cooccurrence, then capped at ten. "
            "The detector does not report which filter excluded this target.")

    errors = []

    def add_error(category, key, target_id, reason):
        source, qualifier, target_type = key
        rule = rules.get((types[source], qualifier, target_type), {})
        errors.append({
            "category": category, "source_id": source, "source_type": types[source],
            "qualifier": qualifier, "target_id": target_id, "target_type": target_type,
            "removed_target_ids": "; ".join(sorted(removed_targets[key])),
            "min_outgoing": rule.get("min_outgoing"),
            "actual_outgoing": actual_counts[key],
            "reason": reason[0], "explanation": reason[1],
        })

    sort_key = lambda values: tuple(value or "" for value in values)
    for key in sorted(alerts - truth_groups, key=sort_key):
        add_error("source_alerts_fp", key, "", ("no_removed_relation_in_group",
                  "DVIDEO flagged this group, but no relationship in it was removed from the original."))
    for key in sorted(truth_groups - alerts, key=sort_key):
        add_error("source_alerts_fn", key, "", missed_reason(key))
    for edge in sorted(candidates - truth, key=sort_key):
        reason = (("already_present_in_masked_log", "This candidate relation remains in the masked log; no copy was removed.")
                  if masked[edge] else
                  ("not_in_original_log", "This candidate relation does not exist in the original reference log."))
        add_error("candidate_edges_fp", group(edge), edge[2], reason)
    for edge in sorted(truth - candidates, key=sort_key):
        add_error("candidate_edges_fn", group(edge), edge[2], missed_reason(group(edge)))
    return {
        "candidate_edges": metrics(truth, candidates),
        "source_alerts": metrics(truth_groups, alerts),
        "errors": errors,
    }


def load_dvideo(path: Path, progress: bool = False):
    if not (path / "detection" / "missing_o2o_relation.py").is_file():
        raise ValueError(f"DVIDEO detection directory not found: {path}")
    sys.path.insert(0, str(path.resolve()))
    from importer.ocel_importer import import_ocel_raw
    from detection.missing_object import load_cardinality_constraints
    from detection import missing_o2o_relation
    if not progress:
        missing_o2o_relation.tqdm = partial(missing_o2o_relation.tqdm, disable=True)
    return import_ocel_raw, load_cardinality_constraints, missing_o2o_relation.detect_missing_o2o_relation


def evaluate_pair(original_path: Path, masked_path: Path, constraints_path: Path, dvideo) -> dict:
    objects, original, masked, removed = ground_truth(load_json(original_path), load_json(masked_path))
    importer, loader, detector = dvideo
    constraints = loader(constraints_path)
    ocel = importer(masked_path)
    # The upstream importer omits object columns for an entirely empty log.
    if ocel.objects.empty:
        ocel.objects = ocel.objects.reindex(columns=["ocel:oid", "ocel:type"])
    detections = detector(ocel, constraints).to_dict(orient="records")
    result = score_detections(objects, original, masked, removed, detections,
                              [rule.model_dump() for rule in constraints])
    return {"log": original_path.stem, **result}


def aggregate(results: list[dict]) -> dict:
    return {level: {metric: sum(row[level][metric] for row in results)
                    for metric in ("tp", "fp", "fn")} for level in MEASUREMENTS}


def write_csv(path: Path, fields: list[str], rows, overwrite: bool):
    with path.open("w" if overwrite else "x", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--original-dir", type=Path, default=ROOT)
    parser.add_argument("--masked-dir", type=Path, default=ROOT / "masked_o2o_relations")
    parser.add_argument("--constraints-dir", type=Path, default=ROOT / "container_logistics_constraints")
    parser.add_argument("--dvideo-dir", type=Path, default=ROOT / "coopis-dvideo" / "detection")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "results_dvideo_o2o")
    parser.add_argument("--logs", type=int, nargs="+", help="Only evaluate these log numbers; default: all originals.")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--progress", action="store_true", help="Show DVIDEO's progress bars.")
    args = parser.parse_args(argv)
    try:
        from container_logistics_constraints.generate_o2o_constraints import discover_logs
        originals = discover_logs(args.original_dir)
        if args.logs:
            available = {int(p.stem.rsplit("_", 1)[1]) for p in originals}
            if set(args.logs) - available:
                raise ValueError(f"Original log numbers not found: {sorted(set(args.logs) - available)}")
            originals = [p for p in originals if int(p.stem.rsplit("_", 1)[1]) in args.logs]
        if not originals:
            raise ValueError("No ContainerLogistics originals found.")
        pairs = []
        for original in originals:
            masked = args.masked_dir / f"{original.stem}_masked{original.suffix}"
            constraints = args.constraints_dir / f"cardinality_constraints_{original.stem}.yaml"
            for path in (masked, constraints):
                if not path.is_file():
                    raise ValueError(f"Required input not found: {path}")
            pairs.append((original, masked, constraints))
        outputs = [args.output_dir / name for name in ("summary.csv", "errors.csv")]
        if not args.overwrite and any(p.exists() for p in outputs):
            raise ValueError("Report files already exist; use --overwrite or a different --output-dir.")
        dvideo = load_dvideo(args.dvideo_dir, args.progress)
        args.output_dir.mkdir(parents=True, exist_ok=True)
    except (OSError, ValueError, ImportError) as error:
        parser.error(str(error))

    results = []
    for original, masked, constraints in pairs:
        print(f"Evaluating {original.name} ...", flush=True)
        try:
            result = evaluate_pair(original, masked, constraints, dvideo)
            results.append(result)
            c, a = result["candidate_edges"], result["source_alerts"]
            print(f"  Candidate edges: TP={c['tp']} FP={c['fp']} FN={c['fn']}; "
                  f"source alerts: TP={a['tp']} FP={a['fp']} FN={a['fn']}", flush=True)
        except (OSError, ValueError, KeyError, TypeError) as error:
            print(f"ERROR {original.name}: {error}", file=sys.stderr, flush=True)
            return 1

    total = aggregate(results)
    csv_rows = []
    for result in [*results, {"log": "TOTAL", **total}]:
        row = {"log": result["log"]}
        for level in MEASUREMENTS:
            row.update({f"{level}_{k}": v for k, v in result[level].items()})
        csv_rows.append(row)
    try:
        write_csv(outputs[0], list(csv_rows[0]), csv_rows, args.overwrite)
        write_csv(outputs[1], ERROR_FIELDS,
                  ({"log": result["log"], **error} for result in results for error in result["errors"]),
                  args.overwrite)
    except OSError as error:
        print(f"ERROR writing reports: {error}. Close open CSV files or use another --output-dir.", file=sys.stderr)
        return 1
    print(f"\nEvaluated {len(results)}/{len(pairs)} logs")
    for level in ("candidate_edges", "source_alerts"):
        m = total[level]
        print(f"TOTAL {level}: TP={m['tp']} FP={m['fp']} FN={m['fn']}")
    print(f"Reports: {args.output_dir.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
