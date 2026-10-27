"""Generate one DVIDEO cardinality YAML per Container Logistics OCEL 2.0 log.

Run on reference logs BEFORE deleting/masking edges. Bounds are observed extrema,
not inferred business requirements. All objects of each type participate in the
counts, including objects with zero matching connections. Consequently a zero
minimum cannot detect missing edges for that relation type.

Only explicit, directed, qualified O2O relations between existing objects are
used. Duplicate relation rows are counted, matching DVIDEO's implementation.
Existing output files are preserved unless --overwrite is supplied.

Usage:
    python generate_o2o_constraints.py
    python generate_o2o_constraints.py --input-dir reference_logs --output-dir constraints
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
import re
import sys

import yaml


def infer_constraints(log: dict) -> tuple[list[dict], dict[str, int]]:
    """Compute min/max incoming/outgoing row counts per (source type, q, target type).

    Missing endpoints are reported and excluded: DVIDEO's forward matching also
    only counts relations whose endpoints exist. Blank qualifiers are separate
    quality issues and are not used to define expected relation types.
    """
    if not isinstance(log, dict):
        raise ValueError("Expected an OCEL 2.0 JSON object.")
    objects = log.get("objects")
    if not isinstance(objects, list):
        raise ValueError("Expected OCEL 2.0 with an 'objects' list.")

    object_types = {}
    for obj in objects:
        if not isinstance(obj, dict):
            raise ValueError("Each object must be a JSON object.")
        oid, object_type = obj.get("id"), obj.get("type")
        if not isinstance(oid, str) or not oid.strip():
            raise ValueError("Each object must have a non-empty string id.")
        if not isinstance(object_type, str) or not object_type.strip():
            raise ValueError(f"Object {oid!r} has no valid type.")
        if oid in object_types:
            raise ValueError(f"Duplicate object id: {oid!r}")
        object_types[oid] = object_type

    population = Counter(object_types.values())
    outgoing = defaultdict(Counter)
    incoming = defaultdict(Counter)
    seen_edges = set()
    stats = {
        "objects": len(objects),
        "relations": 0,
        "skipped_missing_targets": 0,
        "skipped_qualifiers": 0,
        "duplicate_relations": 0,
    }
    
    for obj in objects:
        relationships = obj.get("relationships", [])
        if not isinstance(relationships, list):
            raise ValueError(f"Object {obj['id']!r}: relationships must be a list.")
        for relation in relationships:
            if not isinstance(relation, dict):
                raise ValueError(f"Object {obj['id']!r}: invalid relationship.")
            target = relation.get("objectId")
            if not isinstance(target, str) or target not in object_types:
                stats["skipped_missing_targets"] += 1
                continue
            qualifier = relation.get("qualifier")
            if qualifier is None or (isinstance(qualifier, str) and not qualifier.strip()):
                stats["skipped_qualifiers"] += 1
                continue
            if not isinstance(qualifier, str):
                raise ValueError(f"Object {obj['id']!r}: qualifier must be a string.")

            # Preserve direction and exact qualifier spelling; do not infer reverse edges.
            key = (obj["type"], qualifier, object_types[target])
            outgoing[key][obj["id"]] += 1
            incoming[key][target] += 1
            edge = (obj["id"], qualifier, target)
            stats["duplicate_relations"] += int(edge in seen_edges)
            seen_edges.add(edge)
            stats["relations"] += 1

    rules = []
    for key in sorted(outgoing):
        source_type, qualifier, target_type = key
        out_counts, in_counts = outgoing[key], incoming[key]
        rules.append({
            "source_type": source_type,
            "qualifier": qualifier,
            "target_type": target_type,
            "min_outgoing": (
                min(out_counts.values()) if len(out_counts) == population[source_type] else 0
            ),
            "max_outgoing": max(out_counts.values()),
            "min_incoming": (
                min(in_counts.values()) if len(in_counts) == population[target_type] else 0
            ),
            "max_incoming": max(in_counts.values()),
        })
    stats["rules"] = len(rules)
    stats["positive_outgoing_minima"] = sum(rule["min_outgoing"] > 0 for rule in rules)
    return rules, stats


def write_constraints(
    input_path: Path, output_path: Path, *, overwrite: bool = False,
) -> dict[str, int] | None:
    """Write a top-level YAML rule list accepted by load_cardinality_constraints.

    Return generation statistics, or None when preserving an existing output.
    """
    if input_path.resolve() == output_path.resolve():
        raise ValueError("Output must not replace the input log.")
    if output_path.exists() and not overwrite:
        return None
    raw = input_path.read_bytes()
    rules, stats = infer_constraints(json.loads(raw.decode("utf-8-sig")))
    header = (
        f"# Reference log: {json.dumps(input_path.name)}\n"
        "# Observed min/max relation-row counts, including zero-degree objects.\n"
        "# Generate before masking edges; these are not universal domain rules.\n"
        "# A zero min_outgoing disables missing-edge detection for that rule.\n"
        "# DVIDEO's missing-O2O detector currently does not check min_incoming.\n"
        f"# Excluded missing targets: {stats['skipped_missing_targets']}; "
        f"blank qualifiers: {stats['skipped_qualifiers']}; "
        f"duplicate rows counted: {stats['duplicate_relations']}.\n\n"
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    # Exclusive creation also protects existing files if another process creates one.
    with output_path.open("w" if overwrite else "x", encoding="utf-8", newline="\n") as file:
        file.write(header)
        yaml.safe_dump(rules, file, sort_keys=False, allow_unicode=True)
    return stats


def discover_logs(input_dir: Path) -> list[Path]:
    """Find the numbered Container Logistics JSON and JSONOCEL files, in order."""
    numbered = []
    used_numbers = set()
    for path in input_dir.iterdir():
        match = re.fullmatch(r"ContainerLogistics_(\d+)\.(json|jsonocel)", path.name)
        if not path.is_file() or not match:
            continue
        number = int(match.group(1))
        if number in used_numbers:
            raise ValueError(f"Multiple input logs have ContainerLogistics number {number}.")
        used_numbers.add(number)
        numbered.append((number, path))
    return [path for _, path in sorted(numbered)]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input-dir", type=Path, default=Path("."))
    parser.add_argument("--output-dir", type=Path, default=None,
                        help="Default: <input-dir>/container_logistics_constraints")
    parser.add_argument("--overwrite", action="store_true", help="Replace existing constraint files.")
    args = parser.parse_args(argv)
    output_dir = args.output_dir if args.output_dir is not None else args.input_dir / "container_logistics_constraints"
    try:
        logs = discover_logs(args.input_dir)
        if not logs:
            raise ValueError(f"No ContainerLogistics_<number>.json/.jsonocel logs in {args.input_dir}")
    except (OSError, ValueError) as error:
        parser.error(str(error))
        

    failures = 0
    for path in logs:
        output_path = output_dir / f"cardinality_constraints_{path.stem}.yaml"
        try:
            stats = write_constraints(path, output_path, overwrite=args.overwrite)
        except (OSError, ValueError) as error:
            print(f"ERROR {path.name}: {error}", file=sys.stderr)
            failures += 1
            continue
        if stats is None:
            print(f"SKIP {output_path} (already exists; use --overwrite to replace)")
            continue
        print(f"WRITE {output_path}: {stats['rules']} rules, "
              f"{stats['positive_outgoing_minima']} with positive outgoing minima")
        for field in ("skipped_missing_targets", "skipped_qualifiers", "duplicate_relations"):
            if stats[field]:
                print(f"  {path.name}: {field}={stats[field]}", file=sys.stderr)
        if not stats["rules"]:
            print(f"  {path.name}: no usable explicit O2O relations; wrote an empty list.", file=sys.stderr)
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
