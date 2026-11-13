"""Mask O2O relation rows while preserving target-ID support in the remaining log.

Inputs must contain OCEL 2.0 JSON.
The original extension and all content other than sampled object
relationships are preserved. Shuffle rows using seed 42 by default, then remove
a row only if its target ID still occurs in another remaining O2O row. 
Aim for floor(number_of_o2o_rows / 10) removals per file, or the maximum allowed by this
constraint if fewer are possible. 
Singleton target IDs are never masked.

Examples:
    .\venv\Scripts\python.exe -X utf8 mask_o2o_relations.py --domain logistics
    .\venv\Scripts\python.exe -X utf8 mask_o2o_relations.py --domain manufacturing ContainerLogistics_1.json --overwrite
    .\venv\Scripts\python.exe -X utf8 mask_o2o_relations.py "ContainerLogistics_*.json" "ContainerLogistics_*.jsonocel" --domain logistics --overwrite
    
With no inputs, process only ContainerLogistics_1.json and ContainerLogistics_2.jsonocel.
This script only masks relations; it does not load constraints or run detection.
This script assumes only logistics and manufacturing domain (can be adjusted afterwards)

Existing outputs require --overwrite.
"""

from __future__ import annotations

import argparse
from collections import Counter
from copy import deepcopy
import glob
import json
from pathlib import Path
import random
import sys

ROOT = Path(__file__).resolve().parents[2]
LOGISTICS_DIR = ROOT / "Logistics"
MANUFACTURING_DIR = ROOT / "Manufacturing"
DEFAULT_INPUTS = [LOGISTICS_DIR / "ContainerLogistics_1.json", LOGISTICS_DIR / "ContainerLogistics_2.jsonocel"]

# --- Domain Configuration ---
# Extend this dictionary to add new domains and their associated qualifiers.
DOMAIN_QUALIFIERS = {
    "logistics": ["CR contains HU", "CR for TD", "TD for CO", "TR loads CR"],
    "manufacturing": None,  # Consider all qualifiers for this domain.
}

def mask_relations(raw: dict, seed: int = 42, target_qualifiers: list[str] | None = None) -> tuple[dict, list[dict], int]:
    """Return a masked copy, removed rows with original indices, and total rows."""
    if (not isinstance(raw, dict)
            or not isinstance(raw.get("objects"), list)
            or not isinstance(raw.get("events"), list)):
        raise ValueError("Expected OCEL 2.0 JSON with 'objects' and 'events' lists.")

    positions = []
    remaining_counts = Counter()
    total_rows = 0

    # First pass: Count all targets and identify eligible rows
    for object_index, obj in enumerate(raw["objects"]):
        if not isinstance(obj, dict) or "id" not in obj or "type" not in obj:
            raise ValueError("Each object must have an 'id' and 'type'.")
        relations = obj.get("relationships", [])
        if not isinstance(relations, list):
            raise ValueError(f"Object {obj['id']!r}: relationships must be a list.")
        for relation_index, relation in enumerate(relations):
            if (not isinstance(relation, dict)
                    or "objectId" not in relation or "qualifier" not in relation):
                raise ValueError(f"Object {obj['id']!r}: invalid O2O relationship.")
            if not isinstance(relation["objectId"], str) or not relation["objectId"].strip():
                raise ValueError(f"Object {obj['id']!r}: target ID must be a non-empty string.")
            
            total_rows += 1
            remaining_counts[relation["objectId"]] += 1
            
            # If target_qualifiers is None, all rows are eligible. 
            # Otherwise, only rows matching the specified qualifiers are eligible.
            if target_qualifiers is None or relation["qualifier"] in target_qualifiers:
                positions.append((object_index, relation_index))

    desired_count = len(positions) // 10
    
    random.Random(seed).shuffle(positions)
    selected = set()
    
    for object_index, relation_index in positions:
        if len(selected) >= desired_count:
            break
        target = raw["objects"][object_index]["relationships"][relation_index]["objectId"]
        if remaining_counts[target] > 1:
            selected.add((object_index, relation_index))
            remaining_counts[target] -= 1
            
    masked = deepcopy(raw)
    removed = []
    for object_index, obj in enumerate(masked["objects"]):
        if "relationships" not in obj:
            continue
        kept = []
        for relation_index, relation in enumerate(obj["relationships"]):
            if (object_index, relation_index) in selected:
                removed.append({
                    "source_id": obj["id"],
                    "object_index": object_index,
                    "relationship_index": relation_index,
                    "relationship": relation,
                })
            else:
                kept.append(relation)
        obj["relationships"] = kept
        
    return masked, removed, total_rows


def output_path(input_path: Path, output_dir: Path) -> Path:
    """Returns the path for the masked file."""
    return output_dir / f"{input_path.stem}_masked{input_path.suffix}"


def process_file(
    input_path: str | Path,
    output_dir: str | Path,
    *,
    seed: int = 42,
    overwrite: bool = False,
    qualifiers: list[str] | None = None,
) -> dict:
    """Save the masked log and return counts and its output path."""
    input_path, output_dir = Path(input_path), Path(output_dir)
    if input_path.suffix.lower() not in {".json", ".jsonocel"}:
        raise ValueError(f"Expected a .json or .jsonocel input: {input_path}")
    
    masked_path = output_path(input_path, output_dir)
    
    if masked_path.resolve() == input_path.resolve():
        raise ValueError(f"Output would replace an input: {masked_path}")
    if masked_path.exists() and not overwrite:
        raise FileExistsError(f"Output already exists: {masked_path}; use --overwrite.")

    with input_path.open(encoding="utf-8-sig") as file:
        raw = json.load(file)
    
    masked, removed, total = mask_relations(raw, seed, target_qualifiers=qualifiers)
    
    output_dir.mkdir(parents=True, exist_ok=True)
    mode = "w" if overwrite else "x"
    with masked_path.open(mode, encoding="utf-8") as file:
        json.dump(masked, file, indent=2, ensure_ascii=True)
        file.write("\n")
        
    target_counts = Counter(relation["objectId"] for obj in raw["objects"]
                            for relation in obj.get("relationships", []))
                            
    result = {
        "masked_path": masked_path,
        "total_rows": total,
        "requested_rows": len(removed),
        "masked_rows": len(removed),
        "masked_fraction": len(removed) / total if total else 0.0,
        "singleton_target_ids": sum(count == 1 for count in target_counts.values()),
    }
    
    print(f"{input_path.name}: masked {len(removed)}/{total} O2O rows "
          f"({result['masked_fraction']:.2%}); "
          f"excluded {result['singleton_target_ids']} singleton target IDs")
    
    print(f"Masked log: {masked_path}")
    return result


def expand_inputs(patterns: list[str]) -> list[Path]:
    """Expand globs in Python (including on Windows), preserving argument order."""
    paths = []
    seen = set()
    for pattern in patterns:
        matches = [Path(pattern)] if Path(pattern).is_file() else [
            Path(match) for match in sorted(glob.glob(pattern))
        ]
        if not matches:
            raise ValueError(f"No input files match: {pattern}")
        for path in matches:
            if not path.is_file() or path.suffix.lower() not in {
                ".jsonocel", ".json"
            }:
                raise ValueError(f"Expected an OCEL JSON file: {path}")
            if path.resolve() not in seen:
                paths.append(path)
                seen.add(path.resolve())
    return paths


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("inputs", nargs="*", help="Paths or quoted globs; default: ContainerLogistics 1 and 2 only.")
    parser.add_argument("--domain", choices=list(DOMAIN_QUALIFIERS.keys()), default=None)
    parser.add_argument("--output-dir", type=Path, default=None, 
                        help="Output directory. Defaults to masked_o2o_relations[_domain] under Manufacturing for manufacturing, otherwise Logistics.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args(argv)

    # Handle default output directory based on domain
    if args.output_dir is None:
        dir_name = "masked_o2o_relations"
        if args.domain:
            dir_name += f"_{args.domain}"
        output_root = MANUFACTURING_DIR if args.domain == "manufacturing" else LOGISTICS_DIR
        args.output_dir = output_root / dir_name
    
    try:
        paths = expand_inputs(args.inputs or [str(path) for path in DEFAULT_INPUTS])
        
        outputs = [output_path(path, args.output_dir).resolve() for path in paths]
        
        if len(outputs) != len(set(outputs)):
            raise ValueError("Inputs have colliding output names; use separate output directories.")
        if set(outputs) & {path.resolve() for path in paths}:
            raise ValueError("An output would replace another input in this batch.")
    
    except ValueError as error:
        parser.error(str(error))
        
    target_qualifiers = None
    if args.domain:
        target_qualifiers = DOMAIN_QUALIFIERS[args.domain]

    failures = 0
    for path in paths:
        try:
            process_file(
                path, 
                args.output_dir, 
                seed=args.seed, 
                overwrite=args.overwrite,
                qualifiers=target_qualifiers,
            )
        except (OSError, ValueError, KeyError, TypeError) as error:
            print(f"ERROR {path}: {error}", file=sys.stderr)
            failures += 1
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())