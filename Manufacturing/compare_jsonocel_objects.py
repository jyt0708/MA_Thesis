#!/usr/bin/env python3
"""Print and count object attributes in two JSONOCEL files."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping


def load_objects(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8-sig") as file:
        document = json.load(file)

    objects = document.get("ocel:objects") if isinstance(document, dict) else None
    if not isinstance(objects, dict):
        raise ValueError(f"{path}: 'ocel:objects' must be an object keyed by IDs")
    return objects


def attribute_names(obj: Mapping[str, Any]) -> Iterable[str]:
    """Yield every attribute occurrence from an object's ``ocel:ovmap``."""
    ovmap = obj.get("ocel:ovmap", {})
    if ovmap is None:
        return
    if not isinstance(ovmap, dict):
        raise ValueError("ocel:ovmap must be an object or null")

    for name, value in ovmap.items():
        if name != "list":
            yield str(name)
        elif isinstance(value, list) and value and all(
            isinstance(item, dict)
            and set(item) == {"name", "value"}
            and isinstance(item["name"], str)
            for item in value
        ):
            for item in value:
                yield item["name"]
        else:
            yield "list"


def count_attributes(objects: Mapping[str, Any]) -> Counter[str]:
    counts: Counter[str] = Counter()

    for object_id, obj in objects.items():
        if not isinstance(obj, dict):
            raise ValueError(f"object {object_id!r} must be a JSON object")
        counts.update(attribute_names(obj))

    return counts


def print_attribute_counts(label: str, path: Path) -> None:
    counts = count_attributes(load_objects(path))

    print(f"{label}: {path}")
    if counts:
        for attribute, count in sorted(counts.items()):
            print(f"  {attribute}: {count}")
    else:
        print("  No object attributes")

    print(f"  Total object attributes: {sum(counts.values())}")
    print(f"  Unique object attribute names: {len(counts)}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Print each object attribute and count the total object "
            "attributes in two JSONOCEL files."
        )
    )
    parser.add_argument("first", type=Path, help="First .jsonocel file")
    parser.add_argument("second", type=Path, help="Second .jsonocel file")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    print_attribute_counts("First OCEL", args.first)
    print()
    print_attribute_counts("Second OCEL", args.second)


if __name__ == "__main__":
    main()
