"""Convert an OCEL JSON/JSONOCEL file into tabular CSV views.

Outputs
-------
event_level_<suffix>.csv
    One row per event, including event attributes and the number of relations
    to retained objects. Events without retained objects are included.
object_level_<suffix>.csv
    One row per object, including object attributes and lifecycle context.
    Rows with a present but null, empty, or whitespace-only attribute are omitted.
    Rows with a present timing attribute that is not a string beginning with
    an ASCII digit (0-9) are also omitted. This is not full timestamp validation.
    E2O and O2O rows reference only objects retained in this view.
    Sensor points are retained in order as a JSON array in
    ``obj_attr__stream:point`` (including single-point bundles).
    For Manufacturing, nested timestamps outside arrays also get individual
    ``obj_attr__<parent>__<field>`` columns and matching ``__exists`` flags.
timestamp_sample_level_<suffix>.csv (Manufacturing only)
    One row per array item containing valid timestamps, linked by object_id.
    Timestamp targets use ``obj_attr__<path>``; other scalar sample fields use
    ``sample_attr__<path>``. Array positions are metadata, not target columns.
    Nested arrays get separate rows identified by array_item_path.
e2o_level_<suffix>.csv
    One row per event-to-object relation, including event attributes and
    the object's position in its event lifecycle.
o2o_level_<suffix>.csv
    One row per object-to-object relation. If the source log has no explicit
    O2O relations (as in OCEL 1.0), relations are inferred from objects that
    co-occur in at least one event.

The converter accepts both the legacy OCEL JSON structure used by the supplied
log (``ocel:events`` / ``ocel:objects`` dictionaries) and the OCEL 2.0 JSON
structure (``events`` / ``objects`` lists).
Existing output files are skipped; only missing CSV files are created.
Manufacturing extraction is enabled for inputs beneath a Manufacturing folder,
or explicitly with ``manufacturing=True``. Timestamp text, timezone offsets,
and fractional seconds are preserved. Full date-and-time strings are detected
by shape and calendar validity, not by field name. Date-only strings, numeric
epochs, nulls, and invalid timestamps are not extracted.

Original object JSON values remain available. Exclude these source columns
and equivalent timestamps when training on an extracted target to avoid leakage.

"""

from __future__ import annotations

import json
import re
from collections import defaultdict
from datetime import datetime
from functools import lru_cache
from itertools import combinations
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd


def load_json_with_literal_string_escapes(file: Any) -> Any:
    """    
    Reads a JSON file and preserves backslash sequences inside quoted
    strings as literal text. 
    The exception is "\"", which is still interpreted as an embedded 
    double quote. Finally, the modified text is converted into a 
    Python dictionary or list using json.loads().
    """
    source = file.read()
    protected: list[str] = []
    in_string = False
    index = 0

    while index < len(source):
        char = source[index]

        if char == '"':
            in_string = not in_string
            protected.append(char)
            index += 1
            continue

        if in_string and char == "\\":
            if index + 1 < len(source) and source[index + 1] == '"':
                protected.extend(("\\", '"'))
                index += 2
            else:
                protected.extend(("\\", "\\"))
                index += 1
            continue

        protected.append(char)
        index += 1

    return json.loads("".join(protected))


def csv_value(value: Any) -> Any:
    """Keep scalar values as-is and serialize nested values deterministically."""
    if value is None:
        return np.nan
    if isinstance(value, (dict, list, tuple)):
        return json.dumps(value, ensure_ascii=False, sort_keys=True)
    return value


def clean_activity(value: Any) -> Any:
    """Replace non-ASCII/control characters in an activity with one space."""
    if not isinstance(value, str):
        return value
    cleaned = "".join(char if 32 <= ord(char) <= 126 else " " for char in value)
    return " ".join(cleaned.split())


def join_values(values: Iterable[Any], separator: str = "|") -> Any:
    """Join unique, non-missing values while preserving their order."""
    result = []
    seen = set()
    for value in values:
        if value is None or (isinstance(value, float) and np.isnan(value)):
            continue
        text = str(value)
        if text not in seen:
            seen.add(text)
            result.append(text)
    return separator.join(result) if result else np.nan


def attributes_from_legacy_ovmap(ovmap: Any) -> dict[str, Any]:
    """Flatten legacy attributes, retaining every point in each sensor bundle."""
    if not isinstance(ovmap, dict):
        return {}

    items = ovmap.get("list")
    if not isinstance(items, list):
        return {str(key): csv_value(value) for key, value in ovmap.items()}

    result: dict[str, Any] = {}
    stream_points: list[Any] = []
    for item in items:
        if not isinstance(item, dict):
            continue

        # Encoding used by data objects: {"name": ..., "value": ...}
        if "name" in item and "value" in item:
            result[str(item["name"])] = csv_value(item["value"])
            continue

        # Encoding used by stream objects: {"stream:name": ...}, etc.
        # Sensor points are a sequence, not updates to a single attribute.
        # Other repeated attributes retain the existing last-value behavior.
        for key, value in item.items():
            if key == "stream:point":
                stream_points.append(value)
            else:
                result[str(key)] = csv_value(value)

    if stream_points:
        # Serialize once so the array contains objects, not JSON strings.
        result["stream:point"] = csv_value(stream_points)

    return result


def latest_ocel2_attributes(attributes: Any) -> dict[str, Any]:
    """Select the latest value for each timestamped OCEL 2.0 attribute."""
    if not isinstance(attributes, list):
        return {}

    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for attribute in attributes:
        if isinstance(attribute, dict) and "name" in attribute:
            grouped[str(attribute["name"])].append(attribute)

    result = {}
    for name, values in grouped.items():
        latest = max(values, key=lambda value: str(value.get("time", "")))
        result[name] = csv_value(latest.get("value"))
    return result


_NESTED_TIMESTAMP_PATTERN = re.compile(
    r"[0-9]{4}-[0-9]{2}-[0-9]{2}[T ][0-9]{2}:[0-9]{2}:[0-9]{2}"
    r"(?:\.[0-9]{1,9})?(?:Z| ?[+-][0-9]{2}:?[0-9]{2})?"
)


@lru_cache(maxsize=8192)
def _is_timestamp_text(value: str) -> bool:
    """Validate repeated sensor timestamps once, after a cheap shape check."""
    if not _NESTED_TIMESTAMP_PATTERN.fullmatch(value):
        return False
    try:
        datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return False
    return True


def _nested_path(parts: tuple) -> str:
    """Keep paths readable while distinguishing keys from separators/indices."""
    return "__".join(
        f"[{part}]" if isinstance(part, int) else (
            str(part).replace("%", "%25").replace("__", "%5F%5F")
            .replace("[", "%5B").replace("]", "%5D")
        )
        for part in parts
    )


def extract_manufacturing_timestamps(objects: list[dict]) -> list[dict]:
    """Add scalar timestamp targets to objects and return array sample rows.

    Normalization has already serialized nested attributes as JSON strings.
    Decode them for traversal without modifying their stored source values.
    Each array item is a separate sample; all of its timestamps share that
    row, including timestamps nested inside its metadata dictionaries.
    """
    samples = []
    for obj in objects:
        extracted = {}

        def visit(value, path, location, sample=None):
            if isinstance(value, dict):
                for key, child in value.items():
                    visit(child, path + (key,), location + (key,), sample)
            elif isinstance(value, list):
                for index, item in enumerate(value):
                    row = {
                        "object_id": obj["id"],
                        "object_type": obj["type"],
                        "array_path": _nested_path(path),
                        "array_item_path": _nested_path(location + (index,)),
                        "sample_index": index,
                    }
                    visit(item, path, location + (index,), row)
                    if any(key.startswith("obj_attr__") for key in row):
                        samples.append(row)
            else:
                name = _nested_path(path)
                if isinstance(value, str) and _is_timestamp_text(value):
                    if sample is None:
                        if name in obj["attributes"]:
                            raise ValueError(
                                f"Extracted timestamp conflicts with attribute: {name}"
                            )
                        extracted[name] = value
                    else:
                        sample[f"obj_attr__{name}"] = value
                        sample[f"obj_attr__{name}__exists"] = 1
                elif sample is not None:
                    sample[f"sample_attr__{name}"] = csv_value(value)

        for name, value in obj["attributes"].items():
            if isinstance(value, str) and value.lstrip().startswith(("{", "[")):
                try:
                    value = json.loads(value)
                except json.JSONDecodeError:
                    continue
            if isinstance(value, (dict, list)):
                visit(value, (name,), (name,))
        obj["attributes"].update(extracted)
    return samples


def timestamp_sample_view(
    samples: list[dict], retained_object_ids: set[str]
) -> pd.DataFrame:
    """Keep samples of retained objects and complete their existence flags."""
    columns = [
        "object_id", "object_type", "array_path", "array_item_path", "sample_index",
    ]
    frame = pd.DataFrame(samples)
    if frame.empty:
        return pd.DataFrame(columns=columns)
    frame = frame.loc[frame["object_id"].isin(retained_object_ids)].copy()
    for column in frame.columns:
        if column.startswith("obj_attr__") and column.endswith("__exists"):
            frame[column] = frame[column].fillna(0).astype("int8")
    frame = frame.reindex(columns=columns + sorted(set(frame.columns) - set(columns)))
    # Traversal order preserves source array order, including nested arrays.
    return frame.reset_index(drop=True)


def normalize_ocel(
    ocel: dict[str, Any],
) -> tuple[list[dict], list[dict], str]:
    """
    Convert OCEL 1.0 or OCEL 2.0 JSON into the same internal format.

    Returns:
        events: Normalized event records.
        objects: Normalized object records.
        version: "OCEL 1.0" or "OCEL 2.0".
    """

    def normalize_ocel2_attributes(raw_attributes: list[dict]) -> dict:
        """Convert the OCEL 2.0 attribute list into a dictionary."""
        attributes = {}

        for attribute in raw_attributes or []:
            name = attribute.get("name")

            if name is not None:
                attributes[str(name)] = csv_value(
                    attribute.get("value", np.nan)
                )

        return attributes

    def normalize_relationships(raw_relationships: list[dict]) -> list[dict]:
        """Convert OCEL 2.0 relationships into the internal format."""
        relationships = []

        for relationship in raw_relationships or []:
            object_id = relationship.get("objectId")

            if object_id is None:
                continue

            qualifier = relationship.get("qualifier", np.nan)

            if qualifier is None:
                qualifier = np.nan

            relationships.append(
                {
                    "objectId": str(object_id),
                    "qualifier": qualifier,
                }
            )

        return relationships

    # ---------------------------------------------------------
    # OCEL 1.0
    # ---------------------------------------------------------
    if "ocel:events" in ocel and "ocel:objects" in ocel:
        objects = []

        for object_id, raw_object in ocel["ocel:objects"].items():
            objects.append(
                {
                    "id": str(object_id),
                    "type": raw_object.get("ocel:type", np.nan),
                    "attributes": attributes_from_legacy_ovmap(
                        raw_object.get("ocel:ovmap", {})
                    ),
                    # OCEL 1.0 does not contain explicit
                    # object-to-object relationships.
                    "relationships": [],
                }
            )

        events = []

        for event_id, raw_event in ocel["ocel:events"].items():
            events.append(
                {
                    "id": str(event_id),
                    "type": clean_activity(
                        raw_event.get("ocel:activity", np.nan)
                    ),
                    "time": raw_event.get("ocel:timestamp", np.nan),
                    "attributes": {
                        str(key): csv_value(value)
                        for key, value
                        in raw_event.get("ocel:vmap", {}).items()
                    },
                    "relationships": [
                        {
                            "objectId": str(object_id),
                            "qualifier": np.nan,
                        }
                        for object_id
                        in raw_event.get("ocel:omap", [])
                    ],
                }
            )

        return events, objects, "OCEL 1.0"

    # ---------------------------------------------------------
    # OCEL 2.0
    # ---------------------------------------------------------
    if "events" in ocel and "objects" in ocel:
        objects = []

        for raw_object in ocel["objects"]:
            objects.append(
                {
                    "id": str(raw_object.get("id")),
                    "type": raw_object.get("type", np.nan),
                    "attributes": normalize_ocel2_attributes(
                        raw_object.get("attributes", [])
                    ),
                    "relationships": normalize_relationships(
                        raw_object.get("relationships", [])
                    ),
                }
            )

        events = []

        for raw_event in ocel["events"]:
            events.append(
                {
                    "id": str(raw_event.get("id")),
                    "type": clean_activity(
                        raw_event.get("type", np.nan)
                    ),
                    "time": raw_event.get("time", np.nan),
                    "attributes": normalize_ocel2_attributes(
                        raw_event.get("attributes", [])
                    ),
                    "relationships": normalize_relationships(
                        raw_event.get("relationships", [])
                    ),
                }
            )

        return events, objects, "OCEL 2.0"

    raise ValueError(
        "Unsupported OCEL format: expected OCEL 1.0 or OCEL 2.0 JSON."
    )


def rename_object_ids_by_type(
    events: list[dict], objects: list[dict]
) -> None:
    """Rename object IDs to ``<object_type>_<number>`` in-place.

    Numbering starts at 1 independently for each object type.  All event-to-
    object and explicit object-to-object references are updated with the same
    mapping so the normalized OCEL remains internally consistent.  An event
    attribute is updated only when its complete string value exactly matches
    an original object ID.
    """
    counters: dict[str, int] = defaultdict(int)
    id_mapping: dict[str, str] = {}

    for obj in objects:
        original_id = str(obj["id"])
        object_type = str(obj["type"])
        counters[object_type] += 1
        renamed_id = f"{object_type}_{counters[object_type]}"
        id_mapping[original_id] = renamed_id
        obj["id"] = renamed_id

    for event in events:
        for relation in event.get("relationships", []):
            original_id = str(relation.get("objectId"))
            if original_id in id_mapping:
                relation["objectId"] = id_mapping[original_id]

        for attribute_name, value in event.get("attributes", {}).items():
            if isinstance(value, str) and value in id_mapping:
                event["attributes"][attribute_name] = id_mapping[value]

    for obj in objects:
        for relation in obj.get("relationships", []):
            original_id = str(relation.get("objectId"))
            if original_id in id_mapping:
                relation["objectId"] = id_mapping[original_id]


def abbreviate_event_type(event_type: Any) -> str:
    """Create a compact, CSV-safe abbreviation for an event type."""
    text = str(event_type)
    text = re.sub(r"(?<=[a-z0-9])(?=[A-Z])", " ", text)
    words = re.findall(r"[A-Za-z0-9]+", text)
    if not words:
        return "event"
    if len(words) == 1:
        return words[0].lower()
    return "".join(word[0].lower() for word in words)


def rename_event_ids_by_type(
    events: list[dict], objects: list[dict]
) -> None:
    """Rename event IDs to ``<event_type_abbreviation>_<number>`` in-place.

    Counters are maintained per abbreviation.  Exact string matches to old
    event IDs in event or object attributes are updated with the same mapping.
    """
    counters: dict[str, int] = defaultdict(int)
    id_mapping: dict[str, str] = {}

    for event in events:
        original_id = str(event["id"])
        abbreviation = abbreviate_event_type(event["type"])
        counters[abbreviation] += 1
        renamed_id = f"{abbreviation}_{counters[abbreviation]}"
        id_mapping[original_id] = renamed_id
        event["id"] = renamed_id

    for event in events:
        for attribute_name, value in event.get("attributes", {}).items():
            if isinstance(value, str) and value in id_mapping:
                event["attributes"][attribute_name] = id_mapping[value]

    for obj in objects:
        for attribute_name, value in obj.get("attributes", {}).items():
            if isinstance(value, str) and value in id_mapping:
                obj["attributes"][attribute_name] = id_mapping[value]


def safe_sort(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    """Sort a DataFrame only when it contains rows and the requested columns."""
    if df.empty:
        return df.reset_index(drop=True)
    available = [column for column in columns if column in df.columns]
    return df.sort_values(available).reset_index(drop=True) if available else df


def convert_ocel_to_tabular_views(
    json_path: str | Path,
    output_dir: str,
    output_suffix: str | int | None = None,
    *,
    manufacturing: bool | None = None,
) -> dict[str, pd.DataFrame]:
    """Create missing event-, object-, E2O-, and O2O-level CSV files.

    Return all views computed from the input, even when existing CSVs are skipped.
    Manufacturing inputs also extract nested timestamps and export array samples.
    Override automatic input-folder detection with the manufacturing keyword.
    """
    json_path = Path(json_path)
    if manufacturing is None:
        manufacturing = any(
            part.casefold() == "manufacturing"
            for part in json_path.resolve().parent.parts
        )
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if output_suffix is None:
        output_suffix = json_path.stem
    output_suffix = str(output_suffix)

    with json_path.open("r", encoding="utf-8") as file:
        raw_ocel = load_json_with_literal_string_escapes(file)
        events, objects, detected_format = normalize_ocel(raw_ocel)
    del raw_ocel
    rename_object_ids_by_type(events, objects)
    rename_event_ids_by_type(events, objects)
    timestamp_samples = extract_manufacturing_timestamps(objects) if manufacturing else []

    object_by_id = {obj["id"]: obj for obj in objects}
    object_type_by_id = {obj["id"]: obj["type"] for obj in objects}

    # Infer attribute applicability from attributes observed for each type.
    attributes_by_type: dict[str, set[str]] = defaultdict(set)
    for obj in objects:
        attributes_by_type[str(obj["type"])].update(obj["attributes"])
    all_object_attributes = sorted(
        {name for names in attributes_by_type.values() for name in names}
    )
    all_event_attributes = sorted(
        {name for event in events for name in event["attributes"]}
    )
    
    if len(all_event_attributes)==0:
        print(f"{json_path}: Event does not have any attribute.")

    # Build chronologically sorted object lifecycles.
    object_events: dict[str, list[dict]] = defaultdict(list)
    for event in events:
        seen_in_event = set()
        for relation in event["relationships"]:
            object_id = str(relation.get("objectId"))
            if object_id in object_by_id and object_id not in seen_in_event:
                object_events[object_id].append(event)
                seen_in_event.add(object_id)
    for lifecycle in object_events.values():
        lifecycle.sort(key=lambda event: (str(event["time"]), event["id"]))

    # ------------------------------------------------------------------
    # 1. Object-level view
    # ------------------------------------------------------------------
    object_rows = []
    for obj in objects:
        object_id = obj["id"]
        object_type = obj["type"]
        observed_attributes = obj["attributes"]
        lifecycle = object_events.get(object_id, [])

        row = {"object_id": object_id, "object_type": object_type}
        for attribute_name in all_object_attributes:
            exists = attribute_name in observed_attributes
            row[f"obj_attr__{attribute_name}__exists"] = int(exists)
            row[f"obj_attr__{attribute_name}"] = (
                observed_attributes[attribute_name] if exists else ""
            )

        row["number_of_events"] = len(lifecycle)
        row["first_activity"] = lifecycle[0]["type"] if lifecycle else np.nan
        row["last_activity"] = lifecycle[-1]["type"] if lifecycle else np.nan
        row["first_event_time"] = lifecycle[0]["time"] if lifecycle else np.nan
        row["last_event_time"] = lifecycle[-1]["time"] if lifecycle else np.nan
        row["activity_sequence"] = (
            ">".join(str(event["type"]) for event in lifecycle)
            if lifecycle
            else np.nan
        )
        object_rows.append(row)

    object_df = safe_sort(
        pd.DataFrame(object_rows), ["object_type", "object_id"]
    )
    if not object_df.empty:
        invalid_rows = pd.Series(False, index=object_df.index)
        for attribute_name in all_object_attributes:
            values = object_df[f"obj_attr__{attribute_name}"]
            invalid_values = values.isna() | values.map(
                lambda value: isinstance(value, str) and not value.strip()
            )
            if attribute_name == "timing":
                # Cheap prefix check; no datetime parsing or string conversion.
                invalid_values |= ~values.map(
                    lambda value: isinstance(value, str)
                    and bool(value) and "0" <= value[0] <= "9"
                )
            invalid_rows |= (
                object_df[f"obj_attr__{attribute_name}__exists"].eq(1)
                & invalid_values
            )
        object_df = object_df.loc[~invalid_rows].reset_index(drop=True)

    # All relationship views must use the objects that survived validation.
    retained_object_ids = set(object_df["object_id"]) if not object_df.empty else set()

    # Lifecycle position of every E2O pair.
    lifecycle_info = {}
    for object_id, lifecycle in object_events.items():
        for index, event in enumerate(lifecycle):
            lifecycle_info[(event["id"], object_id)] = {
                "position": index + 1,
                "previous_activity": (
                    lifecycle[index - 1]["type"] if index > 0 else np.nan
                ),
                "next_activity": (
                    lifecycle[index + 1]["type"]
                    if index + 1 < len(lifecycle)
                    else np.nan
                ),
            }

    # ------------------------------------------------------------------
    # 2. Event-level and event-to-object-level views
    # ------------------------------------------------------------------
    event_columns = [
        "event_id", "event_activity", "event_timestamp", "event_relation_count",
        *[f"event_attr__{name}" for name in all_event_attributes],
    ]
    event_rows = []
    e2o_rows = []
    for event in events:
        relationships = [
            relation for relation in event["relationships"]
            if str(relation.get("objectId")) in retained_object_ids
        ]
        event_row = {
            "event_id": event["id"],
            "event_activity": event["type"],
            "event_timestamp": event["time"],
            "event_relation_count": len(relationships),
        }
        for attribute_name in all_event_attributes:
            event_row[f"event_attr__{attribute_name}"] = event["attributes"].get(
                attribute_name, np.nan
            )
        event_rows.append(event_row)
        for relation in relationships:
            object_id = str(relation.get("objectId"))
            lifecycle = lifecycle_info.get((event["id"], object_id), {})
            row = {
                "event_id": event["id"],
                "event_activity": event["type"],
                "event_timestamp": event["time"],
                "object_id": object_id,
                "object_type": object_type_by_id.get(object_id, np.nan),
                "e2o_qualifier": relation.get("qualifier", np.nan),
                "event_relation_count": len(relationships),
                "object_event_count": len(object_events.get(object_id, [])),
                "object_event_position": lifecycle.get("position", np.nan),
                "previous_object_activity": lifecycle.get(
                    "previous_activity", np.nan
                ),
                "next_object_activity": lifecycle.get("next_activity", np.nan),
            }
            for attribute_name in all_event_attributes:
                row[f"event_attr__{attribute_name}"] = event["attributes"].get(
                    attribute_name, np.nan
                )
            e2o_rows.append(row)

    event_df = safe_sort(
        pd.DataFrame(event_rows, columns=event_columns),
        ["event_timestamp", "event_id"],
    )
    e2o_df = safe_sort(
        pd.DataFrame(e2o_rows), ["event_timestamp", "event_id", "object_id"]
    )

    event_ids_by_object = {
        object_id: {event["id"] for event in lifecycle}
        for object_id, lifecycle in object_events.items()
    }

    # ------------------------------------------------------------------
    # 3. Object-to-object-level view
    # ------------------------------------------------------------------
    explicit_relations = [
        (obj, relation)
        for obj in objects
        for relation in obj.get("relationships", [])
        if str(relation.get("objectId")) in object_by_id
    ]

    relation_records: list[dict[str, Any]] = []
    # Choose explicit/inferred mode before filtering, so removing all explicit
    # edges does not unexpectedly replace them with inferred relationships.
    if explicit_relations:
        for source_obj, relation in explicit_relations:
            if (
                source_obj["id"] not in retained_object_ids
                or str(relation["objectId"]) not in retained_object_ids
            ):
                continue
            relation_records.append(
                {
                    "source_id": source_obj["id"],
                    "target_id": str(relation["objectId"]),
                    "qualifier": relation.get("qualifier", np.nan),
                    "relation_origin": "explicit",
                }
            )
    else:
        # OCEL 1.0 has no native O2O section. Create one undirected pair for
        # every two distinct objects that participate in the same event.
        co_occurrence: dict[tuple[str, str], dict[str, list[Any]]] = defaultdict(
            lambda: {"event_ids": [], "activities": [], "timestamps": []}
        )
        for event in events:
            object_ids = sorted(
                {
                    str(relation.get("objectId"))
                    for relation in event["relationships"]
                    if str(relation.get("objectId")) in retained_object_ids
                }
            )
            for source_id, target_id in combinations(object_ids, 2):
                record = co_occurrence[(source_id, target_id)]
                record["event_ids"].append(event["id"])
                record["activities"].append(event["type"])
                record["timestamps"].append(event["time"])

        for (source_id, target_id), record in co_occurrence.items():
            relation_records.append(
                {
                    "source_id": source_id,
                    "target_id": target_id,
                    "qualifier": "co_occurs_in_event",
                    "relation_origin": "inferred_from_e2o",
                    "shared_event_ids": join_values(record["event_ids"]),
                    "shared_activities": join_values(record["activities"]),
                    "first_shared_event_time": min(
                        record["timestamps"], key=str
                    ),
                    "last_shared_event_time": max(record["timestamps"], key=str),
                }
            )

    # Degree is based on the O2O rows produced above. For inferred relations,
    # both endpoints count because co-occurrence is undirected.
    o2o_degree: dict[str, int] = defaultdict(int)
    for relation in relation_records:
        o2o_degree[relation["source_id"]] += 1
        o2o_degree[relation["target_id"]] += 1

    o2o_rows = []
    for relation in relation_records:
        source_id = relation["source_id"]
        target_id = relation["target_id"]
        source_events = event_ids_by_object.get(source_id, set())
        target_events = event_ids_by_object.get(target_id, set())
        shared_events = source_events & target_events
        row = {
            "source_object_id": source_id,
            "source_object_type": object_type_by_id.get(source_id, np.nan),
            "target_object_id": target_id,
            "target_object_type": object_type_by_id.get(target_id, np.nan),
            "o2o_qualifier": relation["qualifier"],
            "relation_origin": relation["relation_origin"],
            "source_event_count": len(source_events),
            "target_event_count": len(target_events),
            "shared_event_count": len(shared_events),
            "source_o2o_degree": o2o_degree[source_id],
            "target_o2o_degree": o2o_degree[target_id],
        }
        for optional_column in (
            "shared_event_ids",
            "shared_activities",
            "first_shared_event_time",
            "last_shared_event_time",
        ):
            if optional_column in relation:
                row[optional_column] = relation[optional_column]
        o2o_rows.append(row)

    o2o_df = safe_sort(
        pd.DataFrame(o2o_rows), ["source_object_id", "target_object_id"]
    )

    object_df = object_df.drop_duplicates().reset_index(drop=True)
    e2o_df = e2o_df.drop_duplicates().reset_index(drop=True)
    o2o_df = o2o_df.drop_duplicates().reset_index(drop=True)

    output_paths = {
        "event": output_dir / f"event_level_{output_suffix}.csv",
        "object": output_dir / f"object_level_{output_suffix}.csv",
        "e2o": output_dir / f"e2o_level_{output_suffix}.csv",
        "o2o": output_dir / f"o2o_level_{output_suffix}.csv",
    }
    views = {"event": event_df, "object": object_df, "e2o": e2o_df, "o2o": o2o_df}
    if manufacturing:
        views["timestamp_sample"] = timestamp_sample_view(
            timestamp_samples, retained_object_ids
        )
        output_paths["timestamp_sample"] = (
            output_dir / f"timestamp_sample_level_{output_suffix}.csv"
        )
    for name, output_path in output_paths.items():
        if output_path.exists():
            print(f"Skipping existing output: {output_path}")
            continue
        views[name].to_csv(
            output_path, index=False, sep=",", encoding="utf-8-sig", mode="x"
        )

    print(f"Detected format: {detected_format}")
    print(f"Input:           {json_path.resolve()}")
    print(f"Event-level:     {event_df.shape[0]:,} rows x {event_df.shape[1]:,} columns")
    print(f"Object-level:    {object_df.shape[0]:,} rows x {object_df.shape[1]:,} columns")
    print(f"E2O-level:       {e2o_df.shape[0]:,} rows x {e2o_df.shape[1]:,} columns")
    print(f"O2O-level:       {o2o_df.shape[0]:,} rows x {o2o_df.shape[1]:,} columns")
    if manufacturing:
        sample_df = views["timestamp_sample"]
        print(f"Timestamp samples: {sample_df.shape[0]:,} rows x {sample_df.shape[1]:,} columns")
    print(f"Output directory: {output_dir.resolve()}")

    return views


if __name__ == "__main__":
    input_prefix = "log" # "ContainerLogistics" # "log"
    output_dir = "Manufacturing/manufact_tabular" # "container_logistics_tabular" # "Manufacturing/manufact_tabular"

    for suffix in range(2, 18):
        json_path = Path(f"Manufacturing/{input_prefix}_{suffix}.jsonocel")

        if not json_path.exists():
            print(f"Skipping missing file: {json_path}")
            continue

        print(f"\nProcessing: {json_path}")
        convert_ocel_to_tabular_views(
            json_path=json_path,
            output_dir=output_dir,
            output_suffix=suffix,
        )
