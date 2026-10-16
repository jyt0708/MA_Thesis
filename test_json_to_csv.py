"""Regression checks for exported views and preservation of existing files."""

import io
import json
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path

import pandas as pd

from json_to_csv import convert_ocel_to_tabular_views, extract_manufacturing_timestamps


class RelationshipFilteringTests(unittest.TestCase):
    def make_log(self, legacy=False, edges=()):
        attributes = [
            {"weight": 10}, {"weight": 20}, {"weight": None},
            {"weight": ""}, {"weight": "   "}, {},
            {"weight": 0}, {"weight": False},
        ]
        object_ids = [f"old_{index}" for index in range(1, 9)]
        # Object 6 is deliberately isolated. The unknown ID never existed.
        participants = [oid for oid in object_ids if oid != "old_6"] + ["unknown"]
        if legacy:
            return {
                "ocel:objects": {
                    oid: {"ocel:type": "Item", "ocel:ovmap": attrs}
                    for oid, attrs in zip(object_ids, attributes)
                },
                "ocel:events": {"event": {
                    "ocel:activity": "Move", "ocel:timestamp": "2026-01-01",
                    "ocel:omap": participants, "ocel:vmap": {},
                }},
            }
        return {
            "objects": [
                {
                    "id": oid, "type": "Item",
                    "attributes": [{"name": key, "value": value}
                                   for key, value in attrs.items()],
                    "relationships": [
                        {"objectId": target, "qualifier": "linked"}
                        for source, target in edges if source == oid
                    ],
                }
                for oid, attrs in zip(object_ids, attributes)
            ],
            "events": [{
                "id": "event", "type": "Move", "time": "2026-01-01",
                "relationships": [{"objectId": oid} for oid in participants],
            }],
        }

    def convert(self, log):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "input.json"
            path.write_text(json.dumps(log), encoding="utf-8")
            with redirect_stdout(io.StringIO()):
                views = convert_ocel_to_tabular_views(path, directory, "test")
            # Verify the actual exports as well as the returned tables.
            for name, frame in views.items():
                csv_path = Path(directory) / f"{name}_level_test.csv"
                self.assertTrue(csv_path.exists())
                if not frame.empty:
                    exported = pd.read_csv(csv_path)
                    for column in frame.columns:
                        if column.endswith("object_id"):
                            self.assertEqual(exported[column].tolist(), frame[column].tolist())
            return views

    def assert_retained_views(self, views):
        retained = {"Item_1", "Item_2", "Item_6", "Item_7", "Item_8"}
        self.assertEqual(set(views["object"]["object_id"]), retained)
        self.assertEqual(set(views["e2o"]["object_id"]), retained - {"Item_6"})
        self.assertTrue(views["e2o"]["event_relation_count"].eq(4).all())
        self.assertTrue(views["e2o"]["object_event_count"].eq(1).all())
        self.assertTrue(views["e2o"]["object_event_position"].eq(1).all())
        isolated = views["object"].set_index("object_id").loc["Item_6"]
        self.assertEqual(isolated["number_of_events"], 0)

    def test_inferred_relations_in_both_formats(self):
        for legacy in (True, False):
            with self.subTest(legacy=legacy):
                views = self.convert(self.make_log(legacy=legacy))
                self.assert_retained_views(views)
                o2o = views["o2o"]
                expected = {"Item_1", "Item_2", "Item_7", "Item_8"}
                self.assertEqual(set(o2o["source_object_id"]) | set(o2o["target_object_id"]), expected)
                self.assertEqual(len(o2o), 6)
                self.assertTrue(o2o["source_o2o_degree"].eq(3).all())
                self.assertTrue(o2o["target_o2o_degree"].eq(3).all())

    def test_explicit_relations_filter_both_endpoints(self):
        views = self.convert(self.make_log(edges=[
            ("old_1", "old_2"), ("old_1", "old_3"),
            ("old_3", "old_2"), ("old_2", "unknown"),
        ]))
        self.assert_retained_views(views)
        o2o = views["o2o"]
        self.assertEqual(list(zip(o2o["source_object_id"], o2o["target_object_id"])),
                         [("Item_1", "Item_2")])
        self.assertEqual(o2o["source_o2o_degree"].tolist(), [1])
        self.assertEqual(o2o["target_o2o_degree"].tolist(), [1])

    def test_removed_explicit_relations_do_not_trigger_inference(self):
        views = self.convert(self.make_log(edges=[("old_1", "old_3")]))
        self.assert_retained_views(views)
        self.assertTrue(views["o2o"].empty)

    def test_all_objects_removed(self):
        for legacy in (True, False):
            with self.subTest(legacy=legacy):
                log = self.make_log(legacy=legacy)
                if legacy:
                    for obj in log["ocel:objects"].values():
                        obj["ocel:ovmap"] = {"weight": None}
                else:
                    for obj in log["objects"]:
                        obj["attributes"] = [{"name": "weight", "value": None}]
                views = self.convert(log)
                self.assertTrue(all(views[name].empty for name in ("object", "e2o", "o2o")))
                self.assertEqual(views["event"]["event_relation_count"].tolist(), [0])

    def test_event_view_in_both_formats(self):
        for legacy in (True, False):
            with self.subTest(legacy=legacy):
                log = self.make_log(legacy=legacy)
                if legacy:
                    log["ocel:events"]["event"]["ocel:vmap"] = {"cost": 7}
                    log["ocel:events"]["isolated"] = {
                        "ocel:activity": "Start", "ocel:timestamp": "2025-12-31",
                        "ocel:omap": [], "ocel:vmap": {},
                    }
                else:
                    log["events"][0]["attributes"] = [{"name": "cost", "value": 7}]
                    log["events"].append({
                        "id": "isolated", "type": "Start", "time": "2025-12-31",
                        "relationships": [],
                    })
                views = self.convert(log)
                event = views["event"]
                self.assertEqual(event["event_id"].tolist(), ["start_1", "move_1"])
                self.assertEqual(event["event_activity"].tolist(), ["Start", "Move"])
                self.assertEqual(event["event_timestamp"].tolist(), ["2025-12-31", "2026-01-01"])
                self.assertEqual(event["event_relation_count"].tolist(), [0, 4])
                self.assertTrue(pd.isna(event.iloc[0]["event_attr__cost"]))
                self.assertEqual(event.iloc[1]["event_attr__cost"], 7)
                self.assertEqual(set(views["e2o"]["event_id"]), {"move_1"})

    def test_empty_event_export_has_headers(self):
        views = self.convert({"events": [], "objects": []})
        self.assertTrue(views["event"].empty)
        self.assertEqual(list(views["event"].columns), [
            "event_id", "event_activity", "event_timestamp", "event_relation_count",
        ])

    def test_existing_outputs_are_preserved_and_missing_outputs_created(self):
        names = ("event", "object", "e2o", "o2o")
        for missing in names:
            with self.subTest(missing=missing), tempfile.TemporaryDirectory() as directory:
                path = Path(directory) / "input.json"
                path.write_text(json.dumps(self.make_log()), encoding="utf-8")
                outputs = {name: Path(directory) / f"{name}_level_test.csv" for name in names}
                original = {}
                for name, output in outputs.items():
                    if name != missing:
                        output.write_bytes(f"existing {name}\n".encode())
                        original[name] = (output.read_bytes(), output.stat().st_mtime_ns)
                with redirect_stdout(io.StringIO()):
                    views = convert_ocel_to_tabular_views(path, directory, "test")
                self.assertEqual(set(views), set(names))
                self.assertTrue(outputs[missing].exists())
                self.assertEqual(len(pd.read_csv(outputs[missing])), len(views[missing]))
                for name, (content, modified) in original.items():
                    self.assertEqual(outputs[name].read_bytes(), content)
                    self.assertEqual(outputs[name].stat().st_mtime_ns, modified)
                # Once all outputs exist, a subsequent run changes none of them.
                before = {name: (output.read_bytes(), output.stat().st_mtime_ns)
                          for name, output in outputs.items()}
                with redirect_stdout(io.StringIO()):
                    convert_ocel_to_tabular_views(path, directory, "test")
                for name, output in outputs.items():
                    self.assertEqual((output.read_bytes(), output.stat().st_mtime_ns), before[name])


class ManufacturingTimestampTests(unittest.TestCase):
    stamp = "2026-01-01T08:02:05.700000+01:00"
    local_stamp = "2026-01-01T08:02:05.700000"

    def make_log(self, legacy):
        point = {
            "stream:id": "Spindle/actSpeed",
            "stream:value": 1800,
            "stream:timestamp": self.local_stamp,
            "stream:source": {"proto": "opcua", "host": "MaxxTurn45"},
            "stream:meta": {
                "SourceTimestamp": self.local_stamp,
                "ServerTimestamp": self.stamp,
            },
        }
        objects = [
            ("data", "data", {
                "at": "2026-01-01 08:02:05 +0100",
                "xxx": {
                    "timestamp": self.stamp,
                    "value": {"LastTransitionAt": "2026-01-01T08:02:00.123456"},
                    "note": "not a date",
                    "invalid": "2026-02-30T08:00:00+01:00",
                    "missing": None,
                },
                "yyy": {"timestamp": self.local_stamp},
            }),
            ("stream", "stream", {"stream:point": [point, point]}),
            ("removed", "stream", {"stream:point": [point], "label": None}),
        ]
        if legacy:
            return {
                "ocel:objects": {
                    oid: {"ocel:type": typ, "ocel:ovmap": {"list": [
                        entry
                        for key, value in attrs.items()
                        for entry in (
                            [{"stream:point": sample} for sample in value]
                            if key == "stream:point"
                            else [{"name": key, "value": value}]
                        )
                    ]}}
                    for oid, typ, attrs in objects
                },
                "ocel:events": {"read": {
                    "ocel:activity": "Read", "ocel:timestamp": self.stamp,
                    "ocel:omap": [oid for oid, _, _ in objects], "ocel:vmap": {},
                }},
            }
        return {
            "objects": [
                {"id": oid, "type": typ, "attributes": [
                    {"name": key, "value": value} for key, value in attrs.items()
                ]}
                for oid, typ, attrs in objects
            ],
            "events": [{
                "id": "read", "type": "Read", "time": self.stamp,
                "relationships": [{"objectId": oid} for oid, _, _ in objects],
            }],
        }

    def convert(self, log, folder="Manufacturing", **kwargs):
        with tempfile.TemporaryDirectory() as directory:
            input_dir = Path(directory) / folder
            input_dir.mkdir()
            path = input_dir / "input.json"
            path.write_text(json.dumps(log), encoding="utf-8")
            with redirect_stdout(io.StringIO()):
                views = convert_ocel_to_tabular_views(path, directory, "nested", **kwargs)
            for name, view in views.items():
                if not len(view.columns):
                    self.assertTrue(view.empty)
                    continue
                exported = pd.read_csv(Path(directory) / f"{name}_level_nested.csv")
                self.assertEqual(list(exported.columns), list(view.columns))
                self.assertEqual(len(exported), len(view))
                for column in view.columns:
                    if column.startswith("obj_attr__") and "timestamp" in column.casefold():
                        self.assertEqual(exported[column].fillna("").tolist(),
                                         view[column].fillna("").tolist())
            return views

    def test_nested_targets_and_sample_alignment_in_both_formats(self):
        for legacy in (True, False):
            with self.subTest(legacy=legacy):
                views = self.convert(self.make_log(legacy))
                objects = views["object"].set_index("object_id")
                self.assertEqual(set(objects.index), {"data_1", "stream_1"})
                self.assertEqual(objects.loc["data_1", "obj_attr__xxx__timestamp"], self.stamp)
                self.assertEqual(objects.loc["data_1", "obj_attr__yyy__timestamp"], self.local_stamp)
                self.assertEqual(objects.loc["data_1", "obj_attr__xxx__value__LastTransitionAt"],
                                 "2026-01-01T08:02:00.123456")
                self.assertEqual(objects.loc["stream_1", "obj_attr__xxx__timestamp__exists"], 0)
                self.assertEqual(objects.loc["data_1", "obj_attr__xxx__timestamp__exists"], 1)
                self.assertNotIn("obj_attr__xxx__invalid", objects.columns)
                self.assertFalse(any("[0]" in name for name in objects.columns))
                original = json.loads(objects.loc["data_1", "obj_attr__xxx"])
                self.assertEqual(original["timestamp"], self.stamp)
                self.assertEqual(len(json.loads(objects.loc["stream_1", "obj_attr__stream:point"])), 2)

                samples = views["timestamp_sample"]
                self.assertEqual(samples["object_id"].tolist(), ["stream_1", "stream_1"])
                self.assertEqual(samples["sample_index"].tolist(), [0, 1])
                self.assertEqual(samples["array_item_path"].tolist(),
                                 ["stream:point__[0]", "stream:point__[1]"])
                self.assertEqual(samples["obj_attr__stream:point__stream:timestamp"].tolist(),
                                 [self.local_stamp] * 2)
                self.assertEqual(samples["obj_attr__stream:point__stream:meta__ServerTimestamp"].tolist(),
                                 [self.stamp] * 2)
                self.assertEqual(samples["sample_attr__stream:point__stream:id"].tolist(),
                                 ["Spindle/actSpeed"] * 2)
                self.assertEqual(samples["sample_attr__stream:point__stream:value"].tolist(), [1800] * 2)
                self.assertFalse(any(name.startswith("sample_attr__") and "Timestamp" in name
                                     for name in samples.columns))
                self.assertEqual(views["event"]["event_relation_count"].tolist(), [2])

    def test_manufacturing_scope_and_explicit_override(self):
        log = self.make_log(True)
        for folder, kwargs, enabled in (
            ("ContainerLogistics", {}, False),
            ("Manufacturing", {"manufacturing": False}, False),
            ("mAnUfAcTuRiNg", {}, True),
            ("ExternalInput", {"manufacturing": True}, True),
        ):
            with self.subTest(folder=folder, kwargs=kwargs):
                views = self.convert(log, folder, **kwargs)
                self.assertEqual("timestamp_sample" in views, enabled)
                self.assertEqual("obj_attr__xxx__timestamp" in views["object"].columns, enabled)

    def test_timestamp_detection_nested_arrays_and_path_escaping(self):
        objects = [{"id": "data_1", "type": "data", "attributes": {"record": json.dumps({
            "any_name": "2022-12-09 14:09:10 +0100",
            "utc": "2026-01-01T07:00:00Z",
            "bad": "2026-02-30T08:00:00",
            "date_only": "2026-01-01",
            "text": "seen at 2026-01-01T08:00:00",
            "missing": "NaT", "epoch": 1767250800,
            "a__b": self.stamp, "a": {"b": self.local_stamp},
            "samples": [[{"clock": self.stamp}, {"clock": self.local_stamp}]],
        })}}]
        rows = extract_manufacturing_timestamps(objects)
        attrs = objects[0]["attributes"]
        self.assertEqual(set(attrs) - {"record"}, {
            "record__any_name", "record__utc", "record__a%5F%5Fb", "record__a__b",
        })
        self.assertEqual([row["array_item_path"] for row in rows],
                         ["record__samples__[0]__[0]", "record__samples__[0]__[1]"])
        self.assertEqual([row["obj_attr__record__samples__clock"] for row in rows],
                         [self.stamp, self.local_stamp])

    def test_sample_targets_with_different_fields_and_no_retained_samples(self):
        log = self.make_log(False)
        points = log["objects"][1]["attributes"][0]["value"]
        log["objects"][1]["attributes"][0]["value"] = [points[0], {"other": self.stamp}]
        views = self.convert(log)
        samples = views["timestamp_sample"]
        self.assertEqual(samples["obj_attr__stream:point__other__exists"].tolist(), [0, 1])
        self.assertEqual(samples["obj_attr__stream:point__stream:timestamp__exists"].tolist(), [1, 0])
        log["objects"][1]["attributes"].append({"name": "invalid", "value": None})
        self.assertTrue(self.convert(log)["timestamp_sample"].empty)
        self.assertTrue(self.convert({"objects": [], "events": []})["timestamp_sample"].empty)

    def test_existing_outputs_preserved_when_sample_export_is_added(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "input.json"
            path.write_text(json.dumps(self.make_log(True)), encoding="utf-8")
            original = {}
            for name in ("event", "object", "e2o", "o2o"):
                output = Path(directory) / f"{name}_level_nested.csv"
                output.write_bytes(f"existing {name}\n".encode())
                original[output] = (output.read_bytes(), output.stat().st_mtime_ns)
            with redirect_stdout(io.StringIO()):
                views = convert_ocel_to_tabular_views(path, directory, "nested", manufacturing=True)
            sample_path = Path(directory) / "timestamp_sample_level_nested.csv"
            self.assertEqual(len(pd.read_csv(sample_path)), len(views["timestamp_sample"]))
            original[sample_path] = (sample_path.read_bytes(), sample_path.stat().st_mtime_ns)
            with redirect_stdout(io.StringIO()):
                convert_ocel_to_tabular_views(path, directory, "nested", manufacturing=True)
            for output, expected in original.items():
                self.assertEqual((output.read_bytes(), output.stat().st_mtime_ns), expected)


if __name__ == "__main__":
    unittest.main()
