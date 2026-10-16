from __future__ import annotations

import json
import math
import random
import uuid
from collections import Counter, defaultdict
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


TRANSITION_TO_LIFECYCLE = {
    "activity/calling": "start",
    "activity/done": "complete",
    "activity/receiving": "unknown",
    "dataelements/change": "unknown",
    "state/change": "unknown",
    "endpoints/change": "unknown",
    "attributes/change": "unknown",
    "description/change": "unknown",
    "task/instantiation": "unknown",
    "subprocess/receiving": "unknown",
}

EXPECTED_TRANSITIONS = {
    "activity/calling": 1551,
    "activity/done": 1798,
    "activity/receiving": 2558,
    "dataelements/change": 1488,
    "state/change": 654,
    "endpoints/change": 365,
    "attributes/change": 217,
    "description/change": 217,
    "task/instantiation": 216,
    "subprocess/receiving": 204,
}

PARTS = [
    ("order1", "wr04", 0.03, 0.6),
    ("order1", "wr04", 0.03, 0.6),
    ("order1", "wr08", 0.05, 0.2),
    ("order1", "wr08", 0.05, 0.6),
    ("order1", "wr08", 0.05, 0.6),
    ("order2", "wr04", 0.03, 0.4),
    ("order2", "wr04", 0.07, 0.4),
    ("order2", "wr08", 0.05, 0.6),
]

SUBPROCESS_TYPES = [
    "abstraction:sub:- Turm Batch Processing V2",
    "abstraction:sub:Turm Single wr04",
    "abstraction:sub:Turm Single wr08",
    "abstraction:sub:X Generate NC",
    "abstraction:sub:Turm Turn Production",
    "abstraction:sub:X MT45 SP",
    "abstraction:sub:X MT45 Start",
    "abstraction:sub:Turm MT45 Take Out",
    "abstraction:sub:MT45 Check Operational Safety",
    "abstraction:sub:Turm IRB2600 Grip",
    "abstraction:sub:Turm IRB2600 Extract",
    "abstraction:sub:Turm IRB2600 Measure and Tray",
    "abstraction:sub:Turm IRB2600 Tray",
    "abstraction:sub:Turm Keyence Measurement",
    "abstraction:sub:Machining V2",
    "abstraction:sub:X MT45 Control Getter",
    "abstraction:sub:X MT45 Control Setter",
    "abstraction:sub:X IRB2600 Execution",
    "abstraction:sub:IRB2600 Portal to Door",
    "abstraction:sub:IRB2600 Door to GS",
    "abstraction:sub:IRB2600 GS to Take",
    "abstraction:sub:IRB2600 Take to GS",
    "abstraction:sub:IRB2600 IRB2600 GS to Door",
    "abstraction:sub:IRB2600 Door to Portal",
]

ACTIVITY_COUNTS = {
    "external": 1,
    "Get Order Details": 1,
    "Inititalize Batch": 1,
    "Next QR, Order Item and Tray": 8,
    "Get Machine State Cancelled": 8,
    "Spawn Production": 8,
    "Wait for MT45 Takeout": 8,
    "Not First Part": 1,
    "Create Program": 8,
    "Turn": 8,
    "Set Program MT45": 8,
    "MT45 Start": 8,
    "MT45 Take Out": 8,
    "Robot Measuring and Traying": 4,
    "Measure with Keyence": 4,
    "Signal Keyence End": 4,
    "Robot Traying": 4,
    "Aux On": 8,
    "Mode Auto": 8,
    "Door Close": 8,
    "NC Start": 8,
    "MT45 Check Operational Safety": 8,
    "Get Magazine X": 8,
    "Get Magazine Z": 8,
    "Open Clamp 2": 8,
    "Reset Clamp 2": 8,
    "Taken Out": 8,
    "Set Dummy Program": 8,
    "Select Dummy Program": 16,
    "Set Program": 72,
    "Set Var": 48,
    "Read Var": 186,
    "IRB2600 Portal to Door": 8,
    "IRB2600 Door to GS": 8,
    "IRB2600 GS to Take": 8,
    "IRB2600 Grip": 8,
    "IRB2600 Take to GS": 8,
    "IRB2600 GS to Door": 8,
    "IRB2600 Extract": 8,
    "IRB2600 Door to Portal": 8,
    "IRB2600 Unload to Tray Down": 8,
    "IRB2600 Door to Measure": 4,
    "IRB2600 Measure to Door": 4,
    "Move Down": 4,
    "Move Up": 4,
    "wait": 4,
    "Signal Unloading End": 8,
    "Fetch Program": 136,
    "Fetch": 12,
    "Logon": 64,
    "Assign Program": 64,
    "Check State": 590,
    "Start": 64,
    "Set Fetch Endpoint": 128,
    "Assemble Porgram": 64,
    "Replace Patterns with Params": 8,
    "Set Endpoint": 8,
    "Set Paths": 8,
    "Correct  Queue": 12,
    "Detection Machining": 12,
    "Machining": 12,
    "Get Current Time": 4,
    "Wait For Unload End": 7,
}

FOUR_EVENT_READS = {
    "Check State",
    "Read Var",
    "Fetch Program",
    "Detection Machining",
    "Get Magazine X",
    "Get Magazine Z",
}

DATA_UPDATE_ACTIVITIES = {
    "Get Order Details",
    "Inititalize Batch",
    "Next QR, Order Item and Tray",
    "Assemble Porgram",
    "Replace Patterns with Params",
    "Get Current Time",
    "Set Paths",
    "Not First Part",
}

ENDPOINT_UPDATE_ACTIVITIES = {"Set Endpoint", "Correct  Queue"}

PROGRAM_URL = (
    "https://raw.githubusercontent.com/Process-Oriented-Data-Science/"
    "MT45/raw/master/TURM_PSC.MPF"
)

NC_PROGRAM = """; TURM_PSC.MPF
; Parametric rook machining program
DEF REAL VORSCHUB, SCHNITTTIEFE
G17 G40 G90
T="SCHRUPPER-A-L" M6
S4000 M3
G0 X40 Z2
G1 X0 F=VORSCHUB
G1 Z-60 AP=SCHNITTTIEFE
M5
M30
"""

RAPID_PROGRAM = """MODULE OperationModule
  CONST speeddata jFast:=[500,250,200,200];
  PROC Operation()
    MoveL PortalHighPoint,jFast,z50,tool_Parallelgreifer_Standard\\WObj:=wobj0;
    MoveL PortalWaypoint,jFast,z50,tool_Parallelgreifer_Standard\\WObj:=wobj0;
    MoveL MT45Front,jFast,z50,tool_Parallelgreifer_Standard\\WObj:=wobj0;
  ENDPROC
ENDMODULE
"""

ROBOT_CHANNELS = [
    "StateMachine",
    "LastTransition",
    "LastTransitionAt",
    "Rapid.ExecutionStatus",
    "State",
    "OperationMode",
    "BackupInProgress",
    "Connected",
    "Task.Program.Module",
    "Task.Program.Routine",
    "Task.Program.Range",
    "Task.Cycle",
    "Task.Enabled",
    "Task.ExecutionType",
    "Task.Motion",
    "Task.RemainingCycles",
    "Task.TaskType",
    "Task.Type",
    "Task.ExecutionStatus",
    "Task.Name",
    "Task.Motion.Module",
    "Task.Motion.Routine",
    "Rapid.Cycle",
    "Rapid.MaximumIdentifierLength",
    "Rapid.RemainingCycles",
    "Rapid.IsMaster",
    "Rapid.Name",
    "RobJoint.Rax_1",
    "RobJoint.Rax_2",
    "RobJoint.Rax_3",
    "RobJoint.Rax_4",
    "RobJoint.Rax_5",
    "RobJoint.Rax_6",
    "Location.Column",
    "Location.Row",
    "Pos.X",
    "Pos.Y",
    "Pos.Z",
]


class _IdFactory:
    def __init__(self, seed: int) -> None:
        self.namespace = uuid.uuid5(uuid.NAMESPACE_URL, f"nine-rooks:{seed}")
        self.counts: Counter[str] = Counter()

    def new(self, kind: str, compact: bool = False) -> str:
        index = self.counts[kind]
        self.counts[kind] += 1
        value = uuid.uuid5(self.namespace, f"{kind}:{index}")
        return value.hex if compact else str(value)


def _iso(value: datetime) -> str:
    return value.isoformat(timespec="milliseconds")


def _parse_time(value: str) -> datetime:
    return datetime.fromisoformat(value)


def _tray_position(index: int) -> List[float]:
    row, column = divmod(index, 6)
    return [
        round(-137.87 + 50.0 * row + 0.6 * column, 2),
        -178.0 + 80.0 * column,
        199.0,
    ]


def _all_tray_positions() -> List[List[List[float]]]:
    return [
        [_tray_position(row * 6 + column) for column in range(6)]
        for row in range(8)
    ]


def _event_base(event: Dict[str, Any]) -> str:
    return event["ocel:activity"].rsplit("§", 1)[0]


def _stream_headers(name: str) -> List[Dict[str, Any]]:
    return [
        {"stream:name": name},
        {"stream:source": "IRB2600" if name == "IRB2600" else "machine"},
    ]


def _source_timestamp(when: datetime, local: bool) -> str:
    if local:
        return when.replace(tzinfo=None).isoformat(timespec="milliseconds")
    return _iso(when)


def _point(
    channel: str,
    value: Any,
    when: datetime,
    source: Any,
    meta: Optional[dict] = None,
    local_time: bool = False,
) -> Dict[str, Any]:
    result: Dict[str, Any] = {
        "stream:id": channel,
        "stream:value": value,
        "stream:timestamp": _source_timestamp(when, local_time),
        "stream:source": source,
    }
    if meta is not None:
        result["stream:meta"] = meta
    return {"stream:point": result}


def _machine_points(
    count: int,
    bundle_index: int,
    when: datetime,
    part_index: int,
    rng: random.Random,
) -> List[Dict[str, Any]]:
    if count == 0:
        return []

    progress = (bundle_index % 92) / 91.0
    running = 1 if 0.06 < progress < 0.94 else 0
    command_speed = 4000.0 if running else 0.0
    actual_speed = command_speed + (
        rng.uniform(-18.0, 18.0) if running else rng.uniform(-1.0, 1.0)
    )
    feed = PARTS[part_index][2]
    depth = PARTS[part_index][3]

    phase_values: Dict[str, Tuple[float, float, float, float]] = {}
    for family, center in (
        ("Power/Active", 2100.0),
        ("Axes/Power/Active", 1150.0),
        ("Coolant/Power/Active", 230.0),
    ):
        scale = 1.0 + 0.35 * math.sin(
            progress * math.pi * 2.0 + part_index
        )
        a = center * scale + rng.uniform(-20.0, 20.0)
        b = center * scale * 0.98 + rng.uniform(-20.0, 20.0)
        c = center * scale * 1.02 + rng.uniform(-20.0, 20.0)
        phase_values[family] = (
            a,
            b,
            c,
            a + b + c + rng.uniform(-2.0, 2.0),
        )

    axis_position = {
        axis: 100.0
        * math.sin(progress * math.pi * (1.0 + offset / 5.0) + offset)
        for offset, axis in enumerate(("X", "Y", "Z"))
    }
    axis_velocity = {
        axis: (
            25.0
            * math.cos(
                progress * math.pi * (1.0 + offset / 5.0) + offset
            )
            if running
            else 0.0
        )
        for offset, axis in enumerate(("X", "Y", "Z"))
    }

    tool_ids = [
        "SCHRUPPER-A-L",
        "SCHLICHTER-A-L",
        "FRAESER-10-AXHS",
        "AX_UK_IN_SCHLI",
        "EINSTI_2N-LR-HS",
        "EINSTICH-ABSTICH3",
        "SCHLICHTER-A-R",
        "SCHRUPPER-A-R",
    ]
    tool_numbers = [7, 37, 54, 74, 8, 107, 38, 53]
    tool_pos = min(7, int(progress * 8))
    program_message = (
        f"Type: {PARTS[part_index][0]}: {PARTS[part_index][1]}; "
        f"Schnitttiefe: {depth}; Vorschub: {feed}"
    )

    values: Dict[str, Any] = {
        "Spindle/actSpeed": round(actual_speed, 3),
        "Spindle/cmdSpeed": command_speed,
        "Spindle/driveLoad": (
            round(42.0 + 20.0 * math.sin(progress * math.pi), 3)
            if running
            else 0.0
        ),
        "Spindle/aaPower": round(max(0.0, actual_speed * 1.4), 3),
        "Spindle/turnState": running,
        "Program/actBlock": (
            "G1 Z-60 AP=SCHNITTTIEFE" if running else "M30"
        ),
        "Program/actLineNumber": int(10 + 90 * progress),
        "Program/msg": program_message,
        "Program/progName1": (
            "_N_TURM_PSC1_MPF" if running else "_N_ASUP2_SYF"
        ),
        "State/actToolIdent": tool_ids[tool_pos],
        "State/actTNumber": tool_numbers[tool_pos],
        "State/actToolLength1": round(80.0 + tool_pos * 2.5, 3),
        "State/actToolRadius": round(0.4 + tool_pos * 0.1, 3),
        "State/progStatus": 3 if running else 4,
        "State/resetActive": not bool(running),
        "State/progStarts": part_index + 1,
    }

    for axis in ("X", "Y", "Z"):
        values[f"Axes/{axis}/aaLeadP"] = round(
            axis_position[axis], 4
        )
        values[f"Axes/{axis}/aaVactB"] = round(
            axis_velocity[axis], 4
        )
        values[f"Axes/{axis}/actFeedRate"] = round(
            axis_velocity[axis] * 60.0, 4
        )
        values[f"Axes/{axis}/aaLoad"] = round(
            abs(axis_velocity[axis]) * 1.7, 4
        )
        values[f"Axes/{axis}/aaTorque"] = round(
            axis_velocity[axis] * 0.8, 4
        )
        values[f"Axes/{axis}/aaPower"] = round(
            abs(axis_velocity[axis]) * 32.0, 4
        )

    records: List[Tuple[str, Any]] = []
    for repeat in range(16):
        prefix = "MaxxTurn45/" if repeat % 2 else ""
        for family in (
            "Power/Active",
            "Axes/Power/Active",
            "Coolant/Power/Active",
        ):
            a, b, c, total = phase_values[family]
            for suffix, value in zip(
                ("A", "B", "C", "Sum"),
                (a, b, c, total),
            ):
                records.append(
                    (prefix + family + "/" + suffix, round(value, 4))
                )
        for channel, value in values.items():
            records.append((prefix + channel, value))

    records = records[:count]
    source_opc = {
        "proto": "opcua",
        "ip": "192.168.0.45",
        "host": "MaxxTurn45",
    }
    source_modbus = {
        "proto": "modbus",
        "ip": "192.168.0.46",
        "host": "power-meter",
    }

    result = []
    for index, (channel, value) in enumerate(records):
        source = (
            source_modbus
            if "Power/Active" in channel and "Spindle" not in channel
            else source_opc
        )
        meta = {
            "ClientHandle": index + 1,
            "ServerTimestamp": _iso(when),
            "SourceTimestamp": _iso(
                when + timedelta(milliseconds=index % 17)
            ),
            "StatusCode": "Good",
            "VariantType": (
                "VariantType.String"
                if isinstance(value, str)
                else "VariantType.UInt16"
                if isinstance(value, (bool, int))
                else "VariantType.Double"
            ),
        }
        result.append(
            _point(
                channel,
                value,
                when + timedelta(milliseconds=index % 17),
                source,
                meta,
            )
        )
    return result


def _robot_points(
    bundle_index: int,
    when: datetime,
) -> List[Dict[str, Any]]:
    phase = bundle_index / 24.0
    active = bundle_index % 10 < 7
    state = "Active" if active else "Finished"
    transition = (
        "FinishedToActive" if active else "ActiveToFinished"
    )
    joint_ranges = [
        (-102.02, 153.09),
        (-32.24, 73.68),
        (-48.20, 56.73),
        (-91.83, 2.31),
        (7.31, 89.45),
        (20.04, 126.64),
    ]

    values: Dict[str, Any] = {
        "StateMachine": state,
        "LastTransition": transition,
        "LastTransitionAt": _iso(when),
        "Rapid.ExecutionStatus": (
            "Running" if active else "Stopped"
        ),
        "State": "MotorsOn",
        "OperationMode": "Auto",
        "BackupInProgress": False,
        "Connected": True,
        "Task.Program.Module": "OperationModule",
        "Task.Program.Routine": "Operation",
        "Task.Program.Range": "All",
        "Task.Cycle": "Once",
        "Task.Enabled": True,
        "Task.ExecutionType": (
            "None" if bundle_index % 97 == 0 else "Normal"
        ),
        "Task.Motion": True,
        "Task.RemainingCycles": 1,
        "Task.TaskType": "Normal",
        "Task.Type": "Task",
        "Task.ExecutionStatus": (
            "Running"
            if active
            else "Stopped"
            if bundle_index % 2
            else "Ready"
        ),
        "Task.Name": "T_ROB1",
        "Task.Motion.Module": "OperationModule",
        "Task.Motion.Routine": "Operation",
        "Rapid.Cycle": "Once",
        "Rapid.MaximumIdentifierLength": 32,
        "Rapid.RemainingCycles": 1,
        "Rapid.IsMaster": True,
        "Rapid.Name": "T_ROB1",
        "Location.Column": bundle_index % 6,
        "Location.Row": (bundle_index // 6) % 8,
        "Pos.X": round(
            350.0 + 800.0 * math.sin(phase * 0.31), 4
        ),
        "Pos.Y": round(
            110.0 + 600.0 * math.sin(phase * 0.27 + 0.7), 4
        ),
        "Pos.Z": round(
            570.0 + 400.0 * math.sin(phase * 0.23 + 1.2), 4
        ),
    }

    for index, (low, high) in enumerate(joint_ranges, 1):
        midpoint = (low + high) / 2.0
        amplitude = (high - low) * 0.43
        values[f"RobJoint.Rax_{index}"] = round(
            midpoint
            + amplitude * math.sin(phase * 0.19 + index),
            4,
        )

    return [
        _point(
            channel,
            values[channel],
            when,
            "IRB2600",
            {
                "type": type(values[channel]).__name__,
                "enumeration": channel,
            },
        )
        for channel in ROBOT_CHANNELS
    ]


def _keyence_points(
    count: int,
    bundle_index: int,
    when: datetime,
    part_index: int,
    rng: random.Random,
) -> List[Dict[str, Any]]:
    result = []
    noisy = part_index in {1, 6}

    for index in range(count):
        if (
            index < 12
            or index >= count - 12
            or (noisy and index % 53 == 0)
        ):
            value = 999.99
        else:
            center = (
                29.80
                if PARTS[part_index][1] == "wr04"
                else 34.20
            )
            noise = rng.gauss(
                0.0,
                0.16 if noisy else 0.035,
            )
            value = round(
                max(0.36, min(999.98, center + noise)),
                2,
            )

        source = {
            "proto": "serial",
            "access": "/dev/ttyUSB0",
        }
        result.append(
            _point(
                "measurement",
                value,
                when + timedelta(milliseconds=index * 2),
                source,
                None,
                local_time=True,
            )
        )

    return result


def generate_ocel(seed: int, number_of_parts: int) -> dict:
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise TypeError("seed must be an integer")
    if (
        isinstance(number_of_parts, bool)
        or not isinstance(number_of_parts, int)
    ):
        raise TypeError("number_of_parts must be an integer")
    if number_of_parts != 8:
        raise ValueError(
            "The exact-source-fidelity batch is fixed at 8 parts: "
            "four wr04 and four wr08 items across order1 and order2. "
            "Use number_of_parts=8."
        )

    rng = random.Random(seed)
    ids = _IdFactory(seed)
    objects: Dict[str, Dict[str, Any]] = {}
    event_rows: List[
        Tuple[datetime, int, str, Dict[str, Any]]
    ] = []
    subprocesses: Dict[str, Dict[str, Any]] = {}
    activity_specs: List[Dict[str, Any]] = []
    interface_specs: List[Dict[str, Any]] = []
    sequence = 0
    concept_counter = 4819

    base_time = datetime(
        2023,
        3,
        10,
        15,
        49,
        26,
        506000,
        tzinfo=timezone(timedelta(hours=1)),
    )
    base_time += timedelta(
        milliseconds=rng.randint(0, 999)
    )

    component_ids = {}
    for name in ("IRB2600", "MaxxTurn45", "keyence", "NA"):
        object_id = ids.new("component")
        component_ids[name] = object_id
        objects[object_id] = {
            "ocel:type": f"view:component:{name}",
            "ocel:ovmap": {},
        }

    def add_subprocess(
        type_name: str,
        start: float,
        end: float,
        part_index: Optional[int],
        parent: Optional[str] = None,
        caller: Optional[str] = None,
        asynchronous: bool = False,
        equipment: Optional[str] = None,
    ) -> str:
        nonlocal concept_counter

        if end <= start:
            raise ValueError(
                "Subprocess duration must be positive"
            )

        sub_id = ids.new("subprocess")
        objects[sub_id] = {
            "ocel:type": type_name,
            "ocel:ovmap": {},
        }
        subprocesses[sub_id] = {
            "type": type_name,
            "start": start,
            "end": end,
            "part": part_index,
            "parent": parent,
            "concept": concept_counter,
            "equipment": equipment,
        }
        concept_counter += 1

        if parent is not None:
            if caller is None:
                raise ValueError(
                    "A non-root subprocess requires a caller activity"
                )
            spec = {
                "base": caller,
                "owner": parent,
                "child": sub_id,
                "interface": True,
                "async": asynchronous,
                "part": part_index,
            }
            activity_specs.append(spec)
            interface_specs.append(spec)

        return sub_id

    batch = add_subprocess(
        SUBPROCESS_TYPES[0],
        0.0,
        2201.0 + rng.uniform(-8.0, 8.0),
        None,
    )

    top_by_part: Dict[int, str] = {}
    route_by_part: Dict[int, str] = {}
    machine_subs: List[str] = []
    robot_execution_subs: List[str] = []
    getter_subs: List[str] = []
    setter_subs: List[str] = []
    movement_subs: List[str] = []

    wr04_ranges = [
        (334.873, 348.0),
        (350.0, 370.504),
        (334.0, 344.687),
        (360.0, 377.150),
    ]
    wr08_ranges = [
        (306.0, 322.844),
        (283.439, 300.0),
        (283.333, 299.0),
        (301.0, 317.664),
    ]

    n04_seen = 0
    n08_seen = 0

    for part_index, (_, item, _, _) in enumerate(PARTS):
        top_start = part_index * 235.0

        if item == "wr04":
            low, high = wr04_ranges[n04_seen]
            n04_seen += 1
        else:
            low, high = wr08_ranges[n08_seen]
            n08_seen += 1

        duration = rng.uniform(
            min(low, high),
            max(low, high),
        )
        top_end = top_start + duration

        top = add_subprocess(
            f"abstraction:sub:Turm Single {item}",
            top_start,
            top_end,
            part_index,
            batch,
            "Spawn Production",
            asynchronous=True,
        )
        top_by_part[part_index] = top

        add_subprocess(
            "abstraction:sub:X Generate NC",
            top_start + 2.2,
            top_start + rng.uniform(6.0, 8.0),
            part_index,
            top,
            "Create Program",
        )
        turn = add_subprocess(
            "abstraction:sub:Turm Turn Production",
            top_start + 2.4,
            top_start + rng.uniform(139.0, 143.0),
            part_index,
            top,
            "Turn",
        )
        mt45_sp = add_subprocess(
            "abstraction:sub:X MT45 SP",
            top_start + 6.5,
            top_start + rng.uniform(9.5, 11.0),
            part_index,
            top,
            "Set Program MT45",
        )
        mt45_start = add_subprocess(
            "abstraction:sub:X MT45 Start",
            top_start + 11.5,
            top_start + rng.uniform(18.0, 21.0),
            part_index,
            top,
            "MT45 Start",
        )

        takeout_start = top_start + rng.uniform(144.6, 150.0)
        takeout = add_subprocess(
            "abstraction:sub:Turm MT45 Take Out",
            takeout_start,
            top_start + rng.uniform(224.0, 232.0),
            part_index,
            top,
            "MT45 Take Out",
        )
        add_subprocess(
            "abstraction:sub:MT45 Check Operational Safety",
            takeout_start + 0.2,
            takeout_start + 8.0,
            part_index,
            takeout,
            "MT45 Check Operational Safety",
        )
        grip = add_subprocess(
            "abstraction:sub:Turm IRB2600 Grip",
            takeout_start + 9.0,
            takeout_start + 38.0,
            part_index,
            takeout,
            "IRB2600 Grip",
        )
        extract = add_subprocess(
            "abstraction:sub:Turm IRB2600 Extract",
            takeout_start + 49.0,
            takeout_start + 75.0,
            part_index,
            takeout,
            "IRB2600 Extract",
        )

        lathe_machine = add_subprocess(
            "abstraction:sub:Machining V2",
            top_start + 18.5,
            top_start + rng.uniform(136.0, 140.0),
            part_index,
            turn,
            "Machining",
            equipment="MaxxTurn45",
        )
        machine_subs.append(lathe_machine)

        getter = add_subprocess(
            "abstraction:sub:X MT45 Control Getter",
            top_start + 3.0,
            top_start + 17.5,
            part_index,
            turn,
            "Detection Machining",
        )
        getter_subs.append(getter)

        for local_index, parent in enumerate(
            (
                mt45_sp,
                mt45_sp,
                mt45_start,
                mt45_start,
                takeout,
                takeout,
            )
        ):
            parent_rec = subprocesses[parent]
            span = parent_rec["end"] - parent_rec["start"]
            child_start = (
                parent_rec["start"]
                + 0.12
                + (local_index % 2) * span * 0.34
            )
            child_end = min(
                parent_rec["end"] - 0.10,
                child_start + max(0.25, span * 0.28),
            )
            setter = add_subprocess(
                "abstraction:sub:X MT45 Control Setter",
                child_start,
                child_end,
                part_index,
                parent,
                "Set Var",
            )
            setter_subs.append(setter)

        if item == "wr04":
            route_start = top_start + rng.uniform(
                247.0,
                min(293.0, duration - 42.0),
            )
            route = add_subprocess(
                "abstraction:sub:Turm IRB2600 Measure and Tray",
                route_start,
                top_end - 2.0,
                part_index,
                top,
                "Robot Measuring and Traying",
            )
            keyence = add_subprocess(
                "abstraction:sub:Turm Keyence Measurement",
                route_start + 15.0,
                min(top_end - 25.0, route_start + 55.0),
                part_index,
                top,
                "Measure with Keyence",
            )
            key_machine = add_subprocess(
                "abstraction:sub:Machining V2",
                subprocesses[keyence]["start"] + 1.0,
                subprocesses[keyence]["end"] - 1.0,
                part_index,
                keyence,
                "Machining",
                equipment="keyence",
            )
            machine_subs.append(key_machine)
        else:
            route_start = top_start + rng.uniform(
                239.0,
                min(280.0, duration - 35.0),
            )
            route = add_subprocess(
                "abstraction:sub:Turm IRB2600 Tray",
                route_start,
                top_end - 2.0,
                part_index,
                top,
                "Robot Traying",
            )

        route_by_part[part_index] = route

        movement_defs = [
            (
                grip,
                "IRB2600 Portal to Door",
                "abstraction:sub:IRB2600 Portal to Door",
                1.0,
                8.0,
            ),
            (
                grip,
                "IRB2600 Door to GS",
                "abstraction:sub:IRB2600 Door to GS",
                9.0,
                17.0,
            ),
            (
                grip,
                "IRB2600 GS to Take",
                "abstraction:sub:IRB2600 GS to Take",
                18.0,
                27.0,
            ),
            (
                extract,
                "IRB2600 Take to GS",
                "abstraction:sub:IRB2600 Take to GS",
                1.0,
                9.0,
            ),
            (
                extract,
                "IRB2600 GS to Door",
                "abstraction:sub:IRB2600 IRB2600 GS to Door",
                10.0,
                19.0,
            ),
            (
                route,
                "IRB2600 Door to Portal",
                "abstraction:sub:IRB2600 Door to Portal",
                1.0,
                min(
                    18.0,
                    subprocesses[route]["end"]
                    - subprocesses[route]["start"]
                    - 1.0,
                ),
            ),
        ]

        created_movements = []
        for (
            parent,
            caller,
            type_name,
            off_a,
            off_b,
        ) in movement_defs:
            parent_start = subprocesses[parent]["start"]
            movement = add_subprocess(
                type_name,
                parent_start + off_a,
                parent_start + off_b,
                part_index,
                parent,
                caller,
            )
            movement_subs.append(movement)
            created_movements.append(movement)

        for movement in (
            created_movements[0],
            created_movements[3],
        ):
            rec = subprocesses[movement]
            execution = add_subprocess(
                "abstraction:sub:X IRB2600 Execution",
                rec["start"] + 0.2,
                rec["end"] - 0.2,
                part_index,
                movement,
                "Set Program",
            )
            robot_execution_subs.append(execution)

    if (
        len(subprocesses) != 217
        or len(interface_specs) != 216
    ):
        raise AssertionError(
            "Internal workflow construction did not produce "
            "the source hierarchy"
        )

    used_activity_counts = Counter(
        spec["base"] for spec in activity_specs
    )

    owner_groups: Dict[str, List[str]] = {
        "batch": [batch],
        "top": list(top_by_part.values()),
        "machine": machine_subs,
        "robot_execution": robot_execution_subs,
        "getter": getter_subs,
        "setter": setter_subs,
        "movement": movement_subs,
        "route": list(route_by_part.values()),
        "all": list(subprocesses),
    }

    preferred_group = {
        "Get Order Details": "batch",
        "Inititalize Batch": "batch",
        "Next QR, Order Item and Tray": "batch",
        "Get Machine State Cancelled": "batch",
        "Wait for MT45 Takeout": "batch",
        "Not First Part": "batch",
        "Fetch": "machine",
        "Check State": "robot_execution",
        "Read Var": "getter",
        "Fetch Program": "movement",
        "Logon": "robot_execution",
        "Assign Program": "robot_execution",
        "Start": "robot_execution",
        "Set Fetch Endpoint": "movement",
        "Assemble Porgram": "movement",
        "Replace Patterns with Params": "top",
        "Set Endpoint": "top",
        "Set Paths": "top",
        "Correct  Queue": "machine",
        "Get Current Time": "route",
        "Wait For Unload End": "route",
        "Signal Unloading End": "route",
        "Get Magazine X": "getter",
        "Get Magazine Z": "getter",
        "Aux On": "setter",
        "Mode Auto": "setter",
        "Door Close": "setter",
        "NC Start": "setter",
        "Open Clamp 2": "setter",
        "Reset Clamp 2": "setter",
        "Taken Out": "top",
        "Set Dummy Program": "top",
        "Select Dummy Program": "top",
        "Signal Keyence End": "top",
        "IRB2600 Unload to Tray Down": "route",
        "IRB2600 Door to Measure": "route",
        "IRB2600 Measure to Door": "route",
        "Move Down": "route",
        "Move Up": "route",
        "wait": "route",
    }

    for base, target_count in ACTIVITY_COUNTS.items():
        if base == "external":
            continue

        remaining = target_count - used_activity_counts[base]
        if remaining < 0:
            raise AssertionError(
                f"Workflow uses too many {base} activity instances"
            )

        group = preferred_group.get(base, "all")
        owners = owner_groups[group]

        for ordinal in range(remaining):
            if base == "Signal Keyence End":
                owners = [
                    top_by_part[index]
                    for index, part in enumerate(PARTS)
                    if part[1] == "wr04"
                ]

            owner = owners[ordinal % len(owners)]
            activity_specs.append(
                {
                    "base": base,
                    "owner": owner,
                    "child": None,
                    "interface": False,
                    "async": False,
                    "part": subprocesses[owner]["part"],
                }
            )

    expected_activity_instances = Counter(
        {
            key: value
            for key, value in ACTIVITY_COUNTS.items()
            if key != "external"
        }
    )
    actual_activity_instances = Counter(
        spec["base"] for spec in activity_specs
    )

    if actual_activity_instances != expected_activity_instances:
        raise AssertionError(
            "Activity-instance derivation does not match "
            "the source workflow"
        )

    external_object = "external"
    objects[external_object] = {
        "ocel:type": "abstraction:lifecycle:external",
        "ocel:ovmap": {},
    }

    activity_records: List[Dict[str, Any]] = []
    for spec in activity_specs:
        activity_id = ids.new("activity", compact=True)
        spec["activity_id"] = activity_id
        objects[activity_id] = {
            "ocel:type": (
                f"abstraction:lifecycle:{spec['base']}"
            ),
            "ocel:ovmap": {},
        }
        activity_records.append(spec)

    id_id_map: Dict[Tuple[str, str], str] = {}
    for sub_type in SUBPROCESS_TYPES:
        bases = sorted(
            {
                spec["base"]
                for spec in activity_specs
                if subprocesses[spec["owner"]]["type"]
                == sub_type
            }
        )
        for index, base in enumerate(bases, 1):
            id_id_map[(sub_type, base)] = f"a{index}"

    def add_event(
        when_seconds: float,
        owner: str,
        base: str,
        transition: str,
        activity_id: str,
        child: Optional[str] = None,
        component: Optional[str] = None,
        payload_hint: Optional[dict] = None,
    ) -> None:
        nonlocal sequence

        if transition not in TRANSITION_TO_LIFECYCLE:
            raise ValueError(
                f"Unsupported transition {transition}"
            )

        event_id = ids.new("event")
        owner_rec = subprocesses[owner]
        omap = [owner]

        if activity_id != "__UNLINKED_WAIT__":
            omap.append(activity_id)

        if (
            child is not None
            and transition
            in {
                "task/instantiation",
                "subprocess/receiving",
            }
        ):
            omap.append(child)
        elif component is not None:
            omap.append(component_ids[component])

        if base == "external":
            modeled_id = "external"
        else:
            modeled_id = id_id_map.get(
                (owner_rec["type"], base),
                "a_external",
            )

        if base == "external":
            cpee_activity_uuid = "external"
        elif activity_id == "__UNLINKED_WAIT__":
            cpee_activity_uuid = ids.new(
                "unlinked-wait",
                compact=True,
            )
        else:
            cpee_activity_uuid = activity_id

        vmap: Dict[str, Any] = {
            "concept:instance": owner_rec["concept"],
            "id:id": modeled_id,
            "cpee:activity_uuid": cpee_activity_uuid,
            "lifecycle:transition": (
                TRANSITION_TO_LIFECYCLE[transition]
            ),
            "cpee:lifecycle:transition": transition,
        }

        if (
            child is not None
            and transition
            in {
                "task/instantiation",
                "subprocess/receiving",
            }
        ):
            vmap["sub:root"] = subprocesses[child][
                "type"
            ].removeprefix("abstraction:sub:")

        timestamp = base_time + timedelta(
            seconds=when_seconds
        )
        event = {
            "ocel:activity": f"{base}§{transition}",
            "ocel:timestamp": _iso(timestamp),
            "ocel:omap": omap,
            "ocel:vmap": vmap,
            "_payload_hint": payload_hint,
            "_owner": owner,
        }
        event_rows.append(
            (
                timestamp,
                sequence,
                event_id,
                event,
            )
        )
        sequence += 1

    optional_nested_candidates = [
        spec
        for spec in interface_specs
        if not spec["async"]
        and spec["base"] not in FOUR_EVENT_READS
    ]
    optional_nested_data = {
        id(spec)
        for spec in optional_nested_candidates[:108]
    }

    subreceive_specs = {
        id(spec)
        for spec in interface_specs
        if spec["base"] != "Spawn Production"
        and not (
            spec["base"] == "Machining"
            and subprocesses[spec["child"]]["equipment"]
            == "keyence"
        )
    }

    if len(subreceive_specs) != 204:
        raise AssertionError(
            "Expected 204 synchronous subprocess "
            "receiving bundles"
        )

    extra_receiving_budget = 31
    fetch_receive_times: Dict[
        str, List[float]
    ] = defaultdict(list)
    fetch_equipment_ordinals: Counter[str] = Counter()
    robot_stream_times: List[
        Tuple[float, int]
    ] = []
    activity_ordinals: Counter[str] = Counter()

    for spec in activity_records:
        base = spec["base"]
        owner = spec["owner"]
        owner_rec = subprocesses[owner]
        activity_id = spec["activity_id"]
        ordinal = activity_ordinals[base]
        activity_ordinals[base] += 1

        component = None
        if base == "Check State":
            component = "IRB2600"
        elif base == "Fetch":
            component = owner_rec["equipment"]

        if spec["interface"]:
            child = spec["child"]
            child_rec = subprocesses[child]
            start = child_rec["start"] - 0.090

            add_event(
                start,
                owner,
                base,
                "activity/calling",
                activity_id,
            )
            add_event(
                start + 0.025,
                owner,
                base,
                "activity/receiving",
                activity_id,
            )
            add_event(
                child_rec["start"],
                owner,
                base,
                "task/instantiation",
                activity_id,
                child=child,
            )

            if base in FOUR_EVENT_READS:
                add_event(
                    child_rec["start"] + 0.005,
                    owner,
                    base,
                    "dataelements/change",
                    activity_id,
                )

            if spec["async"]:
                add_event(
                    child_rec["start"] + 0.030,
                    owner,
                    base,
                    "activity/done",
                    activity_id,
                )
            else:
                end = child_rec["end"] + 0.010

                add_event(
                    end,
                    owner,
                    base,
                    "activity/receiving",
                    activity_id,
                )

                if id(spec) in subreceive_specs:
                    add_event(
                        end + 0.010,
                        owner,
                        base,
                        "subprocess/receiving",
                        activity_id,
                        child=child,
                    )

                if id(spec) in optional_nested_data:
                    add_event(
                        end + 0.020,
                        owner,
                        base,
                        "dataelements/change",
                        activity_id,
                    )

                add_event(
                    end + 0.035,
                    owner,
                    base,
                    "activity/done",
                    activity_id,
                )

            continue

        span = owner_rec["end"] - owner_rec["start"]
        fraction = (
            (
                ordinal * 0.61803398875
                + rng.random() * 0.07
            )
            % 0.86
        ) + 0.07
        start = (
            owner_rec["start"]
            + span * fraction
        )
        duration = max(
            0.035,
            min(
                1.8,
                span
                * (
                    0.008
                    + rng.random() * 0.012
                ),
            ),
        )

        if base == "Signal Keyence End":
            start = owner_rec["end"] - 1.2

        if base in DATA_UPDATE_ACTIVITIES:
            hint = (
                {"part_index": ordinal}
                if base
                == "Next QR, Order Item and Tray"
                else None
            )
            add_event(
                start,
                owner,
                base,
                "dataelements/change",
                activity_id,
                payload_hint=hint,
            )
            add_event(
                start + duration,
                owner,
                base,
                "activity/done",
                activity_id,
            )

        elif base in ENDPOINT_UPDATE_ACTIVITIES:
            add_event(
                start,
                owner,
                base,
                "endpoints/change",
                activity_id,
            )
            add_event(
                start + duration,
                owner,
                base,
                "activity/done",
                activity_id,
            )

        elif base == "Set Fetch Endpoint":
            add_event(
                start,
                owner,
                base,
                "dataelements/change",
                activity_id,
            )
            add_event(
                start + duration * 0.45,
                owner,
                base,
                "endpoints/change",
                activity_id,
            )
            add_event(
                start + duration,
                owner,
                base,
                "activity/done",
                activity_id,
            )

        elif base == "Fetch":
            equipment = owner_rec["equipment"]
            equipment_ordinal = (
                fetch_equipment_ordinals[equipment]
            )
            fetch_equipment_ordinals[equipment] += 1

            if equipment == "MaxxTurn45":
                receive_total = (
                    92
                    if equipment_ordinal < 5
                    else 91
                )
            else:
                receive_total = (
                    12
                    if equipment_ordinal < 3
                    else 11
                )

            add_event(
                start,
                owner,
                base,
                "activity/calling",
                activity_id,
                component=equipment,
            )

            for receive_index in range(receive_total):
                point_time = (
                    start
                    + duration
                    * (receive_index + 1)
                    / (receive_total + 1)
                )
                add_event(
                    point_time,
                    owner,
                    base,
                    "activity/receiving",
                    activity_id,
                    component=equipment,
                )
                fetch_receive_times[equipment].append(
                    point_time
                )

            add_event(
                start + duration,
                owner,
                base,
                "activity/done",
                activity_id,
                component=equipment,
            )

        else:
            add_event(
                start,
                owner,
                base,
                "activity/calling",
                activity_id,
                component=component,
            )
            add_event(
                start + duration * 0.28,
                owner,
                base,
                "activity/receiving",
                activity_id,
                component=component,
            )

            if (
                extra_receiving_budget
                and base not in FOUR_EVENT_READS
            ):
                add_event(
                    start + duration * 0.42,
                    owner,
                    base,
                    "activity/receiving",
                    activity_id,
                    component=component,
                )
                extra_receiving_budget -= 1

            if base in FOUR_EVENT_READS:
                add_event(
                    start + duration * 0.62,
                    owner,
                    base,
                    "dataelements/change",
                    activity_id,
                    component=component,
                )

                if base == "Check State":
                    robot_stream_times.append(
                        (
                            start + duration * 0.62,
                            owner_rec["part"] or 0,
                        )
                    )

            add_event(
                start + duration,
                owner,
                base,
                "activity/done",
                activity_id,
                component=component,
            )

    if extra_receiving_budget != 0:
        raise AssertionError(
            "Could not allocate source extra receiving messages"
        )

    wr04_routes = [
        route_by_part[index]
        for index, part in enumerate(PARTS)
        if part[1] == "wr04"
    ]

    for route in wr04_routes:
        rec = subprocesses[route]
        add_event(
            rec["start"]
            + (rec["end"] - rec["start"]) * 0.62,
            route,
            "Wait While Moving Up",
            "activity/done",
            "__UNLINKED_WAIT__",
        )

    for sub_index, (sub_id, rec) in enumerate(
        subprocesses.items()
    ):
        external_time = rec["start"] + min(
            0.02,
            (rec["end"] - rec["start"]) / 20.0,
        )

        for offset, transition in enumerate(
            (
                "dataelements/change",
                "endpoints/change",
                "attributes/change",
                "description/change",
            )
        ):
            add_event(
                external_time + offset * 0.001,
                sub_id,
                "external",
                transition,
                external_object,
            )

        state_count = 4 if sub_index < 3 else 3

        for state_index in range(state_count):
            fraction = (
                state_index + 1
            ) / (state_count + 1)
            add_event(
                rec["start"]
                + (
                    rec["end"]
                    - rec["start"]
                )
                * fraction,
                sub_id,
                "external",
                "state/change",
                external_object,
            )

    transition_counts = Counter(
        row[3]["ocel:vmap"][
            "cpee:lifecycle:transition"
        ]
        for row in event_rows
    )
    if transition_counts != Counter(
        EXPECTED_TRANSITIONS
    ):
        raise AssertionError(
            f"Internal transition mismatch: "
            f"{transition_counts}"
        )

    event_rows.sort(
        key=lambda row: (row[0], row[1])
    )
    events: Dict[str, Dict[str, Any]] = {
        event_id: event
        for _, _, event_id, event in event_rows
    }

    for event in list(events.values())[:7128]:
        base = (
            _event_base(event)
            .replace(" ", "_")
            .lower()
        )
        event["ocel:vmap"][
            "concept:endpoint"
        ] = (
            "https://cpee.example.invalid/services/"
            + base
        )

    def payload_for(
        event: Dict[str, Any],
        serial: int,
    ) -> List[Dict[str, Any]]:
        base = _event_base(event)
        owner = event.pop("_owner")
        hint = event.pop("_payload_hint")
        part_index = subprocesses[owner]["part"]

        if hint and "part_index" in hint:
            part_index = hint["part_index"]

        if base == "Get Order Details":
            return [
                {
                    "name": "order",
                    "value": {
                        "order1": [
                            "wr04",
                            "wr04",
                            "wr08",
                            "wr08",
                            "wr08",
                        ],
                        "order2": [
                            "wr04",
                            "wr04",
                            "wr08",
                        ],
                    },
                }
            ]

        if base == "Inititalize Batch":
            return [
                {
                    "name": "order",
                    "value": {
                        "order1": [
                            "wr04",
                            "wr04",
                            "wr08",
                            "wr08",
                            "wr08",
                        ],
                        "order2": [
                            "wr04",
                            "wr04",
                            "wr08",
                        ],
                    },
                },
                {
                    "name": "tray_positions",
                    "value": _all_tray_positions(),
                },
                {
                    "name": "trays",
                    "value": [
                        [row, column]
                        for row in range(8)
                        for column in range(6)
                    ],
                },
                {
                    "name": "program_patterns",
                    "value": [
                        "yyyy",
                        "xxxx",
                        "aaaa",
                    ],
                },
                {
                    "name": "programs",
                    "value": {
                        "wr04": PROGRAM_URL,
                        "wr08": PROGRAM_URL,
                    },
                },
                {
                    "name": "qr",
                    "value": "*268MFA466*TZHZE 001",
                },
                {
                    "name": "seed",
                    "value": 1,
                },
                {
                    "name": "initial",
                    "value": None,
                },
            ]

        if (
            base
            == "Next QR, Order Item and Tray"
            and part_index is not None
        ):
            (
                order_id,
                order_item,
                feed,
                depth,
            ) = PARTS[part_index]

            return [
                {
                    "name": "order_id",
                    "value": order_id,
                },
                {
                    "name": "order_item",
                    "value": order_item,
                },
                {
                    "name": "order_list",
                    "value": [
                        [a, b]
                        for a, b, _, _ in PARTS[
                            part_index + 1 :
                        ]
                    ],
                },
                {
                    "name": "count",
                    "value": part_index,
                },
                {
                    "name": "initial",
                    "value": part_index == 0,
                },
                {
                    "name": "current_tray",
                    "value": _tray_position(
                        part_index
                    ),
                },
                {
                    "name": "program_params",
                    "value": [
                        feed,
                        depth,
                        f"{order_id}: {order_item}",
                    ],
                },
                {
                    "name": "program_patterns",
                    "value": [
                        "yyyy",
                        "xxxx",
                        "aaaa",
                    ],
                },
                {
                    "name": "programs",
                    "value": {
                        "wr04": PROGRAM_URL,
                        "wr08": PROGRAM_URL,
                    },
                },
                {
                    "name": "qr",
                    "value": "*268MFA466*TZHZE 000",
                },
            ]

        if base in {
            "Fetch Program",
            "Assemble Porgram",
            "Replace Patterns with Params",
            "Set Program",
        }:
            idx = (
                0
                if part_index is None
                else part_index
            )
            (
                order_id,
                order_item,
                feed,
                depth,
            ) = PARTS[idx]

            return [
                {
                    "name": "program_url",
                    "value": PROGRAM_URL,
                },
                {
                    "name": "program_params",
                    "value": [
                        feed,
                        depth,
                        f"{order_id}: {order_item}",
                    ],
                },
                {
                    "name": "patterns",
                    "value": [
                        "yyyy",
                        "xxxx",
                        "aaaa",
                    ],
                },
                {
                    "name": "program",
                    "value": (
                        NC_PROGRAM
                        if serial % 2 == 0
                        else RAPID_PROGRAM
                    ),
                },
                {
                    "name": "content",
                    "value": (
                        RAPID_PROGRAM
                        if serial % 2 == 0
                        else NC_PROGRAM
                    ),
                },
            ]

        if base in {
            "Set Var",
            "Read Var",
            "Get Magazine X",
            "Get Magazine Z",
        }:
            return [
                {
                    "name": "endpoint_url",
                    "value": (
                        "opc.tcp://192.168.0.45:4840"
                    ),
                },
                {
                    "name": "namespace_index",
                    "value": "2",
                },
                {
                    "name": "identifier_type",
                    "value": "string",
                },
                {
                    "name": "identifier",
                    "value": (
                        "MaxxTurn45/State/progStatus"
                    ),
                },
                {
                    "name": "value",
                    "value": (
                        "true"
                        if serial % 3
                        else True
                    ),
                },
                {
                    "name": "check_value",
                    "value": [
                        True,
                        "true",
                        "false",
                    ][serial % 3],
                },
                {
                    "name": "state",
                    "value": "true",
                },
            ]

        if base == "Check State":
            return [
                {
                    "name": "state",
                    "value": (
                        "Active"
                        if serial % 4
                        else "Finished"
                    ),
                },
                {
                    "name": "started",
                    "value": True,
                },
                {
                    "name": "check_var1",
                    "value": "StateMachine",
                },
                {
                    "name": "check_var2",
                    "value": (
                        "Rapid.ExecutionStatus"
                    ),
                },
                {
                    "name": "xxx",
                    "value": {
                        "timestamp": event[
                            "ocel:timestamp"
                        ],
                        "id": "StateMachine",
                        "name": "StateMachine",
                        "path": (
                            "IRB2600/StateMachine"
                        ),
                        "source": "IRB2600",
                        "type": "string",
                        "value": "Active",
                    },
                },
                {
                    "name": "yyy",
                    "value": {
                        "timestamp": event[
                            "ocel:timestamp"
                        ],
                        "id": "ExecutionStatus",
                        "name": (
                            "Rapid.ExecutionStatus"
                        ),
                        "path": "IRB2600/Rapid",
                        "source": "IRB2600",
                        "type": "string",
                        "value": "Running",
                    },
                },
            ]

        if base in {
            "Set Endpoint",
            "Set Fetch Endpoint",
            "Correct  Queue",
        }:
            return [
                {
                    "name": "endpoints",
                    "value": None,
                },
                {
                    "name": "attributes",
                    "value": None,
                },
                {
                    "name": "customization",
                    "value": None,
                },
                {
                    "name": "fetch",
                    "value": (
                        "https://cpee.example.invalid/fetch"
                    ),
                },
                {
                    "name": "behavior",
                    "value": (
                        "fork_running"
                        if serial % 2
                        else "wait_running"
                    ),
                },
                {
                    "name": "queue",
                    "value": [],
                },
            ]

        if base == "Get Current Time":
            return [
                {
                    "name": "at",
                    "value": event[
                        "ocel:timestamp"
                    ],
                },
                {
                    "name": "timing",
                    "value": "source-local",
                },
            ]

        generic = [
            {
                "name": "info",
                "value": (
                    "CPEE immutable scope snapshot "
                    f"{serial}"
                ),
            },
            {
                "name": "modeltype",
                "value": "CPEE",
            },
            {
                "name": "status",
                "value": (
                    "running"
                    if serial % 5
                    else "finished"
                ),
            },
            {
                "name": "behavior",
                "value": (
                    "wait_running"
                    if serial % 2
                    else "fork_running"
                ),
            },
            {
                "name": "attributes",
                "value": None,
            },
            {
                "name": "endpoints",
                "value": None,
            },
        ]

        if part_index is not None:
            (
                order_id,
                order_item,
                feed,
                depth,
            ) = PARTS[part_index]
            generic.extend(
                [
                    {
                        "name": "order_id",
                        "value": order_id,
                    },
                    {
                        "name": "order_item",
                        "value": order_item,
                    },
                    {
                        "name": "program_params",
                        "value": [
                            feed,
                            depth,
                            (
                                f"{order_id}: "
                                f"{order_item}"
                            ),
                        ],
                    },
                    {
                        "name": "current_tray",
                        "value": _tray_position(
                            part_index
                        ),
                    },
                ]
            )

        return generic

    data_object_count = 0

    events_for_data = [
        event
        for event in events.values()
        if event["ocel:vmap"][
            "cpee:lifecycle:transition"
        ]
        == "dataelements/change"
    ]
    events_for_data.extend(
        event
        for event in events.values()
        if event["ocel:vmap"][
            "cpee:lifecycle:transition"
        ]
        == "activity/receiving"
    )

    chosen_events = events_for_data[:3134]
    if len(chosen_events) != 3134:
        raise AssertionError(
            "Insufficient events for source data references"
        )

    for serial, event in enumerate(chosen_events):
        data_id = ids.new("data")
        objects[data_id] = {
            "ocel:type": "data",
            "ocel:ovmap": {
                "list": payload_for(event, serial)
            },
        }
        event["ocel:vmap"]["data"] = data_id
        data_object_count += 1

    for event in events.values():
        event.pop("_owner", None)
        event.pop("_payload_hint", None)

    while data_object_count < 3605:
        data_id = ids.new("data")
        mode = data_object_count % 4

        if mode == 0:
            value: Any = None
        elif mode == 1:
            value = []
        else:
            value = [
                {
                    "name": "info",
                    "value": (
                        "retained CPEE snapshot "
                        f"{data_object_count}"
                    ),
                },
                {
                    "name": "customization",
                    "value": None,
                },
            ]

        objects[data_id] = {
            "ocel:type": "data",
            "ocel:ovmap": {
                "list": value
            },
        }
        data_object_count += 1

    stream_count = 0
    machine_times = sorted(
        fetch_receive_times["MaxxTurn45"]
    )
    keyence_times = sorted(
        fetch_receive_times["keyence"]
    )

    if (
        len(machine_times) != 733
        or len(keyence_times) != 47
        or len(robot_stream_times) != 590
    ):
        raise AssertionError(
            "Sensor acquisition counts do not match "
            "their controlling activities: "
            f"machine={len(machine_times)}, "
            f"keyence={len(keyence_times)}, "
            f"robot={len(robot_stream_times)}"
        )

    for index, time_value in enumerate(machine_times):
        if index < 717:
            point_count = 334
        elif index < 725:
            point_count = 0
        else:
            point_count = 214

        part_index = min(
            7,
            int(time_value // 235.0),
        )
        stream_id = ids.new("stream")
        points = _machine_points(
            point_count,
            index,
            base_time + timedelta(seconds=time_value),
            part_index,
            rng,
        )
        objects[stream_id] = {
            "ocel:type": "stream",
            "ocel:ovmap": {
                "list": (
                    _stream_headers("MaxxTurn45")
                    + points
                )
            },
        }
        stream_count += 1

    for index, (
        time_value,
        part_index,
    ) in enumerate(sorted(robot_stream_times)):
        stream_id = ids.new("stream")
        points = _robot_points(
            index,
            base_time + timedelta(seconds=time_value),
        )
        objects[stream_id] = {
            "ocel:type": "stream",
            "ocel:ovmap": {
                "list": (
                    _stream_headers("IRB2600")
                    + points
                )
            },
        }
        stream_count += 1

    keyence_part_indices = [
        index
        for index, part in enumerate(PARTS)
        if part[1] == "wr04"
    ]

    for index, time_value in enumerate(keyence_times):
        if index < 39:
            point_count = 334
        elif index < 43:
            point_count = 0
        else:
            point_count = 216

        part_index = keyence_part_indices[
            min(3, index // 12)
        ]
        stream_id = ids.new("stream")
        points = _keyence_points(
            point_count,
            index,
            base_time + timedelta(seconds=time_value),
            part_index,
            rng,
        )
        objects[stream_id] = {
            "ocel:type": "stream",
            "ocel:ovmap": {
                "list": (
                    _stream_headers("keyence")
                    + points
                )
            },
        }
        stream_count += 1

    if stream_count != 1370:
        raise AssertionError(
            "Internal stream count mismatch"
        )

    lifecycle_types = [
        f"abstraction:lifecycle:{name}"
        for name in ACTIVITY_COUNTS
    ]
    lifecycle_types.append(
        "abstraction:lifecycle:Wait While Moving Up"
    )

    declared_types = (
        SUBPROCESS_TYPES
        + lifecycle_types
        + [
            "data",
            "stream",
            "view:component:IRB2600",
            "view:component:MaxxTurn45",
            "view:component:keyence",
            "view:component:NA",
        ]
    )

    if (
        len(declared_types) != 94
        or len(set(declared_types)) != 94
    ):
        raise AssertionError(
            "The exact source dialect requires "
            "94 distinct declared object types"
        )

    ocel = {
        "ocel:global-log": {
            "ocel:attribute-names": [
                "concept:instance",
                "id:id",
                "cpee:activity_uuid",
                "lifecycle:transition",
                "cpee:lifecycle:transition",
                "concept:endpoint",
                "data",
                "sub:root",
                "cpee:uuid",
            ],
            "ocel:object-types": declared_types,
            "ocel:version": "0.1",
            "ocel:ordering": "timestamp",
        },
        "ocel:global-event": {
            "ocel:activity": "__INVALID__"
        },
        "ocel:global-object": {
            "ocel:type": "__INVALID__"
        },
        "ocel:events": events,
        "ocel:objects": objects,
    }

    validate_ocel(ocel)
    return ocel


# Validate both the OCEL 0.1 structure and the source-specific invariants of the
# synthetic Nine Rooks log. In addition to checking references, lifecycle
# consistency, subprocess relationships, production routes, and sensor values,
# this validator enforces the exact event, object, transition, and stream counts
# expected from the fixed eight-part source scenario. It is therefore a bespoke
# source-fidelity validator rather than a general-purpose OCEL schema validator.
# Any violation raises TypeError or ValueError and prevents the log from being
# returned or saved.
def validate_ocel(ocel: dict) -> None:
    required_top = {
        "ocel:global-log",
        "ocel:global-event",
        "ocel:global-object",
        "ocel:events",
        "ocel:objects",
    }

    if set(ocel) != required_top:
        raise ValueError(
            "Top-level OCEL keys must be exactly "
            f"{sorted(required_top)}"
        )

    global_log = ocel["ocel:global-log"]
    if (
        global_log.get("ocel:version") != "0.1"
        or global_log.get("ocel:ordering")
        != "timestamp"
    ):
        raise ValueError(
            "The log must use OCEL 0.1 "
            "with timestamp ordering"
        )

    if ocel["ocel:global-event"] != {
        "ocel:activity": "__INVALID__"
    }:
        raise ValueError(
            "Invalid legacy global-event defaults"
        )

    if ocel["ocel:global-object"] != {
        "ocel:type": "__INVALID__"
    }:
        raise ValueError(
            "Invalid legacy global-object defaults"
        )

    events = ocel["ocel:events"]
    objects = ocel["ocel:objects"]

    if (
        not isinstance(events, dict)
        or not isinstance(objects, dict)
    ):
        raise TypeError(
            "ocel:events and ocel:objects must be "
            "dictionaries keyed by IDs"
        )

    if len(events) != 9268 or len(objects) != 6991:
        raise ValueError(
            "Exact-source-fidelity output must contain "
            "9,268 events and 6,991 objects"
        )

    if set(events) & set(objects):
        raise ValueError(
            "Event and object IDs must be globally disjoint"
        )

    declared = global_log.get("ocel:object-types")
    if (
        not isinstance(declared, list)
        or len(declared) != 94
        or len(set(declared)) != 94
    ):
        raise ValueError(
            "Exactly 94 distinct object types "
            "must be declared"
        )

    instantiated_types = {
        obj.get("ocel:type")
        for obj in objects.values()
    }

    if len(instantiated_types) != 93:
        raise ValueError(
            "Exactly 93 object types must be instantiated"
        )

    wait_type = (
        "abstraction:lifecycle:Wait While Moving Up"
    )

    if wait_type not in declared:
        raise ValueError(
            "The source-only Wait While Moving Up "
            "type must be declared"
        )

    if wait_type in instantiated_types:
        raise ValueError(
            "Exact source fidelity preserves the missing "
            "Wait While Moving Up object instances"
        )

    if not instantiated_types <= set(declared):
        raise ValueError(
            "An instantiated object type is not "
            "globally declared"
        )

    transition_counts: Counter[str] = Counter()
    omap_sizes: Counter[int] = Counter()
    concept_owner: Dict[int, str] = {}
    endpoint_count = 0
    data_reference_count = 0
    sub_root_count = 0
    activity_events: Dict[
        str,
        List[Tuple[datetime, Dict[str, Any]]],
    ] = defaultdict(list)
    data_references: List[str] = []
    wait_events = 0

    subprocess_object_ids = {
        object_id
        for object_id, obj in objects.items()
        if str(
            obj.get("ocel:type", "")
        ).startswith("abstraction:sub:")
    }
    stream_ids = {
        object_id
        for object_id, obj in objects.items()
        if obj.get("ocel:type") == "stream"
    }

    for event_id, event in events.items():
        try:
            activity_label = event["ocel:activity"]
            timestamp = _parse_time(
                event["ocel:timestamp"]
            )
            omap = event["ocel:omap"]
            vmap = event["ocel:vmap"]
        except (
            KeyError,
            TypeError,
            ValueError,
        ) as exc:
            raise ValueError(
                f"Malformed event {event_id}"
            ) from exc

        if timestamp.utcoffset() != timedelta(hours=1):
            raise ValueError(
                f"Event {event_id} does not use "
                "the required +01:00 offset"
            )

        if (
            not isinstance(omap, list)
            or not 1 <= len(omap) <= 3
            or len(set(omap)) != len(omap)
        ):
            raise ValueError(
                f"Invalid ocel:omap on {event_id}"
            )

        if any(
            reference not in objects
            for reference in omap
        ):
            raise ValueError(
                f"Unresolved ocel:omap reference "
                f"on {event_id}"
            )

        if any(
            reference in stream_ids
            for reference in omap
        ):
            raise ValueError(
                "Source-fidelity stream bundles "
                "must remain unlinked"
            )

        transition = vmap.get(
            "cpee:lifecycle:transition"
        )
        if transition not in TRANSITION_TO_LIFECYCLE:
            raise ValueError(
                f"Unknown CPEE transition on {event_id}"
            )

        base = (
            activity_label.rsplit("§", 1)[0]
            if "§" in activity_label
            else None
        )
        if (
            base is None
            or activity_label
            != f"{base}§{transition}"
        ):
            raise ValueError(
                f"Event label/transition mismatch "
                f"on {event_id}"
            )

        if (
            vmap.get("lifecycle:transition")
            != TRANSITION_TO_LIFECYCLE[transition]
        ):
            raise ValueError(
                f"Lifecycle mapping mismatch "
                f"on {event_id}"
            )

        if (
            not isinstance(
                vmap.get("concept:instance"),
                int,
            )
            or not isinstance(
                vmap.get("id:id"),
                str,
            )
        ):
            raise ValueError(
                f"Invalid mandatory event attributes "
                f"on {event_id}"
            )

        owner = omap[0]
        if owner not in subprocess_object_ids:
            raise ValueError(
                f"First relation of {event_id} must be "
                "its owning subprocess"
            )

        concept = vmap["concept:instance"]
        if (
            concept in concept_owner
            and concept_owner[concept] != owner
        ):
            raise ValueError(
                "One concept:instance maps to multiple "
                "owning subprocesses"
            )
        concept_owner[concept] = owner

        activity_uuid = vmap.get(
            "cpee:activity_uuid"
        )

        if base == "Wait While Moving Up":
            wait_events += 1
            if (
                len(omap) != 1
                or transition != "activity/done"
                or activity_uuid in objects
            ):
                raise ValueError(
                    "Invalid exact-source "
                    "Wait While Moving Up quirk"
                )
        else:
            if (
                len(omap) < 2
                or omap[1] != activity_uuid
                or activity_uuid not in objects
            ):
                raise ValueError(
                    f"Activity relation mismatch "
                    f"on {event_id}"
                )

            expected_type = (
                f"abstraction:lifecycle:{base}"
            )
            if (
                objects[activity_uuid].get("ocel:type")
                != expected_type
            ):
                raise ValueError(
                    f"Activity object type mismatch "
                    f"on {event_id}"
                )

            activity_events[activity_uuid].append(
                (timestamp, event)
            )

        if transition in {
            "task/instantiation",
            "subprocess/receiving",
        }:
            sub_refs = [
                reference
                for reference in omap
                if reference in subprocess_object_ids
            ]
            if (
                len(sub_refs) != 2
                or "sub:root" not in vmap
            ):
                raise ValueError(
                    f"Subprocess interface event "
                    f"{event_id} lacks parent/child "
                    "relations"
                )

        if "concept:endpoint" in vmap:
            if not isinstance(
                vmap["concept:endpoint"],
                str,
            ):
                raise ValueError(
                    "concept:endpoint must be a string"
                )
            endpoint_count += 1

        if "sub:root" in vmap:
            sub_root_count += 1

        if "data" in vmap:
            data_id = vmap["data"]
            if (
                data_id not in objects
                or objects[data_id].get("ocel:type")
                != "data"
            ):
                raise ValueError(
                    f"Invalid data foreign key "
                    f"on {event_id}"
                )
            data_references.append(data_id)
            data_reference_count += 1

        transition_counts[transition] += 1
        omap_sizes[len(omap)] += 1

    if transition_counts != Counter(
        EXPECTED_TRANSITIONS
    ):
        raise ValueError(
            "Lifecycle transition counts differ "
            f"from the exact source: {transition_counts}"
        )

    if omap_sizes != Counter(
        {
            1: 4,
            2: 5680,
            3: 3584,
        }
    ):
        raise ValueError(
            "Incorrect event relation cardinalities: "
            f"{omap_sizes}"
        )

    if (
        endpoint_count != 7128
        or data_reference_count != 3134
        or sub_root_count != 420
    ):
        raise ValueError(
            "Event-attribute presence counts differ "
            "from the exact source"
        )

    if len(data_references) != len(
        set(data_references)
    ):
        raise ValueError(
            "A referenced immutable data snapshot "
            "is reused by multiple events"
        )

    if wait_events != 4 or len(concept_owner) != 217:
        raise ValueError(
            "Incorrect source exception count "
            "or subprocess-instance count"
        )

    for activity_uuid, rows in activity_events.items():
        base = objects[activity_uuid][
            "ocel:type"
        ].removeprefix("abstraction:lifecycle:")

        if base == "external":
            continue

        transitions = [
            event["ocel:vmap"][
                "cpee:lifecycle:transition"
            ]
            for _, event in rows
        ]

        if transitions.count("activity/done") != 1:
            raise ValueError(
                f"Activity {activity_uuid} must have "
                "exactly one terminal done event"
            )

        calling_times = [
            time
            for time, event in rows
            if event["ocel:vmap"][
                "cpee:lifecycle:transition"
            ]
            == "activity/calling"
        ]
        done_time = next(
            time
            for time, event in rows
            if event["ocel:vmap"][
                "cpee:lifecycle:transition"
            ]
            == "activity/done"
        )

        if (
            calling_times
            and not calling_times[0] < done_time
        ):
            raise ValueError(
                f"Calling does not precede done "
                f"for activity {activity_uuid}"
            )

        if base in {
            "Check State",
            "Read Var",
            "Fetch Program",
            "Detection Machining",
            "Get Magazine X",
            "Get Magazine Z",
        }:
            expected = {
                "activity/calling",
                "activity/receiving",
                "dataelements/change",
                "activity/done",
            }
            if not expected <= set(transitions):
                raise ValueError(
                    f"Read/poll activity {activity_uuid} "
                    "lacks its required lifecycle bundle"
                )

    type_counts = Counter(
        obj["ocel:type"]
        for obj in objects.values()
    )

    if (
        type_counts["data"] != 3605
        or type_counts["stream"] != 1370
    ):
        raise ValueError(
            "Incorrect data or SensorStream object count"
        )

    if sum(
        type_counts[type_name]
        for type_name in SUBPROCESS_TYPES
    ) != 217:
        raise ValueError(
            "Incorrect subprocess object count"
        )

    if sum(
        count
        for type_name, count in type_counts.items()
        if type_name.startswith(
            "abstraction:lifecycle:"
        )
    ) != 1795:
        raise ValueError(
            "Incorrect activity-instance object count"
        )

    for component in (
        "IRB2600",
        "MaxxTurn45",
        "keyence",
        "NA",
    ):
        if (
            type_counts[
                f"view:component:{component}"
            ]
            != 1
        ):
            raise ValueError(
                f"Expected one singleton {component} "
                "component view"
            )

    data_objects = {
        object_id: obj
        for object_id, obj in objects.items()
        if obj["ocel:type"] == "data"
    }

    forbidden_names = {
        "quality",
        "good",
        "bad",
        "chips",
    }

    for object_id, obj in data_objects.items():
        if (
            set(obj)
            != {
                "ocel:type",
                "ocel:ovmap",
            }
            or "list" not in obj["ocel:ovmap"]
        ):
            raise ValueError(
                f"Malformed data snapshot {object_id}"
            )

        values = obj["ocel:ovmap"]["list"]
        if (
            values is not None
            and not isinstance(values, list)
        ):
            raise ValueError(
                "Data snapshot list must be an array "
                f"or null: {object_id}"
            )

        if isinstance(values, list):
            for pair in values:
                if (
                    not isinstance(pair, dict)
                    or set(pair)
                    != {
                        "name",
                        "value",
                    }
                ):
                    raise ValueError(
                        f"Malformed name/value pair "
                        f"in {object_id}"
                    )

                if (
                    str(pair["name"]).lower()
                    in forbidden_names
                ):
                    raise ValueError(
                        "An unsupported quality/chips "
                        "ground-truth label was added"
                    )

    selection_rows = []
    top_route_calling: Dict[
        str, Counter[str]
    ] = defaultdict(Counter)

    for event in events.values():
        base = _event_base(event)
        transition = event["ocel:vmap"][
            "cpee:lifecycle:transition"
        ]
        owner = event["ocel:omap"][0]
        owner_type = objects[owner]["ocel:type"]

        if (
            base
            == "Next QR, Order Item and Tray"
            and transition
            == "dataelements/change"
        ):
            snapshot = data_objects[
                event["ocel:vmap"]["data"]
            ]["ocel:ovmap"]["list"]
            selection_rows.append(
                {
                    pair["name"]: pair["value"]
                    for pair in snapshot
                }
            )

        if (
            owner_type
            in {
                "abstraction:sub:Turm Single wr04",
                "abstraction:sub:Turm Single wr08",
            }
            and transition == "activity/calling"
        ):
            top_route_calling[owner][base] += 1

    selection_rows.sort(
        key=lambda row: row["count"]
    )

    if len(selection_rows) != 8:
        raise ValueError(
            "The batch must select exactly eight items"
        )

    for index, row in enumerate(selection_rows):
        order_id, item, feed, depth = PARTS[index]

        if (
            row.get("count") != index
            or row.get("order_id") != order_id
            or row.get("order_item") != item
        ):
            raise ValueError(
                "Batch count/order selection invariant "
                "is violated"
            )

        if row.get("initial") is not (index == 0):
            raise ValueError(
                "Only count=0 may be the initial part"
            )

        if row.get(
            "current_tray"
        ) != _tray_position(index):
            raise ValueError(
                "Tray coordinates do not agree with "
                "the row-major coordinate rule"
            )

        if row.get("program_params") != [
            feed,
            depth,
            f"{order_id}: {item}",
        ]:
            raise ValueError(
                "Program parameters do not agree "
                "with the selected item"
            )

        expected_suffix = [
            [a, b]
            for a, b, _, _ in PARTS[index + 1 :]
        ]
        if row.get("order_list") != expected_suffix:
            raise ValueError(
                "order_list is not the unconsumed "
                "batch suffix"
            )

    expected_common = {
        "Create Program",
        "Turn",
        "Set Program MT45",
        "MT45 Start",
        "MT45 Take Out",
    }

    top_objects = [
        (object_id, obj["ocel:type"])
        for object_id, obj in objects.items()
        if obj["ocel:type"]
        in {
            "abstraction:sub:Turm Single wr04",
            "abstraction:sub:Turm Single wr08",
        }
    ]

    if Counter(
        type_name
        for _, type_name in top_objects
    ) != Counter(
        {
            "abstraction:sub:Turm Single wr04": 4,
            "abstraction:sub:Turm Single wr08": 4,
        }
    ):
        raise ValueError(
            "The top-level batch must contain "
            "four wr04 and four wr08 routes"
        )

    for object_id, type_name in top_objects:
        calls = top_route_calling[object_id]

        if any(
            calls[name] != 1
            for name in expected_common
        ):
            raise ValueError(
                f"Incomplete common top-level route "
                f"on {object_id}"
            )

        if type_name.endswith("wr04"):
            required = {
                "Robot Measuring and Traying",
                "Measure with Keyence",
            }
            forbidden = {"Robot Traying"}
        else:
            required = {"Robot Traying"}
            forbidden = {
                "Robot Measuring and Traying",
                "Measure with Keyence",
                "Signal Keyence End",
            }

        if any(
            calls[name] != 1
            for name in required
        ) or any(
            calls[name]
            for name in forbidden
        ):
            raise ValueError(
                f"Wrong product-specific route "
                f"on {object_id}"
            )

    interface_children = set()
    for event in events.values():
        if (
            event["ocel:vmap"][
                "cpee:lifecycle:transition"
            ]
            == "task/instantiation"
        ):
            sub_refs = [
                reference
                for reference in event["ocel:omap"]
                if reference
                in subprocess_object_ids
            ]
            interface_children.add(sub_refs[1])

    root_candidates = (
        subprocess_object_ids - interface_children
    )
    if (
        len(interface_children) != 216
        or len(root_candidates) != 1
    ):
        raise ValueError(
            "Parent/child interface events do not "
            "encode one 217-node process tree"
        )

    component_event_counts = Counter()
    for event in events.values():
        for reference in event["ocel:omap"]:
            obj_type = objects[reference][
                "ocel:type"
            ]
            if obj_type.startswith(
                "view:component:"
            ):
                component_event_counts[
                    obj_type.removeprefix(
                        "view:component:"
                    )
                ] += 1

    if component_event_counts != Counter(
        {
            "IRB2600": 2360,
            "MaxxTurn45": 749,
            "keyence": 55,
        }
    ):
        raise ValueError(
            "Incorrect component E2O cardinalities: "
            f"{component_event_counts}"
        )

    stream_counts: Counter[str] = Counter()
    point_total = 0
    machine_groups_checked = 0

    for object_id in stream_ids:
        obj = objects[object_id]
        records = obj.get(
            "ocel:ovmap", {}
        ).get("list")

        if (
            not isinstance(records, list)
            or len(records) < 2
        ):
            raise ValueError(
                f"Malformed SensorStream object "
                f"{object_id}"
            )

        if (
            set(records[0]) != {"stream:name"}
            or set(records[1])
            != {"stream:source"}
        ):
            raise ValueError(
                f"Malformed SensorStream headers "
                f"in {object_id}"
            )

        name = records[0]["stream:name"]
        stream_counts[name] += 1
        points = records[2:]
        point_total += len(points)

        if name == "IRB2600":
            if len(points) != 38:
                raise ValueError(
                    "Every IRB2600 bundle must contain "
                    "exactly 38 channels"
                )

            values = {
                record["stream:point"]["stream:id"]:
                record["stream:point"]["stream:value"]
                for record in points
            }

            if set(values) != set(ROBOT_CHANNELS):
                raise ValueError(
                    "IRB2600 channel set is incomplete"
                )

            expected_transition = (
                "FinishedToActive"
                if values["StateMachine"] == "Active"
                else "ActiveToFinished"
            )

            if (
                values["LastTransition"]
                != expected_transition
            ):
                raise ValueError(
                    "Robot LastTransition disagrees "
                    "with StateMachine"
                )

            if (
                values["State"] != "MotorsOn"
                or values["OperationMode"] != "Auto"
                or values["Connected"] is not True
                or values["BackupInProgress"] is not False
            ):
                raise ValueError(
                    "Robot hard state domains are violated"
                )

        elif name == "keyence":
            for record in points:
                point = record.get(
                    "stream:point", {}
                )
                value = point.get("stream:value")

                if (
                    point.get("stream:id")
                    != "measurement"
                    or not isinstance(
                        value,
                        (int, float),
                    )
                ):
                    raise ValueError(
                        "Invalid Keyence measurement point"
                    )

                if (
                    not 0.36 <= value <= 999.99
                    or round(value, 2) != value
                ):
                    raise ValueError(
                        "Keyence measurement violates "
                        "range or 0.01 mm resolution"
                    )

        elif name == "MaxxTurn45":
            values: Dict[str, Any] = {}

            for record in points:
                point = record.get(
                    "stream:point", {}
                )

                if (
                    point.get(
                        "stream:meta", {}
                    ).get("StatusCode")
                    != "Good"
                ):
                    raise ValueError(
                        "Machine OPC UA StatusCode "
                        "must be Good"
                    )

                values[
                    point.get("stream:id")
                ] = point.get("stream:value")

            for prefix in ("", "MaxxTurn45/"):
                for family in (
                    "Power/Active",
                    "Axes/Power/Active",
                    "Coolant/Power/Active",
                ):
                    channels = [
                        prefix
                        + family
                        + "/"
                        + suffix
                        for suffix in (
                            "A",
                            "B",
                            "C",
                            "Sum",
                        )
                    ]

                    if all(
                        channel in values
                        for channel in channels
                    ):
                        a, b, c, total = (
                            float(values[channel])
                            for channel in channels
                        )

                        if abs(
                            total - (a + b + c)
                        ) > 6.1:
                            raise ValueError(
                                "Three-phase power sum "
                                "is inconsistent"
                            )

                        machine_groups_checked += 1

        else:
            raise ValueError(
                f"Unknown SensorStream name {name}"
            )

    if stream_counts != Counter(
        {
            "MaxxTurn45": 733,
            "IRB2600": 590,
            "keyence": 47,
        }
    ):
        raise ValueError(
            "Incorrect SensorStream bundle counts: "
            f"{stream_counts}"
        )

    if (
        point_total != 277500
        or machine_groups_checked == 0
    ):
        raise ValueError(
            "Incorrect sensor-point count or no "
            "complete machine phase group"
        )


def save_ocel(ocel: dict, output_path: str) -> None:
    validate_ocel(ocel)

    path = Path(output_path)
    if path.suffix != ".jsonocel":
        raise ValueError(
            "output_path must end with .jsonocel"
        )

    with path.open("w", encoding="utf-8") as file:
        json.dump(
            ocel,
            file,
            ensure_ascii=False,
            separators=(",", ":"),
        )


if __name__ == "__main__":
    seed = 42
    for i in range(2,3):
        ocel = generate_ocel(
            seed=seed+i,
            number_of_parts=8,
        )
        save_ocel(
            ocel,
            f"log_{i}.jsonocel",
        )