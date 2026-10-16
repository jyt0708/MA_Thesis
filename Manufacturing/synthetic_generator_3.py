"""Executable Nine Rooks simulator, using only the Python standard library.

This is the source-schema-fidelity variant: OCEL 0.1, original spellings,
unlinked streams, retained snapshots, and four orphan Wait While Moving Up
completions. The fixed source batch accepts exactly eight parts. Observed
aggregate message/point counts are not invented padding targets: polling and
acquisition counts follow the sampled operation durations. The specification
does not give every original endpoint, program body, model ID, or sensor
metadata value; those values are synthesized without changing their fields.

The timing and at data attributes record simulated clock time as
YYYY-MM-DD HH:MM:SS +0100, with whole-second precision.
In Get Current Time snapshots, timing is the start of the preceding
IRB2600 Door to Measure activity; at is the observation time after that
activity completes. Their separation follows the simulated operation,
including its setup, movement, polling, and handoff. Configuration snapshots
retain only at (the batch epoch), since no positioning has occurred yet.
These are explicit simulator semantics for the two data fields.

No source log, network service, machine, or third-party package is required.

"""

import bisect
import copy
import hashlib
import json
import math
import random
import re
import uuid
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path


TZ = timezone(timedelta(hours=1))
EPOCH = datetime(2026, 1, 1, 8, tzinfo=TZ)
TIMING_FORMAT = "%Y-%m-%d %H:%M:%S %z"

SUB = "abstraction:sub:"
ACT = "abstraction:lifecycle:"
COMP = "view:component:"
BATCH = "- Turm Batch Processing V2"
GHOST = "Wait While Moving Up"

TRANSITIONS = {
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

ATTRIBUTES = [
    "concept:instance",
    "id:id",
    "cpee:activity_uuid",
    "lifecycle:transition",
    "cpee:lifecycle:transition",
    "concept:endpoint",
    "data",
    "sub:root",
    "cpee:uuid",
]

ORDER = {
    "order1": ["wr04", "wr04", "wr08", "wr08", "wr08"],
    "order2": ["wr04", "wr04", "wr08"],
}
ITEMS = [
    (order, item)
    for order, items in ORDER.items()
    for item in items
]
PARAMS = [
    (0.03, 0.6),
    (0.03, 0.6),
    (0.05, 0.2),
    (0.05, 0.6),
    (0.05, 0.6),
    (0.03, 0.4),
    (0.07, 0.4),
    (0.05, 0.6),
]
PATTERNS = ["yyyy", "xxxx", "aaaa"]

PROGRAM_URL = (
    "https://synthetic.example/MT45/raw/master/TURM_PSC.MPF"
)
HTTP = "https://synthetic.example/services/"
OPC = "opc.tcp://synthetic.example:4840"

TOOLS = dict(
    zip(
        [
            "SCHRUPPER-A-L",
            "SCHLICHTER-A-L",
            "FRAESER-10-AXHS",
            "AX_UK_IN_SCHLI",
            "EINSTI_2N-LR-HS",
            "EINSTICH-ABSTICH3",
            "SCHLICHTER-A-R",
            "SCHRUPPER-A-R",
        ],
        [7, 37, 54, 74, 8, 107, 38, 53],
    )
)

MOVES = {
    "IRB2600 Portal to Door": "IRB2600 Portal to Door",
    "IRB2600 Door to GS": "IRB2600 Door to GS",
    "IRB2600 GS to Take": "IRB2600 GS to Take",
    "IRB2600 Take to GS": "IRB2600 Take to GS",
    "IRB2600 GS to Door": "IRB2600 IRB2600 GS to Door",
    "IRB2600 Door to Portal": "IRB2600 Door to Portal",
    "IRB2600 Door to Measure": "X IRB2600 Execution",
    "IRB2600 Measure to Door": "X IRB2600 Execution",
    "IRB2600 Unload to Tray Down": "X IRB2600 Execution",
}

SERVICE = set(
    (
        "Assign Program|Logon|Set Program|Set Var|Signal Keyence End|"
        "Signal Unloading End|Taken Out|Set Dummy Program|"
        "Select Dummy Program|Start|Move Down|Move Up|wait|"
        "Wait for MT45 Takeout|Wait For Unload End"
    ).split("|")
)
READ = set(
    (
        "Check State|Read Var|Fetch Program|Detection Machining|"
        "Get Magazine X|Get Magazine Z"
    ).split("|")
)
UPDATE = set(
    (
        "Get Order Details|Inititalize Batch|"
        "Next QR, Order Item and Tray|Assemble Porgram|"
        "Replace Patterns with Params|Get Current Time|"
        "Set Paths|Not First Part"
    ).split("|")
)
ENDPOINT = {"Set Endpoint", "Correct  Queue"}

CHILD_TYPE = {
    "Get Machine State Cancelled": "X MT45 Control Getter",
    "Create Program": "X Generate NC",
    "Turn": "Turm Turn Production",
    "Set Program MT45": "X MT45 SP",
    "MT45 Start": "X MT45 Start",
    "MT45 Take Out": "Turm MT45 Take Out",
    "MT45 Check Operational Safety": "MT45 Check Operational Safety",
    "IRB2600 Grip": "Turm IRB2600 Grip",
    "IRB2600 Extract": "Turm IRB2600 Extract",
    "Robot Measuring and Traying": "Turm IRB2600 Measure and Tray",
    "Robot Traying": "Turm IRB2600 Tray",
    "Measure with Keyence": "Turm Keyence Measurement",
    "Machining": "Machining V2",
    **{
        name: "X MT45 Control Setter"
        for name in [
            "Aux On",
            "Mode Auto",
            "Door Close",
            "NC Start",
            "Open Clamp 2",
            "Reset Clamp 2",
        ]
    },
    **MOVES,
}

BASES = sorted(
    SERVICE
    | READ
    | UPDATE
    | ENDPOINT
    | set(CHILD_TYPE)
    | {
        "external",
        "Spawn Production",
        "Fetch",
        "Set Fetch Endpoint",
        GHOST,
    }
)
PROCESS_TYPES = sorted(
    set(CHILD_TYPE.values())
    | {BATCH, "Turm Single wr04", "Turm Single wr08"}
)
OBJECT_TYPES = (
    [SUB + name for name in PROCESS_TYPES]
    + [ACT + name for name in BASES]
    + ["data", "stream"]
    + [
        COMP + name
        for name in ["IRB2600", "MaxxTurn45", "keyence", "NA"]
    ]
)
MODEL_IDS = {
    base: "a" + str(i + 1)
    for i, base in enumerate(BASES)
}

ROBOT_FIXED = {
    "State": "MotorsOn",
    "OperationMode": "Auto",
    "BackupInProgress": False,
    "Connected": True,
    "Task.Program.Module": "OperationModule",
    "Task.Program.Routine": "Operation",
    "Task.Program.Range": "All",
    "Task.Cycle": "Once",
    "Task.Enabled": True,
    "Task.Motion": True,
    "Task.RemainingCycles": 1,
    "Task.TaskType": "Normal",
    "Task.Type": "Task",
    "Task.Name": "T_ROB1",
    "Task.Motion.Module": "OperationModule",
    "Task.Motion.Routine": "Operation",
    "Rapid.Cycle": "Once",
    "Rapid.MaximumIdentifierLength": 32,
    "Rapid.RemainingCycles": 1,
    "Rapid.IsMaster": True,
    "Rapid.Name": "RAPID",
    "Location.Column": 0,
    "Location.Row": 0,
}
ROBOT_CHANNELS = set(ROBOT_FIXED) | {
    "StateMachine",
    "LastTransition",
    "LastTransitionAt",
    "Rapid.ExecutionStatus",
    "Task.ExecutionType",
    "Task.ExecutionStatus",
    "Pos.X",
    "Pos.Y",
    "Pos.Z",
    *["RobJoint.Rax_" + str(i) for i in range(1, 7)],
}

JOINT_LIMITS = [
    (-102.02, 153.09),
    (-32.24, 73.68),
    (-48.20, 56.73),
    (-91.83, 2.31),
    (7.31, 89.45),
    (20.04, 126.64),
]
POS_LIMITS = [
    (-699.30, 1399.97),
    (-689.29, 923.30),
    (128.09, 1014.49),
]
POWER_GROUPS = [
    "Power/Active",
    "Axes/Power/Active",
    "Coolant/Power/Active",
]
LATHE_CHANNELS = (
    [
        f"{group}/{phase}"
        for group in POWER_GROUPS
        for phase in ["A", "B", "C", "Sum"]
    ]
    + [
        "Spindle/" + name
        for name in [
            "actSpeed",
            "cmdSpeed",
            "driveLoad",
            "aaPower",
            "turnState",
        ]
    ]
    + [
        f"Axes/{axis}/{name}"
        for axis in "XYZ"
        for name in [
            "aaLeadP",
            "aaVactB",
            "actFeedRate",
            "aaLoad",
            "aaTorque",
            "aaPower",
        ]
    ]
    + [
        "Program/" + name
        for name in [
            "actBlock",
            "actLineNumber",
            "msg",
            "progName1",
        ]
    ]
    + [
        "State/" + name
        for name in [
            "actToolIdent",
            "actTNumber",
            "actToolLength1",
            "actToolRadius",
            "progStatus",
            "resetActive",
            "progStarts",
        ]
    ]
)


def _timing_stamp(seconds):
    """Format timing and at attributes using the simulated UTC+01 clock."""
    return (EPOCH + timedelta(seconds=seconds)).strftime(TIMING_FORMAT)


def _stamp(seconds, local=False):
    dt = EPOCH + timedelta(seconds=seconds)
    if local:
        dt = dt.replace(tzinfo=None)
    return dt.isoformat(timespec="microseconds")


def _time(value, event=False):
    if not isinstance(value, str):
        raise ValueError("Timestamp must be a string")
    try:
        dt = datetime.fromisoformat(value)
    except ValueError as exc:
        raise ValueError("Invalid ISO timestamp: " + value) from exc
    if event and (
        not value.endswith("+01:00")
        or dt.utcoffset() != timedelta(hours=1)
    ):
        raise ValueError(
            "Event timestamps must have the original +01:00 offset"
        )
    # Unzoned source clocks in this simulator use the same UTC+01 clock.
    # Their interpretation never depends on the host's local timezone.
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=TZ)
    return (dt - EPOCH).total_seconds()


def _tray(i):
    row, col = divmod(i, 6)
    return [
        round(-137.87 + 50 * row + 0.6 * col, 2),
        -178.0 + 80 * col,
        199.0,
    ]


def _part_state(i):
    order, item = ITEMS[i]
    feed, depth = PARAMS[i]
    return {
        "order_id": order,
        "order_item": item,
        "count": i,
        "initial": i == 0,
        "current_tray": _tray(i),
        "order_list": [list(x) for x in ITEMS[i + 1:]],
        "program_params": [
            feed,
            depth,
            f"{order}: {item}",
        ],
        "program_patterns": PATTERNS[:],
        "program_url": PROGRAM_URL,
        "qr": "*268MFA466*TZHZE 000",
    }


def _nc(i):
    order, item = ITEMS[i]
    feed, depth = PARAMS[i]
    message = (
        f"Type: {order}: {item}; "
        f"Schnitttiefe: {depth}; Vorschub: {feed}"
    )
    return (
        f"%_N_TURM_PSC1_MPF\n; {message}\n"
        "N10 G18 G90 G95\n"
        f'N20 MSG("{message}")\n'
        "N30 T7 D1\n"
        "N40 G97 S1800 M3\n"
        "N50 G0 X32 Z2\n"
        f"N60 G1 Z-30 F{feed}\n"
        f"N70 X{32 - 2 * depth:.2f}\n"
        "N80 G0 X100 Z100\n"
        "N90 M5\n"
        "N100 M30\n"
    )


def _rapid(target, gripping):
    x, y, z = target
    return (
        "MODULE OperationModule\n"
        "  CONST speeddata vMove:=[100,50,50,50];\n"
        "  PERS tooldata toolGrip:=[TRUE,"
        "[[0,0,100],[1,0,0,0]],"
        "[1,[0,0,50],[1,0,0,0],0,0,0]];\n"
        f"  CONST robtarget destination:=[[{x},{y},{z}],"
        "[1,0,0,0],[0,0,0,0],"
        "[9E9,9E9,9E9,9E9,9E9,9E9]];\n"
        "  PROC Operation()\n"
        "    MoveL destination,vMove,fine,toolGrip;\n"
        f"    SetDO doGripper,{1 if gripping else 0};\n"
        "  ENDPROC\n"
        "ENDMODULE\n"
    )


@dataclass
class _Process:
    id: str
    name: str
    instance: int
    start: float
    end: float
    part: int = -1
    activities: list = field(default_factory=list)


@dataclass
class _Activity:
    id: str
    owner: _Process
    base: str
    start: float
    end: float
    kind: str
    data: dict = field(default_factory=dict)
    child: object = None
    component: str = ""
    receives: list = field(default_factory=list)


class _Simulator:
    def __init__(self, seed):
        self.rng = random.Random(seed)
        self.namespace = uuid.uuid5(
            uuid.NAMESPACE_URL,
            "nine-rooks-synthetic/" + str(seed),
        )
        self.serial = 0
        digest = hashlib.sha256(str(seed).encode()).hexdigest()
        self.instance_base = 1000000 + int(digest[:10], 16) * 1000

        self.objects = {}
        self.processes = []
        self.activities = []
        self.motion = []
        self.position = [600.0, 200.0, 700.0]
        self.robot_free = 0.0
        self.streams = []
        self.records = []

        self.new_object("external", ACT + "external", {})
        self.components = {}
        for name in ["IRB2600", "MaxxTurn45", "keyence", "NA"]:
            self.components[name] = self.uid()
            self.new_object(
                self.components[name],
                COMP + name,
                {},
            )

    def uid(self, hex_only=False):
        self.serial += 1
        value = uuid.uuid5(self.namespace, str(self.serial))
        return value.hex if hex_only else str(value)

    def new_object(self, oid, typ, ovmap):
        if oid in self.objects:
            raise ValueError("Duplicate planned object ID")
        self.objects[oid] = {
            "ocel:type": typ,
            "ocel:ovmap": ovmap,
        }

    def snapshot(self, values):
        oid = self.uid()
        entries = (
            None
            if values is None
            else [
                {"name": key, "value": copy.deepcopy(value)}
                for key, value in values.items()
            ]
        )
        self.new_object(oid, "data", {"list": entries})
        return oid

    def process(self, name, start, part=-1):
        p = _Process(
            self.uid(),
            name,
            self.instance_base + len(self.processes),
            start,
            start + 1,
            part,
        )
        self.new_object(p.id, SUB + name, {})
        self.processes.append(p)
        return p

    def activity(
        self,
        p,
        base,
        start,
        end,
        kind=None,
        data=None,
        component="",
    ):
        if end - start < 0.02:
            raise ValueError(
                "Activity duration too short for its lifecycle"
            )
        if kind is None:
            if base in READ:
                kind = "read"
            elif base in UPDATE:
                kind = "update"
            elif base in ENDPOINT:
                kind = "endpoint"
            elif base == "Set Fetch Endpoint":
                kind = "fetch_endpoint"
            else:
                kind = "service"

        a = _Activity(
            self.uid(True),
            p,
            base,
            start,
            end,
            kind,
            copy.deepcopy(data or {}),
            component=component,
        )
        if kind != "ghost":
            self.new_object(a.id, ACT + base, {})
        p.activities.append(a)
        self.activities.append(a)
        return a

    def finish(self, p):
        p.end = max(
            (a.end for a in p.activities),
            default=p.start + 0.1,
        ) + 0.05
        return p.end

    def call(
        self,
        p,
        base,
        start,
        build,
        child_name=None,
        asynchronous=False,
    ):
        child = self.process(
            child_name or CHILD_TYPE[base],
            start + 0.06,
            p.part,
        )
        build(child)
        self.finish(child)
        a = self.activity(
            p,
            base,
            start,
            start + 0.09 if asynchronous else child.end + 0.06,
            "spawn" if asynchronous else "nested",
        )
        a.child = child
        return a

    def control(self, p, start, identifier, getter=False):
        base = {
            "endpoint_url": OPC,
            "namespace_index": "2",
            "identifier_type": "string",
            "identifier": identifier,
            "parameters": json.dumps([True]),
            "value": (
                True
                if identifier in {"Aux On", "Door Close"}
                else "true"
            ),
            "check_value": True if getter else "true",
            "state": "true",
        }
        t = start + 0.1
        if not getter:
            self.activity(
                p, "Set Var", t, t + 0.25, data=base
            )
            t += 0.3

        duration = (
            self.rng.uniform(0.7, 1.3)
            if getter
            else self.rng.uniform(1.7, 2.5)
        )
        count = max(1, int(math.ceil(duration / 0.75)))
        for j in range(count):
            values = dict(
                base,
                check_value=(
                    (j == count - 1)
                    if getter
                    else str(j == count - 1).lower()
                ),
            )
            self.activity(
                p,
                "Read Var",
                t,
                t + duration / count - 0.02,
                data=values,
            )
            t += duration / count

    def robot(self, p, target, gripping=False):
        if p.start < self.robot_free:
            raise ValueError("Robot reservation conflict")

        duration = (
            self.rng.uniform(13, 17)
            if p.name != "X IRB2600 Execution"
            else self.rng.uniform(15, 20)
        )
        program = _rapid(target, gripping)
        values = {
            "program": program,
            "content": program,
            "vars": "PERS bool ready:=TRUE;\n",
            "includes": ["OperationModule"],
            "move": "MoveL",
            "x": target[0],
            "y": target[1],
            "z": target[2],
            "gripper": "close" if gripping else "open",
            "check_var1": "StateMachine",
            "check_var2": "Task.ExecutionStatus",
        }

        t = p.start + 0.1
        for _ in range(2):
            self.activity(
                p,
                "Set Fetch Endpoint",
                t,
                t + 0.1,
                data={"fetch": HTTP + "robot/program"},
            )
            self.activity(
                p,
                "Fetch Program",
                t + 0.12,
                t + 0.25,
                data=values,
            )
            t += 0.3

        self.activity(
            p,
            "Assemble Porgram",
            t,
            t + 0.1,
            data=values,
        )
        t += 0.12
        for base in ["Logon", "Set Program", "Assign Program"]:
            self.activity(
                p, base, t, t + 0.1, data=values
            )
            t += 0.12

        motion_start = p.start + 1.6
        motion_end = p.start + duration - 0.5
        self.motion.append(
            (
                motion_start,
                motion_end,
                self.position[:],
                target[:],
            )
        )
        self.position = target[:]
        self.activity(
            p,
            "Start",
            motion_start - 0.06,
            motion_end + 0.2,
            data={"started": True},
        )

        poll_count = max(
            3,
            math.ceil(
                duration / self.rng.uniform(1.6, 1.9)
            ),
        )
        checks = [p.start + 1.25]
        checks += [
            motion_start
            + (motion_end - motion_start) * j / (poll_count - 1)
            for j in range(1, poll_count - 1)
        ]
        checks.append(motion_end + 0.04)

        for t in checks:
            self.activity(
                p,
                "Check State",
                t,
                t + 0.06,
                data=values,
                component="IRB2600",
            )
        self.finish(p)
        self.robot_free = p.end

    def movement(self, p, base, t, target, gripping=False):
        return self.call(
            p,
            base,
            t,
            lambda child: self.robot(
                child, target, gripping
            ),
        )

    def simple_motion(self, p, base, t, target):
        duration = self.rng.uniform(6.5, 8)
        if t < self.robot_free:
            raise ValueError("Robot reservation conflict")
        a = self.activity(
            p,
            base,
            t,
            t + duration,
            data={
                "x": target[0],
                "y": target[1],
                "z": target[2],
            },
        )
        self.motion.append(
            (
                t + 0.05,
                a.end - 0.05,
                self.position[:],
                target[:],
            )
        )
        self.position = target[:]
        self.robot_free = a.end
        return a

    def machining(self, p, begin, finish, component):
        self.activity(
            p,
            "Correct  Queue",
            p.start + 0.1,
            p.start + 0.2,
            data={
                "queue": [],
                "finished": False,
                "fetch": HTTP + component + "/fetch",
            },
        )
        a = self.activity(
            p,
            "Fetch",
            begin,
            finish,
            "sensor",
            component=component,
        )
        self.streams.append(a)

    def sensed_process(self, p, begin, finish, component):
        self.activity(
            p,
            "Detection Machining",
            p.start + 0.1,
            p.start + 0.3,
            data={
                "machine": component,
                "trigger": "start",
                "state": "true",
            },
        )
        self.call(
            p,
            "Machining",
            begin - 0.4,
            lambda child: self.machining(
                child, begin, finish, component
            ),
        )

    def nc_process(self, p):
        t, i = p.start + 0.1, p.part
        self.activity(
            p,
            "Set Endpoint",
            t,
            t + 0.1,
            data={"fetch": PROGRAM_URL},
        )
        self.activity(
            p,
            "Fetch Program",
            t + 0.12,
            t + 0.4,
            data={
                "content": _nc(i),
                "program_url": PROGRAM_URL,
            },
        )
        self.activity(
            p,
            "Replace Patterns with Params",
            t + 0.42,
            t + 1.2,
            data={
                **_part_state(i),
                "program": _nc(i),
                "patterns": PATTERNS[:],
                "params": _part_state(i)["program_params"],
            },
        )

    def set_program(self, p):
        t = p.start + 0.1
        paths = {
            "program_path": "/nc/TURM_PSC1.MPF",
            "program_value": _nc(p.part),
            "assignment_path": "/nc/active",
            "assignment_path_stub": "/nc/",
            "setting_path": "/nc/TURM_PSC1.MPF",
            "setting_path_stub": "/nc/",
            "dummy_assignment_path": "/nc/dummy-active",
            "dummy_setting_path": "/nc/ASUP2.SYF",
            "assign_program": "_N_TURM_PSC1_MPF",
            "dir": "/nc",
            "name": "TURM_PSC1.MPF",
            "mode": "Auto",
        }
        self.activity(
            p, "Set Paths", t, t + 0.1, data=paths
        )
        t += 0.12
        for base in [
            "Set Dummy Program",
            "Select Dummy Program",
            "Set Program",
            "Select Dummy Program",
        ]:
            self.activity(
                p, base, t, t + 0.3, data=paths
            )
            t += 0.32

    def start_lathe(self, p):
        t = p.start + 0.1
        for base in [
            "Aux On",
            "Mode Auto",
            "Door Close",
            "NC Start",
        ]:
            a = self.call(
                p,
                base,
                t,
                lambda c, b=base: self.control(
                    c, c.start, b
                ),
            )
            t = a.end + 0.03

    def safety(self, p):
        t = p.start + 0.1
        for base, key, value in [
            ("Get Magazine X", "magazine_x", 0.0),
            ("Get Magazine Z", "magazine_z", 100.0),
        ]:
            self.activity(
                p,
                base,
                t,
                t + 1.0,
                data={key: value, "state": "true"},
            )
            t += 1.05

    def grip(self, p):
        t = p.start + 0.1
        for base, target in [
            (
                "IRB2600 Portal to Door",
                [200.0, 0.0, 600.0],
            ),
            (
                "IRB2600 Door to GS",
                [100.0, -100.0, 400.0],
            ),
            (
                "IRB2600 GS to Take",
                [-200.0, 0.0, 300.0],
            ),
        ]:
            t = (
                self.movement(
                    p, base, t, target, True
                ).end
                + 0.03
            )

    def extract(self, p):
        t = p.start + 0.1
        for base, target in [
            (
                "IRB2600 Take to GS",
                [100.0, -100.0, 400.0],
            ),
            (
                "IRB2600 GS to Door",
                [200.0, 0.0, 600.0],
            ),
        ]:
            t = (
                self.movement(
                    p, base, t, target, True
                ).end
                + 0.03
            )

    def takeout(self, p):
        t = p.start + 0.1
        for base, build in [
            ("MT45 Check Operational Safety", self.safety),
            ("IRB2600 Grip", self.grip),
            (
                "Open Clamp 2",
                lambda c: self.control(
                    c, c.start, "Open Clamp 2"
                ),
            ),
            ("IRB2600 Extract", self.extract),
            (
                "Reset Clamp 2",
                lambda c: self.control(
                    c, c.start, "Reset Clamp 2"
                ),
            ),
        ]:
            t = self.call(p, base, t, build).end + 0.03
        self.activity(
            p,
            "Taken Out",
            t,
            t + 0.3,
            data={
                "message": "taken-out",
                "id": self.uid(),
            },
        )

    def tray(self, p):
        t = p.start + 0.1
        measurement = None

        if p.name == "Turm IRB2600 Measure and Tray":
            a = self.movement(
                p,
                "IRB2600 Door to Measure",
                t,
                [500.0, -350.0, 500.0],
                True,
            )
            t = a.end + 0.03
            self.activity(
                p,
                "Get Current Time",
                t,
                t + 0.1,
                data={
                    "at": _timing_stamp(t),
                    # Record the actual positioning activity start. No extra
                    # randomness or artificial timestamp offset is needed.
                    "timing": _timing_stamp(a.start),
                },
            )
            begin = t + 1.0
            down = self.simple_motion(
                p,
                "Move Down",
                begin + 0.7,
                [500.0, -350.0, 150.0],
            )
            wait = self.activity(
                p,
                "wait",
                down.end + 0.03,
                down.end + self.rng.uniform(4, 6),
            )
            up = self.simple_motion(
                p,
                "Move Up",
                wait.end + 0.03,
                [500.0, -350.0, 500.0],
            )
            self.activity(
                p,
                GHOST,
                up.end + 0.01,
                up.end + 0.04,
                "ghost",
            )
            measurement = (begin, up.end + 0.07)
            t = (
                self.movement(
                    p,
                    "IRB2600 Measure to Door",
                    up.end + 0.3,
                    [200.0, 0.0, 600.0],
                    True,
                ).end
                + 0.03
            )

        t = (
            self.movement(
                p,
                "IRB2600 Door to Portal",
                t,
                [600.0, 200.0, 700.0],
                True,
            ).end
            + 0.03
        )
        t = (
            self.movement(
                p,
                "IRB2600 Unload to Tray Down",
                t,
                _tray(p.part),
            ).end
            + 0.03
        )
        signal = self.activity(
            p,
            "Signal Unloading End",
            t,
            t + 0.2,
            data={
                "message": "unloading-end",
                "id": self.uid(),
            },
        )
        return measurement, signal.end

    def build(self):
        batch = self.process(BATCH, 0)
        self.activity(
            batch,
            "Get Order Details",
            0.1,
            0.3,
            data={"order": ORDER},
        )
        self.activity(
            batch,
            "Inititalize Batch",
            0.35,
            0.55,
            data=self.configuration(),
        )
        t, previous_unload = 0.7, 0.0

        for i, (_, item) in enumerate(ITEMS):
            self.activity(
                batch,
                "Next QR, Order Item and Tray",
                t,
                t + 0.15,
                data=_part_state(i),
            )
            getter = self.call(
                batch,
                "Get Machine State Cancelled",
                t + 0.2,
                lambda c: self.control(
                    c, c.start, "State/resetActive", True
                ),
            )
            start = getter.end + 0.05
            part = self.process(
                "Turm Single " + item,
                start + 0.06,
                i,
            )
            spawn = self.activity(
                batch,
                "Spawn Production",
                start,
                start + 0.09,
                "spawn",
                _part_state(i),
            )
            spawn.child = part
            s = part.start

            self.call(
                part,
                "Create Program",
                s + self.rng.uniform(2.2, 2.8),
                self.nc_process,
            )
            program = self.call(
                part,
                "Set Program MT45",
                s + self.rng.uniform(6.5, 7.2),
                self.set_program,
            )
            startup = self.call(
                part,
                "MT45 Start",
                max(s + 11.5, program.end + 0.05),
                self.start_lathe,
            )
            machining_begin = startup.end + 0.1
            machining_end = s + self.rng.uniform(145, 166)
            self.call(
                part,
                "Turn",
                s + self.rng.uniform(2.8, 3.2),
                lambda c, b=machining_begin, e=machining_end:
                    self.sensed_process(c, b, e, "MaxxTurn45"),
            )

            take_start = machining_end + 0.3
            if i:
                wstart = startup.end + 0.03
                wend = max(
                    wstart + 0.2,
                    previous_unload + 0.01,
                )
                self.activity(
                    part,
                    "Wait For Unload End",
                    wstart,
                    wend,
                )
                take_start = max(
                    take_start,
                    wend + 0.05,
                )

            take = self.call(
                part,
                "MT45 Take Out",
                max(
                    take_start,
                    self.robot_free + 0.1,
                ),
                self.takeout,
            )
            branch_name = (
                "Robot Measuring and Traying"
                if item == "wr04"
                else "Robot Traying"
            )
            result = []
            branch = self.call(
                part,
                branch_name,
                take.end + 0.1,
                lambda c: result.append(self.tray(c)),
            )
            window, previous_unload = result[0]

            if window:
                begin, end = window
                self.call(
                    part,
                    "Measure with Keyence",
                    begin - 0.9,
                    lambda c, b=begin, e=end:
                        self.sensed_process(c, b, e, "keyence"),
                )
                self.activity(
                    part,
                    "Signal Keyence End",
                    branch.end + 0.05,
                    branch.end + 0.3,
                    data={
                        "message": "keyence-end",
                        "id": self.uid(),
                    },
                )

            self.finish(part)
            wait = self.activity(
                batch,
                "Wait for MT45 Takeout",
                spawn.end + 0.03,
                take.end + 0.04,
            )
            t = wait.end + 0.1
            if i == 0:
                self.activity(
                    batch,
                    "Not First Part",
                    t,
                    t + 0.1,
                    data={"initial": False},
                )
                t += 0.12

        self.finish(batch)
        batch.end = max(
            p.end for p in self.processes
        ) + 0.1
        self.motion.sort()

    def configuration(self):
        return {
            "order": ORDER,
            "order_list": [list(x) for x in ITEMS],
            "initial": None,
            "seed": 1,
            "qr": "*268MFA466*TZHZE 001",
            "program_patterns": PATTERNS[:],
            "programs": {
                name: PROGRAM_URL
                for name in ["wr04", "wr08"]
            },
            "tray_positions": [
                [_tray(row * 6 + col) for col in range(6)]
                for row in range(8)
            ],
            "trays": [
                [row, col]
                for row in range(8)
                for col in range(6)
            ],
            "attributes": None,
            "endpoints": None,
            "customization": None,
            "info": BATCH,
            "modeltype": "CPEE",
            "theme": "manufacturing",
            "customer": "synthetic",
            "status": "ready",
            "artefacts": [],
            "explanation": "Lathe, robot and micrometer batch",
            "creator": "simulator",
            "author": "simulator",
            "design_stage": "execution",
            "design_dir": "/models",
            "sensors": ["MaxxTurn45", "IRB2600", "keyence"],
            "behavior": "fork_running",
            "url": HTTP + "batch",
            "init": True,
            "timeout": 600,
            "start_instance": self.instance_base,
            "start_url": HTTP + "start",
            "start_git": PROGRAM_URL,
            "command": "start",
            "ipc_receive": HTTP + "receive",
            "ipc_send": HTTP + "send",
            "fetch": HTTP + "fetch",
            "fetch_microvu": HTTP + "microvu",
            "measure": HTTP + "keyence",
            "decide": HTTP + "decide",
            "at": _timing_stamp(0),
            "move_lift": HTTP + "lift",
            "opc_execute": HTTP + "opc/execute",
            "opc_read": HTTP + "opc/read",
            "opc_write": HTTP + "opc/write",
            "ep_check": HTTP + "check",
            "urls": [PROGRAM_URL],
            "logon": HTTP + "logon",
            "set": HTTP + "set",
            "assign": HTTP + "assign",
            "reset": HTTP + "reset",
            "start": HTTP + "start",
            "regex": "^(true|false)$",
            "check": "StateMachine",
            "target": "Finished",
            "trigger_path": "State/progStatus",
            "trigger_start": 3,
            "trigger_pause": "4",
            "trigger_end": 4,
            "trigger_start_value": "3",
            "trigger_pause_value": 4,
            "trigger_end_value": "4",
            "event_batch": 334,
            "refinement": "source",
            "test": False,
            "ttl": 60,
            "delete": False,
            "label": "production",
            "magazine_x": HTTP + "magazine/x",
            "magazine_z": HTTP + "magazine/z",
            "x_measure": 500.0,
            "y_measure": -350.0,
            "z_measure": 500.0,
        }

    def robot_values(self, t):
        position = [600.0, 200.0, 700.0]
        active = False
        transition = 0.0

        for begin, end, source, target in self.motion:
            if t < begin:
                break
            if t <= end:
                u = (t - begin) / (end - begin)
                h = u * u * (3 - 2 * u)
                position = [
                    x + (y - x) * h
                    for x, y in zip(source, target)
                ]
                active, transition = True, begin
                break
            position, transition = target[:], end

        x, y, z = position
        factors = [
            (x + 200) / 800,
            (z - 150) / 550,
            (y + 350) / 700,
            (x + z + 50) / 1350,
            (z - y + 200) / 1250,
            (x - y + 600) / 1600,
        ]

        values = dict(ROBOT_FIXED)
        values.update(
            {
                "StateMachine": "Active" if active else "Finished",
                "LastTransition": (
                    "FinishedToActive"
                    if active
                    else "ActiveToFinished"
                ),
                "LastTransitionAt": _stamp(transition, True),
                "Rapid.ExecutionStatus": (
                    "Running" if active else "Stopped"
                ),
                "Task.ExecutionStatus": (
                    "Running" if active else "Stopped"
                ),
                "Task.ExecutionType": (
                    "Normal" if active else "None"
                ),
                "Pos.X": round(x, 6),
                "Pos.Y": round(y, 6),
                "Pos.Z": round(z, 6),
            }
        )

        for i, ((low, high), factor) in enumerate(
            zip(JOINT_LIMITS, factors), 1
        ):
            values[f"RobJoint.Rax_{i}"] = round(
                low
                + (high - low)
                * (0.2 + 0.6 * max(0, min(1, factor))),
                6,
            )
        return values

    def lathe_frames(self, a):
        count = max(
            3,
            math.ceil(
                (a.end - a.start - 0.04)
                / self.rng.uniform(0.40, 0.44)
            ) + 1,
        )
        i = a.owner.part
        order, item = ITEMS[i]
        feed, depth = PARAMS[i]
        message = (
            f"Type: {order}: {item}; "
            f"Schnitttiefe: {depth}; Vorschub: {feed}"
        )
        old_positions = None
        old_time = None
        phase = self.rng.uniform(0, math.pi)
        tool_names = list(TOOLS)

        for j in range(count):
            u = j / (count - 1)
            t = (
                a.start
                + 0.01
                + u * (a.end - a.start - 0.04)
            )
            running = j not in {0, count - 1}
            envelope = (
                min(1.0, u / 0.04, (1 - u) / 0.04)
                if running
                else 0.0
            )
            h = u * u * (3 - 2 * u)
            positions = [
                32 - 2 * depth * h,
                0.0,
                2 - 32 * h,
            ]
            values = {}

            for group, total in zip(
                POWER_GROUPS,
                [
                    700 + 4500 * envelope,
                    500 + 1800 * envelope,
                    650 + 180 * envelope,
                ],
            ):
                modulation = (
                    1 + 0.03 * math.sin(8 * math.pi * u + phase)
                )
                phases = [
                    round(total * modulation * share, 6)
                    for share in [0.32, 0.34, 0.34]
                ]
                for channel, value in zip(
                    ["A", "B", "C"], phases
                ):
                    values[group + "/" + channel] = value
                values[group + "/Sum"] = round(sum(phases), 6)

            values.update(
                {
                    "Spindle/actSpeed": round(
                        1800
                        * envelope
                        * (1 + 0.001 * math.sin(12 * u)),
                        6,
                    ),
                    "Spindle/cmdSpeed": round(
                        1800 * envelope, 6
                    ),
                    "Spindle/driveLoad": round(
                        30 * envelope, 6
                    ),
                    "Spindle/aaPower": round(
                        2200 * envelope, 6
                    ),
                    "Spindle/turnState": 1 if running else 0,
                }
            )

            for k, axis in enumerate("XYZ"):
                velocity = (
                    0.0
                    if not running or old_positions is None
                    else (
                        positions[k] - old_positions[k]
                    ) / (t - old_time)
                )
                axis_values = {
                    "aaLeadP": positions[k],
                    "aaVactB": velocity,
                    "actFeedRate": velocity * 60,
                    "aaLoad": abs(velocity) * 12,
                    "aaTorque": velocity * 2,
                    "aaPower": abs(velocity) * 100,
                }
                for name, value in axis_values.items():
                    values[f"Axes/{axis}/{name}"] = round(
                        value, 6
                    )

            tool = tool_names[min(7, int(u * 8))]
            values.update(
                {
                    "Program/actBlock": f"N{10 + int(u * 80)} G1",
                    "Program/actLineNumber": 10 + int(u * 80),
                    "Program/msg": message,
                    "Program/progName1": (
                        "_N_TURM_PSC1_MPF"
                        if running
                        else "_N_ASUP2_SYF"
                    ),
                    "State/actToolIdent": tool,
                    "State/actTNumber": TOOLS[tool],
                    "State/actToolLength1": 100.0,
                    "State/actToolRadius": 0.4,
                    "State/progStatus": 3 if running else 4,
                    "State/resetActive": not running,
                    "State/progStarts": i + 1,
                }
            )

            for prefix in ["", "MaxxTurn45/"]:
                for k, channel in enumerate(LATHE_CHANNELS):
                    value = values[channel]
                    if channel.startswith(tuple(POWER_GROUPS)):
                        source = {
                            "proto": "modbus",
                            "ip": "192.0.2.45",
                            "host": "MaxxTurn45",
                        }
                    elif prefix:
                        source = {
                            "proto": "opcua",
                            "host": "MaxxTurn45",
                            "access": channel,
                        }
                    else:
                        source = {
                            "proto": "opcua",
                            "ip": "192.0.2.45",
                            "host": "MaxxTurn45",
                        }

                    meta = {}
                    if source["proto"] == "opcua":
                        if isinstance(value, str):
                            variant = "String"
                        elif isinstance(value, bool):
                            variant = "UInt16"
                        elif isinstance(value, int):
                            variant = "Int32"
                        else:
                            variant = "Double"

                        # UInt16 flags retain numeric OPC UA encoding.
                        if isinstance(value, bool):
                            value = int(value)
                        meta = {
                            "ClientHandle": k + 1,
                            "ServerTimestamp": _stamp(t),
                            "SourceTimestamp": _stamp(
                                t, bool(prefix)
                            ),
                            "StatusCode": "Good",
                            "VariantType": "VariantType." + variant,
                        }

                    yield self.point(
                        prefix + channel,
                        value,
                        t,
                        source,
                        meta,
                        bool(prefix),
                    )
            old_positions, old_time = positions, t

    @staticmethod
    def point(channel, value, t, source, meta, local=False):
        return {
            "stream:point": {
                "stream:id": channel,
                "stream:value": value,
                "stream:timestamp": _stamp(t, local),
                "stream:source": source,
                "stream:meta": meta,
            }
        }

    def keyence_points(self, a):
        count = max(
            3,
            math.ceil(
                (a.end - a.start - 0.04) / 0.006
            ),
        )
        roughness = self.rng.uniform(0.01, 0.12)
        interruption = self.rng.uniform(0.3, 0.65)

        for j in range(count):
            u = j / (count - 1)
            t = (
                a.start
                + 0.01
                + u * (a.end - a.start - 0.04)
            )
            if (
                u < 0.05
                or u > 0.95
                or (
                    roughness > 0.085
                    and abs(u - interruption) < 0.015
                )
            ):
                value = 999.99
            else:
                value = round(
                    24
                    + 4 * math.sin(math.pi * u) ** 2
                    + self.rng.gauss(0, roughness),
                    2,
                )
            yield self.point(
                "measurement",
                value,
                t,
                {
                    "proto": "serial",
                    "access": "/dev/ttyUSB0",
                },
                {},
                True,
            )

    def stream(self, name, points):
        oid = self.uid()
        header = [
            {"stream:name": name},
            {
                "stream:source": (
                    "IRB2600"
                    if name == "IRB2600"
                    else "machine"
                )
            },
        ]
        self.new_object(
            oid,
            "stream",
            {"list": header + points},
        )

    def make_streams(self):
        for a in self.activities:
            if a.base != "Check State":
                continue

            t = a.start + 0.02
            values = self.robot_values(t)
            points = []
            for channel, value in values.items():
                if isinstance(value, bool):
                    typ = "Boolean"
                elif isinstance(value, str):
                    typ = "String"
                elif isinstance(value, float):
                    typ = "Double"
                else:
                    typ = "Int32"

                points.append(
                    self.point(
                        channel,
                        value,
                        t,
                        "IRB2600",
                        {"type": typ, "enumeration": []},
                        True,
                    )
                )
            self.stream("IRB2600", points)

            a.data.update(
                {
                    "xxx": {
                        "timestamp": _stamp(t),
                        "id": self.uid(),
                        "name": "IRB2600",
                        "path": "StateMachine",
                        "source": "IRB2600",
                        "type": "object",
                        "value": values,
                    },
                    "yyy": {
                        "timestamp": _stamp(t, True),
                        "id": self.uid(),
                        "name": "StateMachine",
                        "path": "StateMachine",
                        "source": "IRB2600",
                        "type": "String",
                        "value": values["StateMachine"],
                    },
                    "state": "true",
                    "statemachine": values["StateMachine"],
                    "check_value": (
                        values["StateMachine"] == "Finished"
                    ),
                }
            )

        for a in self.streams:
            self.stream(a.component, [])
            a.receives = [a.start + 0.005]
            points = []
            source_points = (
                self.lathe_frames(a)
                if a.component == "MaxxTurn45"
                else self.keyence_points(a)
            )
            for point in source_points:
                points.append(point)
                if len(points) == 334:
                    self.stream(a.component, points)
                    last_time = _time(
                        points[-1]["stream:point"][
                            "stream:timestamp"
                        ]
                    )
                    a.receives.append(
                        max(
                            last_time + 0.002,
                            a.receives[-1] + 0.000001,
                        )
                    )
                    points = []
            if points:
                self.stream(a.component, points)
                last_time = _time(
                    points[-1]["stream:point"][
                        "stream:timestamp"
                    ]
                )
                a.receives.append(
                    max(
                        last_time + 0.002,
                        a.receives[-1] + 0.000001,
                    )
                )

    def record(
        self,
        p,
        aid,
        base,
        transition,
        t,
        data=None,
        extra=None,
        component="",
        endpoint=False,
    ):
        omap = [p.id]
        if base != GHOST:
            omap.append(aid)
        if extra:
            omap.append(extra.id)
        if component:
            omap.append(self.components[component])

        vmap = {
            "concept:instance": p.instance,
            "id:id": (
                "external"
                if aid == "external"
                else MODEL_IDS[base]
            ),
            "cpee:activity_uuid": aid,
            "lifecycle:transition": TRANSITIONS[transition],
            "cpee:lifecycle:transition": transition,
        }
        if endpoint:
            vmap["concept:endpoint"] = HTTP + (
                "robot" if component == "IRB2600" else "invoke"
            )
        if extra:
            vmap["sub:root"] = p.name
        if data is not None:
            vmap["data"] = self.snapshot(data)

        self.records.append(
            (t, p, base, omap, vmap)
        )

    def compile(self):
        # The complete subprocess/activity/resource graph already exists.
        self.make_streams()

        for p in self.processes:
            state = (
                _part_state(p.part)
                if p.part >= 0
                else self.configuration()
            )
            state = dict(
                state,
                behavior=(
                    "fork_running"
                    if p.name == BATCH
                    else "wait_running"
                ),
            )
            for offset, transition, payload in [
                (0.001, "dataelements/change", state),
                (
                    0.002,
                    "endpoints/change",
                    {"endpoints": None},
                ),
                (
                    0.003,
                    "attributes/change",
                    {
                        "attributes": None,
                        "customization": None,
                    },
                ),
                (
                    0.004,
                    "description/change",
                    {"info": p.name, "modeltype": "CPEE"},
                ),
                (0.005, "state/change", {"state": "ready"}),
                (0.006, "state/change", {"state": "running"}),
            ]:
                self.record(
                    p,
                    "external",
                    "external",
                    transition,
                    p.start + offset,
                    payload,
                )

            if p.name == BATCH:
                for j, value in enumerate(
                    ["stopping", "stopped", "running"]
                ):
                    self.record(
                        p,
                        "external",
                        "external",
                        "state/change",
                        p.end - 0.01 + j * 0.001,
                        {"state": value},
                    )

            self.record(
                p,
                "external",
                "external",
                "state/change",
                p.end,
                {"state": "finished"},
            )

            # Retain unreferenced intermediate source-style snapshots.
            self.snapshot(state)
            self.snapshot(
                {
                    "endpoints": None,
                    "attributes": None,
                    "customization": None,
                }
            )

        for a in self.activities:
            s, e = a.start, a.end
            if a.kind in {"nested", "spawn"}:
                entries = [
                    ("activity/calling", s),
                    ("activity/receiving", s + 0.02),
                    ("task/instantiation", s + 0.04),
                ]
                if a.kind == "nested":
                    entries += [
                        (
                            "activity/receiving",
                            a.child.end + 0.01,
                        ),
                        (
                            "subprocess/receiving",
                            a.child.end + 0.02,
                        ),
                    ]
                entries.append(("activity/done", e))
            elif a.kind == "sensor":
                entries = (
                    [("activity/calling", s)]
                    + [
                        ("activity/receiving", t)
                        for t in a.receives
                    ]
                    + [("activity/done", e)]
                )
            elif a.kind == "read":
                entries = [
                    ("activity/calling", s),
                    ("activity/receiving", s + 0.02),
                    ("dataelements/change", e - 0.01),
                    ("activity/done", e),
                ]
            elif a.kind == "update":
                entries = [
                    ("dataelements/change", s),
                    ("activity/done", e),
                ]
            elif a.kind == "endpoint":
                entries = [
                    ("endpoints/change", s),
                    ("activity/done", e),
                ]
            elif a.kind == "fetch_endpoint":
                entries = [
                    ("dataelements/change", s),
                    ("endpoints/change", (s + e) / 2),
                    ("activity/done", e),
                ]
            elif a.kind == "ghost":
                entries = [("activity/done", e)]
            else:
                entries = [
                    ("activity/calling", s),
                    ("activity/receiving", (s + e) / 2),
                    ("activity/done", e),
                ]

            for transition, t in entries:
                payload = (
                    a.data
                    if transition in {
                        "dataelements/change",
                        "endpoints/change",
                    }
                    else None
                )
                if (
                    a.data
                    and transition == "activity/receiving"
                    and a.kind == "service"
                ):
                    payload = a.data
                if (
                    a.kind == "spawn"
                    and transition == "task/instantiation"
                ):
                    payload = a.data
                extra = (
                    a.child
                    if transition in {
                        "task/instantiation",
                        "subprocess/receiving",
                    }
                    else None
                )
                self.record(
                    a.owner,
                    a.id,
                    a.base,
                    transition,
                    t,
                    payload,
                    extra,
                    a.component,
                    a.kind not in {
                        "update",
                        "endpoint",
                        "fetch_endpoint",
                        "ghost",
                    },
                )

        # Every referenced object exists before actual event serialization.
        events = {}
        for t, p, base, omap, vmap in sorted(
            self.records, key=lambda row: row[0]
        ):
            events[self.uid()] = {
                "ocel:activity": (
                    base + "§" + vmap["cpee:lifecycle:transition"]
                ),
                "ocel:timestamp": _stamp(t),
                "ocel:omap": omap,
                "ocel:vmap": vmap,
            }

        return {
            "ocel:global-log": {
                "OCEL_VERSION": "0.1",
                "OCEL_ORDERING": "timestamp",
                "OCEL_ATTN": ATTRIBUTES[:],
                "OCEL_OBJT": OBJECT_TYPES[:],
            },
            "ocel:global-event": {
                "ocel:activity": "__INVALID__"
            },
            "ocel:global-object": {
                "ocel:type": "__INVALID__"
            },
            "ocel:events": events,
            "ocel:objects": self.objects,
        }


def generate_ocel(seed: int, number_of_parts: int) -> dict:
    if type(seed) is not int:
        raise ValueError(
            "seed must be an integer (not a boolean)"
        )
    if type(number_of_parts) is not int or number_of_parts != 8:
        raise ValueError(
            "Exact-source batch composition requires "
            "number_of_parts=8: "
            "order1=[wr04,wr04,wr08,wr08,wr08], "
            "order2=[wr04,wr04,wr08]."
        )
    simulator = _Simulator(seed)
    simulator.build()
    ocel = simulator.compile()
    validate_ocel(ocel)
    return ocel


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _number(value):
    return type(value) in (int, float) and math.isfinite(value)


def _json_value(value):
    if value is None or type(value) in (str, bool, int):
        return
    if type(value) is float:
        _require(
            math.isfinite(value),
            "Nonfinite number is not a JSON sensor/state value",
        )
    elif type(value) is list:
        for item in value:
            _json_value(item)
    elif type(value) is dict:
        for key, item in value.items():
            _require(
                type(key) is str,
                "JSON object keys must be strings",
            )
            _json_value(item)
    else:
        raise ValueError(
            "Non-JSON value: " + type(value).__name__
        )


DATA_NAMES = set(
    """
    order order_id order_item order_list count initial current_tray
    tray_positions trays qr seed params program_params program_patterns
    patterns programs info modeltype theme customer status artefacts
    explanation creator author design_stage design_dir sensors behavior
    url init endpoints attributes customization timeout start_instance
    start_url start_git command ipc_receive ipc_send fetch fetch_microvu
    measure decide at move_lift opc_execute opc_read opc_write ep_check
    urls endpoint_url namespace_index identifier_type identifier parameters
    value includes move x y z gripper program vars check_var1 check_var2
    content logon set assign reset start started assign_program
    assignment_path assignment_path_stub setting_path setting_path_stub
    dummy_assignment_path dummy_setting_path program_path program_value
    program_url dir name mode regex xxx yyy state statemachine check
    check_value queue finished target trigger trigger_path trigger_start
    trigger_pause trigger_end trigger_start_value trigger_pause_value
    trigger_end_value event_batch refinement timing test id message ttl
    delete machine label magazine_x magazine_z x_measure y_measure z_measure
    """.split()
)


def _validate_data(obj, oid):
    entries = obj["ocel:ovmap"]["list"]
    _require(
        entries is None or isinstance(entries, list),
        f"Invalid data list: {oid}",
    )
    values = {}
    for entry in entries or []:
        _require(
            isinstance(entry, dict)
            and set(entry) == {"name", "value"},
            f"Invalid data entry: {oid}",
        )
        key = entry["name"]
        _require(
            isinstance(key, str)
            and key in DATA_NAMES
            and key not in values,
            f"Unknown/duplicate data field: {key}",
        )
        values[key] = entry["value"]

    for key in ("timing", "at"):
        if key not in values:
            continue
        value = values[key]
        _require(
            type(value) is str
            and re.fullmatch(
                r"[0-9]{4}-[0-9]{2}-[0-9]{2} "
                r"[0-9]{2}:[0-9]{2}:[0-9]{2} \+0100",
                value,
            ),
            f"{key} must use YYYY-MM-DD HH:MM:SS +0100",
        )
        datetime.strptime(value, TIMING_FORMAT)

    if "timing" in values:
        _require(
            "at" in values
            and datetime.strptime(values["timing"], TIMING_FORMAT)
            < datetime.strptime(values["at"], TIMING_FORMAT),
            "timing must precede its at observation timestamp",
        )

    for key in [
        "program",
        "vars",
        "content",
        "program_value",
        "program_url",
        "qr",
        "order_id",
        "order_item",
    ]:
        if key in values:
            _require(
                type(values[key]) is str,
                f"{key} must be a string",
            )

    if "behavior" in values:
        _require(
            values["behavior"] in {
                "wait_running",
                "fork_running",
            },
            "Invalid behavior",
        )
    if "started" in values:
        _require(
            values["started"] is True,
            "started must be JSON true",
        )
    if "initial" in values:
        _require(
            values["initial"] is None
            or type(values["initial"]) is bool,
            "Invalid initial flag",
        )

    for key in ["check_value", "value"]:
        if key in values:
            _require(
                type(values[key]) in (bool, str),
                f"Invalid mixed-type {key}",
            )
            if key == "check_value":
                _require(
                    values[key] in (
                        True, False, "true", "false"
                    ),
                    "Invalid check_value domain",
                )

    if "state" in values:
        _require(
            type(values["state"]) is str,
            "Source state encoding must remain a string",
        )

    for key in [
        "namespace_index",
        "identifier_type",
        "identifier",
        "endpoint_url",
    ]:
        if key in values:
            _require(
                type(values[key]) is str,
                f"{key} must retain its string encoding",
            )
    if "namespace_index" in values:
        _require(
            values["namespace_index"] == "2",
            "Incorrect OPC namespace",
        )
    if "identifier_type" in values:
        _require(
            values["identifier_type"] == "string",
            "Incorrect OPC identifier type",
        )

    if "parameters" in values:
        _require(
            type(values["parameters"]) is str,
            "OPC parameters must be serialized JSON",
        )
        try:
            json.loads(values["parameters"])
        except (ValueError, TypeError) as exc:
            raise ValueError(
                "Invalid serialized OPC parameters"
            ) from exc

    for key in ["magazine_x", "magazine_z"]:
        if key in values:
            _require(
                type(values[key]) is str
                or _number(values[key]),
                f"Invalid {key}",
            )

    for key in [
        "trigger_start",
        "trigger_pause",
        "trigger_end",
        "trigger_start_value",
        "trigger_pause_value",
        "trigger_end_value",
    ]:
        if key in values:
            value = values[key]
            _require(
                type(value) is int
                or (
                    type(value) is str
                    and re.fullmatch(r"-?\d+", value)
                ),
                f"Invalid {key}",
            )

    for key in [
        "params",
        "program_params",
        "patterns",
        "program_patterns",
        "trays",
        "tray_positions",
        "order_list",
    ]:
        if key in values:
            _require(
                type(values[key]) is list,
                f"{key} must be an array",
            )

    if "count" in values:
        i = values["count"]
        _require(
            type(i) is int and 0 <= i < 8,
            "Invalid part count",
        )
        expected = _part_state(i)
        for key, value in expected.items():
            _require(
                key in values and values[key] == value,
                f"Part {i}: inconsistent {key}",
            )
        _require(
            values["initial"] is (i == 0),
            "initial must be a boolean tied to count",
        )

    if any(
        key in values
        for key in [
            "order_id",
            "order_item",
            "program_params",
            "current_tray",
        ]
    ):
        _require(
            "count" in values,
            "A part-state snapshot is missing its selection count",
        )

    if "params" in values and "program_params" in values:
        _require(
            values["params"] == values["program_params"],
            "Program replacement parameters disagree",
        )

    if "order" in values:
        _require(
            values["order"] == ORDER,
            "Changed fixed eight-item order configuration",
        )
    if "program_patterns" in values:
        _require(
            values["program_patterns"] == PATTERNS,
            "Changed program patterns",
        )
    if "patterns" in values:
        _require(
            values["patterns"] == PATTERNS,
            "Changed replacement patterns",
        )
    if "programs" in values:
        _require(
            values["programs"]
            == {
                item: PROGRAM_URL
                for item in ["wr04", "wr08"]
            },
            "Inconsistent program mapping",
        )
    if "seed" in values:
        _require(
            type(values["seed"]) is int
            and values["seed"] == 1,
            "The source configuration seed is the literal 1",
        )
    if "tray_positions" in values:
        _require(
            values["tray_positions"]
            == [
                [_tray(row * 6 + col) for col in range(6)]
                for row in range(8)
            ],
            "Invalid tray grid",
        )
    if "trays" in values:
        _require(
            values["trays"]
            == [
                [row, col]
                for row in range(8)
                for col in range(6)
            ],
            "Invalid tray index grid",
        )
    return values


def validate_ocel(ocel: dict) -> None:
    """Raise ValueError for malformed schema or violated invariants.

    Timestamps, shared interface events, and data foreign keys reconstruct
    ownership and causality. Dictionary insertion order is ignored.
    Validation does not regenerate the log or depend on its random seed.
    """
    try:
        _validate_ocel(ocel)
    except ValueError:
        raise
    except (
        KeyError,
        TypeError,
        IndexError,
        AttributeError,
        OverflowError,
    ) as exc:
        raise ValueError(
            "Malformed OCEL structure: " + str(exc)
        ) from exc


def _validate_ocel(ocel):
    _require(
        type(ocel) is dict
        and set(ocel)
        == {
            "ocel:global-log",
            "ocel:global-event",
            "ocel:global-object",
            "ocel:events",
            "ocel:objects",
        },
        "Invalid early OCEL 0.1 top-level schema",
    )
    _json_value(ocel)

    g = ocel["ocel:global-log"]
    _require(
        set(g)
        == {
            "OCEL_VERSION",
            "OCEL_ORDERING",
            "OCEL_ATTN",
            "OCEL_OBJT",
        },
        "Invalid global-log fields",
    )
    _require(
        g["OCEL_VERSION"] == "0.1"
        and g["OCEL_ORDERING"] == "timestamp",
        "Wrong OCEL dialect/ordering",
    )
    _require(
        type(g["OCEL_ATTN"]) is list
        and len(g["OCEL_ATTN"]) == 9
        and set(g["OCEL_ATTN"]) == set(ATTRIBUTES),
        "Invalid attribute declarations",
    )
    _require(
        type(g["OCEL_OBJT"]) is list
        and len(g["OCEL_OBJT"]) == 94
        and set(g["OCEL_OBJT"]) == set(OBJECT_TYPES),
        "Invalid object-type declarations",
    )
    _require(
        ocel["ocel:global-event"]
        == {"ocel:activity": "__INVALID__"},
        "Invalid event default",
    )
    _require(
        ocel["ocel:global-object"]
        == {"ocel:type": "__INVALID__"},
        "Invalid object default",
    )

    objects = ocel["ocel:objects"]
    events = ocel["ocel:events"]
    _require(
        type(objects) is dict
        and type(events) is dict
        and objects
        and events,
        "Events and objects must be nonempty ID dictionaries",
    )

    canonical = set()
    for identifier in list(objects) + list(events):
        if identifier == "external" and identifier in objects:
            continue
        _require(
            type(identifier) is str
            and re.fullmatch(
                r"(?:[0-9a-f]{32}|"
                r"[0-9a-f]{8}(?:-[0-9a-f]{4}){3}-[0-9a-f]{12})",
                identifier,
            ),
            "Invalid UUID: " + str(identifier),
        )
        key = uuid.UUID(identifier).hex
        _require(
            key not in canonical,
            "ID collision across event/object families: "
            + identifier,
        )
        canonical.add(key)

    _require(
        "external" not in events
        and objects.get("external")
        == {
            "ocel:type": ACT + "external",
            "ocel:ovmap": {},
        },
        "Invalid singleton external object",
    )

    types = {}
    data = {}
    component_ids = {}

    for oid, obj in objects.items():
        _require(
            type(obj) is dict
            and set(obj) == {"ocel:type", "ocel:ovmap"},
            "Invalid object schema: " + oid,
        )
        typ = obj["ocel:type"]
        attrs = obj["ocel:ovmap"]
        _require(
            typ in OBJECT_TYPES and type(attrs) is dict,
            "Invalid object type or attributes",
        )
        types[oid] = typ

        if typ in {"data", "stream"}:
            _require(
                set(attrs) == {"list"},
                "Snapshot/stream attributes must have only list",
            )
            if typ == "data":
                data[oid] = _validate_data(obj, oid)
        else:
            _require(
                attrs == {},
                "Abstraction and component objects must have no attributes",
            )

        if typ.startswith(COMP):
            name = typ[len(COMP):]
            _require(
                name not in component_ids,
                "Duplicate component singleton",
            )
            component_ids[name] = oid

        if typ.startswith(ACT):
            _require(
                oid == "external"
                or re.fullmatch(r"[0-9a-f]{32}", oid),
                "Activity object ID must be 32 hex characters",
            )
        elif oid != "external":
            _require(
                str(uuid.UUID(oid)) == oid,
                "Non-activity objects must use hyphenated UUIDs",
            )

    _require(
        set(types.values())
        == set(OBJECT_TYPES) - {ACT + GHOST},
        "Must retain 93 instantiated and 94 declared types, "
        "including the missing Wait While Moving Up type",
    )
    _require(
        set(component_ids)
        == {"IRB2600", "MaxxTurn45", "keyence", "NA"},
        "Missing component",
    )

    process_ids = {
        oid
        for oid, typ in types.items()
        if typ.startswith(SUB)
    }
    names = {
        p: types[p][len(SUB):]
        for p in process_ids
    }
    pevents = defaultdict(list)
    groups = defaultdict(list)
    instances = {}
    reverse_instances = {}
    owners = {}
    model_ids = {}
    reverse_models = {}
    data_users = Counter()
    referenced_activities = set()
    ghost_ids = set()
    required = set(ATTRIBUTES[:5])

    for eid, event in events.items():
        _require(
            type(event) is dict
            and set(event)
            == {
                "ocel:activity",
                "ocel:timestamp",
                "ocel:omap",
                "ocel:vmap",
            },
            "Invalid event schema: " + eid,
        )
        label = event["ocel:activity"]
        omap = event["ocel:omap"]
        v = event["ocel:vmap"]

        _require(
            type(omap) is list
            and 1 <= len(omap) <= 3
            and len(omap) == len(set(omap)),
            "Invalid omap cardinality/duplicates",
        )
        _require(
            all(oid in objects for oid in omap),
            "Dangling omap reference",
        )
        _require(
            type(v) is dict
            and required <= set(v)
            <= set(ATTRIBUTES) - {"cpee:uuid"},
            "Invalid event attribute set; cpee:uuid must remain absent",
        )

        transition = v["cpee:lifecycle:transition"]
        aid = v["cpee:activity_uuid"]
        _require(
            transition in TRANSITIONS
            and v["lifecycle:transition"]
            == TRANSITIONS[transition],
            "Incorrect lifecycle mapping",
        )
        _require(
            type(label) is str and label.count("§") == 1,
            "Invalid activity label delimiter",
        )
        base, suffix = label.split("§")
        _require(
            base in BASES and suffix == transition,
            "Label/transition mismatch or altered source spelling",
        )

        t = _time(event["ocel:timestamp"], True)
        owner = omap[0]
        _require(
            owner in process_ids,
            "First omap object must be the owning subprocess",
        )
        instance = v["concept:instance"]
        _require(
            type(instance) is int,
            "concept:instance must be an integer",
        )
        _require(
            instance not in instances
            or instances[instance] == owner,
            "One concept:instance has multiple owners",
        )
        _require(
            owner not in reverse_instances
            or reverse_instances[owner] == instance,
            "Subprocess has multiple concept:instance values",
        )
        instances[instance] = owner
        reverse_instances[owner] = instance

        if base == GHOST:
            _require(
                transition == "activity/done"
                and omap == [owner]
                and aid not in objects,
                "Preserve the orphan Wait While Moving Up completion",
            )
            _require(
                re.fullmatch(r"[0-9a-f]{32}", aid)
                and aid not in canonical
                and aid not in ghost_ids,
                "Invalid/duplicate orphan activity UUID",
            )
            ghost_ids.add(aid)
        else:
            _require(
                aid in objects
                and types[aid] == ACT + base
                and omap[1:2] == [aid],
                "Activity UUID/object relation mismatch",
            )
            referenced_activities.add(aid)
            if base == "external":
                _require(
                    aid == "external"
                    and v["id:id"] == "external"
                    and len(omap) == 2,
                    "Invalid external notification",
                )
            else:
                _require(
                    aid not in owners or owners[aid] == owner,
                    "Activity execution belongs to multiple subprocesses",
                )
                owners[aid] = owner

        if base != "external":
            model_id = v["id:id"]
            _require(
                type(model_id) is str
                and re.fullmatch(r"a\d+", model_id),
                "Invalid modeled activity ID",
            )
            key = (names[owner], base)
            reverse = (names[owner], model_id)
            _require(
                key not in model_ids or model_ids[key] == model_id,
                "id:id changed within a subprocess definition",
            )
            _require(
                reverse not in reverse_models
                or reverse_models[reverse] == base,
                "Two modeled activities share id:id within one definition",
            )
            model_ids[key] = model_id
            reverse_models[reverse] = base

        if "concept:endpoint" in v:
            _require(
                type(v["concept:endpoint"]) is str
                and re.match(
                    r"^[a-z][a-z0-9+.-]*://\S+$",
                    v["concept:endpoint"],
                ),
                "Invalid endpoint URI",
            )

        if "data" in v:
            _require(
                type(v["data"]) is str
                and v["data"] in data
                and v["data"] not in omap,
                "Invalid data foreign key",
            )
            data_users[v["data"]] += 1
            _require(
                data_users[v["data"]] == 1,
                "An immutable data snapshot is reused by multiple events",
            )

        if transition in {
            "dataelements/change",
            "endpoints/change",
            "attributes/change",
            "description/change",
            "state/change",
        }:
            _require(
                "data" in v,
                "Missing state/notification snapshot",
            )

        interface = transition in {
            "task/instantiation",
            "subprocess/receiving",
        }
        _require(
            ("sub:root" in v) == interface,
            "sub:root is required only on shared interface events",
        )
        if interface:
            _require(
                len(omap) == 3
                and omap[2] in process_ids
                and omap[2] != owner
                and v["sub:root"] == names[owner],
                "Invalid parent-child interface",
            )
        elif base == "Check State":
            _require(
                len(omap) == 3
                and omap[2] == component_ids["IRB2600"],
                "Every Check State event must link IRB2600",
            )
        elif base == "Fetch":
            _require(
                len(omap) == 3
                and omap[2]
                in {
                    component_ids["MaxxTurn45"],
                    component_ids["keyence"],
                },
                "Every Fetch event must link its component",
            )
        else:
            _require(
                len(omap) == (1 if base == GHOST else 2),
                "Unexpected component or subprocess E2O relation",
            )

        record = {
            "id": eid,
            "t": t,
            "base": base,
            "tr": transition,
            "v": v,
            "omap": omap,
            "owner": owner,
        }
        pevents[owner].append(record)
        groups[(owner, aid)].append(record)

    _require(
        set(pevents) == process_ids and len(process_ids) == 217,
        "Missing subprocess events or wrong dependent subprocess count",
    )
    _require(
        len(ghost_ids) == 4,
        "Exactly four orphan measurement completions are required",
    )
    _require(
        referenced_activities
        == {
            oid
            for oid, typ in types.items()
            if typ.startswith(ACT)
        },
        "Unreferenced/unused lifecycle object",
    )
    _require(
        len(data_users) < len(data),
        "Source fidelity requires retained unreferenced data snapshots",
    )

    for rows in pevents.values():
        rows.sort(key=lambda r: r["t"])
    bounds = {
        p: (rows[0]["t"], rows[-1]["t"])
        for p, rows in pevents.items()
    }
    activities = {}
    byprocess = defaultdict(list)
    parent = {}
    child_calls = {}

    for (p, aid), rows in groups.items():
        rows.sort(key=lambda r: r["t"])
        base = rows[0]["base"]
        sequence = [r["tr"] for r in rows]
        _require(
            all(r["base"] == base for r in rows),
            "One activity UUID has multiple labels",
        )
        _require(
            all(
                a["t"] < b["t"]
                for a, b in zip(rows, rows[1:])
            ),
            "Lifecycle timestamps must have positive micro-delays",
        )

        if base == "external":
            counts = Counter(sequence)
            _require(
                all(
                    counts[x] == 1
                    for x in [
                        "dataelements/change",
                        "endpoints/change",
                        "attributes/change",
                        "description/change",
                    ]
                ),
                "Missing/duplicate external configuration notifications",
            )
            _require(
                set(counts)
                == {
                    "dataelements/change",
                    "endpoints/change",
                    "attributes/change",
                    "description/change",
                    "state/change",
                }
                and counts["state/change"] >= 2,
                "Invalid external notification vocabulary",
            )
            states = [
                data[r["v"]["data"]].get("state")
                for r in rows
                if r["tr"] == "state/change"
            ]
            _require(
                states[0] == "ready"
                and "running" in states
                and states[-1] == "finished",
                "Invalid subprocess external state lifecycle",
            )
            _require(
                rows[-1]["t"] == bounds[p][1],
                "Activity outlives its subprocess finished state",
            )
            running_time = next(
                r["t"]
                for r in rows
                if r["tr"] == "state/change"
                and data[r["v"]["data"]].get("state") == "running"
            )
            _require(
                all(
                    r["t"] > running_time
                    for r in pevents[p]
                    if r["base"] != "external"
                ),
                "Activity starts before subprocess running state",
            )
            continue

        if base == GHOST:
            expected = ["activity/done"]
        elif base in CHILD_TYPE or base == "Spawn Production":
            expected = [
                "activity/calling",
                "activity/receiving",
                "task/instantiation",
            ]
            if base != "Spawn Production":
                expected += [
                    "activity/receiving",
                    "subprocess/receiving",
                ]
            expected += ["activity/done"]
        elif base == "Fetch":
            _require(
                sequence.count("activity/receiving") >= 2,
                "Fetch needs empty/start and measurement bundles",
            )
            expected = (
                ["activity/calling"]
                + ["activity/receiving"]
                * sequence.count("activity/receiving")
                + ["activity/done"]
            )
            _require(
                len({r["omap"][2] for r in rows}) == 1,
                "Fetch changed component during acquisition",
            )
        elif base in READ:
            expected = [
                "activity/calling",
                "activity/receiving",
                "dataelements/change",
                "activity/done",
            ]
        elif base in UPDATE:
            expected = [
                "dataelements/change",
                "activity/done",
            ]
        elif base in ENDPOINT:
            expected = [
                "endpoints/change",
                "activity/done",
            ]
        elif base == "Set Fetch Endpoint":
            expected = [
                "dataelements/change",
                "endpoints/change",
                "activity/done",
            ]
        else:
            expected = [
                "activity/calling",
                "activity/receiving",
                "activity/done",
            ]

        _require(
            sequence == expected,
            f"Invalid lifecycle bundle for {base}: {sequence}",
        )
        activity = {
            "id": aid,
            "owner": p,
            "base": base,
            "s": rows[0]["t"],
            "e": rows[-1]["t"],
            "rows": rows,
        }
        activities[aid] = activity
        byprocess[p].append(activity)

        if "task/instantiation" in sequence:
            interface = rows[2]
            child = interface["omap"][2]
            _require(
                child not in parent,
                "Child subprocess has multiple instantiations/parents",
            )
            parent[child] = p
            child_calls[child] = activity
            activity["child"] = child

            if base == "Spawn Production":
                _require(
                    names[child]
                    in {"Turm Single wr04", "Turm Single wr08"},
                    "Spawn created wrong child type",
                )
            else:
                _require(
                    names[child] == CHILD_TYPE[base],
                    "Wrapper/child subprocess type mismatch",
                )
                _require(
                    rows[-2]["omap"][2] == child,
                    "Instantiation and return refer to different children",
                )
                _require(
                    bounds[child][1]
                    < rows[-3]["t"]
                    < rows[-2]["t"]
                    < activity["e"],
                    "Child completion/return ordering violated",
                )
            _require(
                interface["t"] < bounds[child][0]
                and bounds[p][0]
                < bounds[child][0]
                < bounds[child][1]
                <= bounds[p][1],
                "Child escapes parent lifetime or starts before instantiation",
            )

    roots = process_ids - set(parent)
    _require(
        len(roots) == 1
        and names[next(iter(roots))] == BATCH,
        "Hierarchy must have exactly one batch root",
    )
    root = next(iter(roots))
    for p in process_ids:
        seen = set()
        node = p
        while node in parent:
            _require(
                node not in seen,
                "Cycle in subprocess hierarchy",
            )
            seen.add(node)
            node = parent[node]
        _require(
            node == root,
            "Disconnected subprocess graph",
        )

    for rows in byprocess.values():
        rows.sort(key=lambda a: a["s"])

    def named(p, base):
        return [
            a for a in byprocess[p]
            if a["base"] == base
        ]

    def one(p, base):
        found = named(p, base)
        _require(
            len(found) == 1,
            f"Expected one {base} in {names[p]}",
        )
        return found[0]

    def want(p, expected):
        _require(
            Counter(a["base"] for a in byprocess[p])
            == Counter(expected),
            "Invalid workflow/cardinality in " + names[p],
        )

    def chain(p, bases):
        selected = [one(p, base) for base in bases]
        _require(
            all(
                a["e"] < b["s"]
                for a, b in zip(selected, selected[1:])
            ),
            "Workflow order violated in " + names[p],
        )
        return selected

    def payload(a, transition="dataelements/change"):
        rows = [
            r for r in a["rows"]
            if r["tr"] == transition
        ]
        _require(
            len(rows) == 1 and "data" in rows[0]["v"],
            "Required activity snapshot missing",
        )
        return data[rows[0]["v"]["data"]]

    selects = named(root, "Next QR, Order Item and Tray")
    spawns, getters, waits = (
        named(root, base)
        for base in [
            "Spawn Production",
            "Get Machine State Cancelled",
            "Wait for MT45 Takeout",
        ]
    )

    want(
        root,
        [
            "Get Order Details",
            "Inititalize Batch",
            "Not First Part",
        ]
        + [
            base
            for _ in range(8)
            for base in [
                "Next QR, Order Item and Tray",
                "Spawn Production",
                "Get Machine State Cancelled",
                "Wait for MT45 Takeout",
            ]
        ],
    )
    order_a, init_a = chain(
        root,
        ["Get Order Details", "Inititalize Batch"],
    )
    _require(
        payload(order_a).get("order") == ORDER,
        "Missing initial order snapshot",
    )
    config = payload(init_a)
    for key in [
        "order",
        "order_list",
        "tray_positions",
        "trays",
        "qr",
        "seed",
        "programs",
        "program_patterns",
    ]:
        _require(
            key in config,
            "Incomplete initial batch snapshot: " + key,
        )
    _require(
        config["order_list"] == [list(x) for x in ITEMS]
        and config["qr"] == "*268MFA466*TZHZE 001"
        and config.get("initial") is None,
        "Invalid initial batch state",
    )
    _require(
        init_a["e"] < selects[0]["s"],
        "Selection started before batch initialization",
    )

    not_first = one(root, "Not First Part")
    _require(
        waits[0]["e"]
        < not_first["s"]
        < not_first["e"]
        < selects[1]["s"]
        and payload(not_first).get("initial") is False,
        "Invalid first-part state update",
    )

    part_index = {}
    part_ids = []
    lathe_intervals = []
    robot_intervals = []
    previous_unload = None

    for i, (sel, spawn, getter, wait_a) in enumerate(
        zip(selects, spawns, getters, waits)
    ):
        state = payload(sel)
        _require(
            state == _part_state(i),
            "Selection count/order/suffix/tray mismatch",
        )
        _require(
            sel["e"] < getter["s"]
            and getter["e"] < spawn["s"]
            and spawn["e"] < wait_a["s"],
            "Batch scheduling order violated",
        )
        if i:
            _require(
                waits[i - 1]["e"] < sel["s"],
                "Next part selected before the preceding takeout barrier",
            )

        p = spawn["child"]
        part_index[p] = i
        part_ids.append(p)
        item = ITEMS[i][1]
        _require(
            names[p] == "Turm Single " + item,
            "Selected item and spawned subprocess disagree",
        )
        _require(
            payload(spawn, "task/instantiation") == state,
            "Spawn uses the wrong part state",
        )

        route = [
            "Create Program",
            "Turn",
            "Set Program MT45",
            "MT45 Start",
            "MT45 Take Out",
        ]
        if item == "wr04":
            route += [
                "Robot Measuring and Traying",
                "Measure with Keyence",
                "Signal Keyence End",
            ]
        else:
            route += ["Robot Traying"]
        if i:
            route += ["Wait For Unload End"]
        want(p, route)

        create, set_program, start = chain(
            p,
            [
                "Create Program",
                "Set Program MT45",
                "MT45 Start",
            ],
        )
        turn = one(p, "Turn")
        take = one(p, "MT45 Take Out")
        _require(
            turn["s"] < start["e"] < turn["e"] < take["s"],
            "Machining/start/takeout ordering violated",
        )

        fetch = one(
            one(turn["child"], "Machining")["child"],
            "Fetch",
        )
        _require(
            start["e"] < fetch["s"] < fetch["e"] < take["s"],
            "Lathe acquisition is not inside the machining phase",
        )
        _require(
            fetch["rows"][0]["omap"][2]
            == component_ids["MaxxTurn45"],
            "Turn uses the wrong sensor component",
        )
        lathe_intervals.append((fetch["s"], fetch["e"]))
        _require(
            take["e"] < wait_a["e"],
            "Batch takeout barrier released too early",
        )

        branch = one(
            p,
            (
                "Robot Measuring and Traying"
                if item == "wr04"
                else "Robot Traying"
            ),
        )
        _require(
            take["e"] < branch["s"],
            "Robot branch starts before completed takeout",
        )
        if i:
            barrier = one(p, "Wait For Unload End")
            _require(
                previous_unload < barrier["e"] < take["s"],
                "Unload dependency violated",
            )
        previous_unload = one(
            branch["child"],
            "Signal Unloading End",
        )["e"]

        if item == "wr04":
            q = branch["child"]
            positioning = one(q, "IRB2600 Door to Measure")
            observation = one(q, "Get Current Time")
            clock_values = payload(observation)
            _require(
                clock_values.get("timing") == _timing_stamp(positioning["s"])
                and clock_values.get("at") == _timing_stamp(observation["s"])
                and abs(observation["s"] - positioning["e"] - 0.03) < 2e-6,
                "Clock snapshot must record positioning start and the "
                "observation after positioning completes",
            )
            measurement = one(p, "Measure with Keyence")
            signal = one(p, "Signal Keyence End")
            keyfetch = one(
                one(
                    measurement["child"],
                    "Machining",
                )["child"],
                "Fetch",
            )
            _require(
                keyfetch["rows"][0]["omap"][2]
                == component_ids["keyence"],
                "Measurement uses wrong component",
            )
            down = one(q, "Move Down")
            up = one(q, "Move Up")
            _require(
                one(q, "IRB2600 Door to Measure")["e"]
                < measurement["s"]
                < keyfetch["s"]
                < down["s"],
                "Measurement starts before positioning "
                "or after downward movement",
            )
            _require(
                up["e"]
                < keyfetch["e"]
                < measurement["e"]
                < one(q, "IRB2600 Measure to Door")["s"],
                "Keyence window does not cover the beam traversal",
            )
            _require(
                branch["e"] < signal["s"],
                "Keyence-end signal precedes completed traying",
            )

    for p in process_ids:
        node = p
        while node in parent and node not in part_index:
            node = parent[node]
        if node in part_index:
            i = part_index[node]
            for record in pevents[p]:
                values = data.get(
                    record["v"].get("data"),
                    {},
                )
                if "count" in values:
                    _require(
                        values["count"] == i,
                        "Snapshot belongs to a different part",
                    )
                for key in [
                    "program",
                    "content",
                    "program_value",
                ]:
                    text_value = values.get(key, "")
                    if "Schnitttiefe:" in text_value:
                        order, item = ITEMS[i]
                        feed, depth = PARAMS[i]
                        expected_message = (
                            f"Type: {order}: {item}; "
                            f"Schnitttiefe: {depth}; Vorschub: {feed}"
                        )
                        _require(
                            expected_message in text_value,
                            "NC program message contradicts part parameters",
                        )

    _require(
        len(part_ids) == 8
        and Counter(names[p] for p in part_ids)
        == {
            "Turm Single wr04": 4,
            "Turm Single wr08": 4,
        },
        "Incorrect part composition",
    )

    fixed_workflows = {
        "X Generate NC": [
            "Set Endpoint",
            "Fetch Program",
            "Replace Patterns with Params",
        ],
        "Turm Turn Production": [
            "Detection Machining",
            "Machining",
        ],
        "Turm Keyence Measurement": [
            "Detection Machining",
            "Machining",
        ],
        "X MT45 SP": [
            "Set Paths",
            "Set Dummy Program",
            "Select Dummy Program",
            "Set Program",
            "Select Dummy Program",
        ],
        "X MT45 Start": [
            "Aux On",
            "Mode Auto",
            "Door Close",
            "NC Start",
        ],
        "Turm MT45 Take Out": [
            "MT45 Check Operational Safety",
            "IRB2600 Grip",
            "Open Clamp 2",
            "IRB2600 Extract",
            "Reset Clamp 2",
            "Taken Out",
        ],
        "MT45 Check Operational Safety": [
            "Get Magazine X",
            "Get Magazine Z",
        ],
        "Turm IRB2600 Grip": [
            "IRB2600 Portal to Door",
            "IRB2600 Door to GS",
            "IRB2600 GS to Take",
        ],
        "Turm IRB2600 Extract": [
            "IRB2600 Take to GS",
            "IRB2600 GS to Door",
        ],
        "Turm IRB2600 Tray": [
            "IRB2600 Door to Portal",
            "IRB2600 Unload to Tray Down",
            "Signal Unloading End",
        ],
        "Turm IRB2600 Measure and Tray": [
            "IRB2600 Door to Measure",
            "Get Current Time",
            "Move Down",
            "wait",
            "Move Up",
            GHOST,
            "IRB2600 Measure to Door",
            "IRB2600 Door to Portal",
            "IRB2600 Unload to Tray Down",
            "Signal Unloading End",
        ],
        "Machining V2": [
            "Correct  Queue",
            "Fetch",
        ],
    }
    robot_names = set(MOVES.values())

    for p in process_ids:
        name = names[p]
        aa = byprocess[p]

        if name in fixed_workflows:
            expected = fixed_workflows[name]
            want(p, expected)
            _require(
                [a["base"] for a in aa] == expected
                and all(
                    a["e"] < b["s"]
                    for a, b in zip(aa, aa[1:])
                ),
                "Subprocess workflow order violated: " + name,
            )
            if name == "X Generate NC":
                values = payload(
                    one(p, "Replace Patterns with Params")
                )
                _require(
                    "program" in values
                    and "params" in values
                    and values["params"] == values["program_params"]
                    and values["program"].endswith("N100 M30\n"),
                    "Missing complete substituted NC program",
                )

        elif name in {
            "X MT45 Control Getter",
            "X MT45 Control Setter",
        }:
            reads = named(p, "Read Var")
            _require(
                len(reads) >= 1,
                "Control operation has no polling",
            )
            want(
                p,
                ["Read Var"] * len(reads)
                + (
                    ["Set Var"]
                    if name.endswith("Setter")
                    else []
                ),
            )
            _require(
                all(
                    a["e"] < b["s"]
                    for a, b in zip(aa, aa[1:])
                ),
                "Overlapping/out-of-order control polls",
            )
            if name.endswith("Setter"):
                _require(
                    aa[0]["base"] == "Set Var",
                    "Read Var precedes Set Var",
                )

            identifier = (
                child_calls[p]["base"]
                if name.endswith("Setter")
                else "State/resetActive"
            )
            _require(
                all(
                    payload(a).get("identifier") == identifier
                    for a in reads
                ),
                "Control poll reads the wrong variable",
            )
            if name.endswith("Setter"):
                _require(
                    payload(
                        one(p, "Set Var"),
                        "activity/receiving",
                    ).get("identifier") == identifier,
                    "Control setter writes the wrong variable",
                )
            _require(
                payload(reads[-1]).get("check_value")
                in (True, "true"),
                "Control polling does not finish in the requested state",
            )
            _require(
                all(
                    payload(a).get("check_value")
                    in (False, "false")
                    for a in reads[:-1]
                ),
                "Polling continued after successful completion",
            )

        elif name in robot_names:
            checks = named(p, "Check State")
            _require(
                len(checks) >= 3,
                "Robot setup/operation requires state polling",
            )
            want(
                p,
                ["Set Fetch Endpoint"] * 2
                + ["Fetch Program"] * 2
                + [
                    "Assemble Porgram",
                    "Logon",
                    "Set Program",
                    "Assign Program",
                    "Start",
                ]
                + ["Check State"] * len(checks),
            )
            chain(
                p,
                [
                    "Assemble Porgram",
                    "Logon",
                    "Set Program",
                    "Assign Program",
                    "Start",
                ],
            )
            setup = [
                a
                for a in aa
                if a["base"]
                in {"Set Fetch Endpoint", "Fetch Program"}
            ]
            _require(
                [a["base"] for a in setup]
                == ["Set Fetch Endpoint", "Fetch Program"] * 2
                and all(
                    a["e"] < b["s"]
                    for a, b in zip(setup, setup[1:])
                )
                and setup[-1]["e"]
                < one(p, "Assemble Porgram")["s"],
                "Robot program fetch/setup order violated",
            )

            start_a = one(p, "Start")
            _require(
                checks[0]["e"] < start_a["s"]
                and start_a["s"]
                < checks[-1]["s"]
                < checks[-1]["e"]
                < start_a["e"],
                "Robot polls escape their controlled operation",
            )
            _require(
                all(
                    checks[j]["e"] < checks[j + 1]["s"]
                    for j in range(len(checks) - 1)
                ),
                "Overlapping robot polls",
            )
            _require(
                payload(checks[0]).get("statemachine") == "Finished"
                and payload(checks[-1]).get("statemachine")
                == "Finished",
                "Robot terminal state is not Finished",
            )
            _require(
                all(
                    payload(c).get("statemachine") == "Active"
                    for c in checks[1:-1]
                ),
                "Intermediate robot poll is not active",
            )

            values = payload(one(p, "Assemble Porgram"))
            _require(
                type(values.get("program")) is str
                and values["program"].startswith(
                    "MODULE OperationModule\n"
                )
                and values["program"].endswith("ENDMODULE\n"),
                "Missing complete robot module",
            )
            if (
                child_calls[p]["base"]
                == "IRB2600 Unload to Tray Down"
            ):
                node = p
                while node not in part_index:
                    node = parent[node]
                _require(
                    [values.get(axis) for axis in "xyz"]
                    == _tray(part_index[node]),
                    "Unload coordinates disagree with allocated tray",
                )

            robot_intervals.append(bounds[p])

        elif not (p == root or p in part_ids):
            raise ValueError(
                "Unhandled subprocess type: " + name
            )

        for a in aa:
            if a["base"] in {"Move Down", "Move Up"}:
                robot_intervals.append((a["s"], a["e"]))

    def exclusive(intervals, label):
        intervals = sorted(intervals)
        _require(
            all(
                a[1] <= b[0]
                for a, b in zip(intervals, intervals[1:])
            ),
            label + " exclusive-resource overlap",
        )

    exclusive(lathe_intervals, "Lathe machining")
    exclusive(robot_intervals, "Robot motion")

    _validate_streams(
        objects,
        activities,
        component_ids,
        data,
        parent,
        part_index,
    )


def _validate_streams(
    objects,
    activities,
    components,
    data,
    parent,
    part_index,
):
    fetches = defaultdict(list)
    checks = {}

    for a in activities.values():
        if a["base"] == "Fetch":
            component = a["rows"][0]["omap"][2]
            name = next(
                name
                for name, oid in components.items()
                if oid == component
            )
            fetches[name].append(a)
        elif a["base"] == "Check State":
            receive = next(
                r for r in a["rows"]
                if r["tr"] == "activity/receiving"
            )
            _require(
                receive["t"] not in checks,
                "Simultaneous robot snapshots violate exclusive operation",
            )
            checks[receive["t"]] = a

    for name, acquisitions in fetches.items():
        acquisitions.sort(key=lambda a: a["s"])
        _require(
            all(
                a["e"] < b["s"]
                for a, b in zip(
                    acquisitions, acquisitions[1:]
                )
            ),
            "Overlapping acquisitions on " + name,
        )

    _require(
        len(fetches["MaxxTurn45"]) == 8
        and len(fetches["keyence"]) == 4,
        "Wrong dependent sensor acquisition count",
    )

    starts = {
        name: [a["s"] for a in acquisitions]
        for name, acquisitions in fetches.items()
    }
    receive_times = {
        a["id"]: {
            round(r["t"], 6)
            for r in a["rows"]
            if r["tr"] == "activity/receiving"
        }
        for acquisitions in fetches.values()
        for a in acquisitions
    }
    used_receives = defaultdict(set)
    empty_counts = Counter()
    stream_counts = Counter()
    frames = defaultdict(dict)
    robot_samples = []
    robot_seen = set()
    aware_seen = False
    local_seen = False
    key_values = defaultdict(list)

    for oid, obj in objects.items():
        if obj["ocel:type"] != "stream":
            continue

        entries = obj["ocel:ovmap"]["list"]
        _require(
            type(entries) is list and len(entries) >= 2,
            "Invalid SensorStream header",
        )
        _require(
            type(entries[0]) is dict
            and set(entries[0]) == {"stream:name"}
            and type(entries[1]) is dict
            and set(entries[1]) == {"stream:source"},
            "Sensor headers were normalized/renamed",
        )
        name = entries[0]["stream:name"]
        _require(
            name in {"MaxxTurn45", "IRB2600", "keyence"}
            and entries[1]["stream:source"]
            == ("IRB2600" if name == "IRB2600" else "machine"),
            "Invalid stream name/source",
        )

        points = entries[2:]
        stream_counts[name] += 1
        _require(
            len(points) <= 334
            and (name != "IRB2600" or len(points) == 38),
            "Invalid SensorStream bundle size",
        )
        if not points:
            empty_counts[name] += 1
            continue

        times = []
        channels = set()
        values = {}
        owner_fetch = None

        for entry in points:
            _require(
                type(entry) is dict
                and set(entry) == {"stream:point"},
                "Invalid stream point wrapper",
            )
            point = entry["stream:point"]
            keys = {
                "stream:id",
                "stream:value",
                "stream:timestamp",
                "stream:source",
            }
            _require(
                type(point) is dict
                and keys <= set(point) <= keys | {"stream:meta"},
                "Invalid stream point schema",
            )
            channel = point["stream:id"]
            value = point["stream:value"]
            timestamp = point["stream:timestamp"]
            source = point["stream:source"]

            _require(
                type(channel) is str,
                "Sensor channel must be a string",
            )
            t = _time(timestamp)
            parsed = datetime.fromisoformat(timestamp)
            aware_seen |= parsed.tzinfo is not None
            local_seen |= parsed.tzinfo is None
            times.append(t)

            meta = point.get("stream:meta", {})
            _require(
                type(meta) is dict,
                "Invalid sensor metadata",
            )

            if name == "IRB2600":
                _require(
                    source == "IRB2600"
                    and set(meta) == {"type", "enumeration"},
                    "Invalid robot source/metadata",
                )
                _require(
                    channel in ROBOT_CHANNELS
                    and channel not in channels,
                    "Unknown or duplicated robot channel",
                )
                if type(value) is bool:
                    expected_type = "Boolean"
                elif type(value) is str:
                    expected_type = "String"
                elif type(value) is float:
                    expected_type = "Double"
                else:
                    expected_type = "Int32"
                _require(
                    meta["type"] == expected_type
                    and type(meta["enumeration"]) is list,
                    "Robot metadata/value type mismatch",
                )
                channels.add(channel)
                values[channel] = value
                continue

            index = bisect.bisect_right(starts[name], t) - 1
            _require(
                index >= 0 and t < fetches[name][index]["e"],
                "Sensor point outside a component-linked acquisition",
            )
            a = fetches[name][index]
            _require(
                owner_fetch is None
                or owner_fetch == a["id"],
                "One stream bundle spans different acquisitions",
            )
            owner_fetch = a["id"]

            _require(
                type(source) is dict and "proto" in source,
                "Invalid machine source metadata",
            )

            if name == "keyence":
                _require(
                    channel == "measurement"
                    and source.get("proto") == "serial"
                    and type(source.get("access")) is str,
                    "Invalid Keyence channel/source",
                )
                _require(
                    _number(value)
                    and 0 < value <= 999.99
                    and abs(value * 100 - round(value * 100)) < 1e-7,
                    "Keyence value must be positive, <=999.99, "
                    "and have 0.01-mm precision",
                )
                key_values[a["id"]].append((t, value))
            else:
                canonical = channel.removeprefix("MaxxTurn45/")
                _require(
                    canonical in LATHE_CHANNELS,
                    "Unknown lathe channel",
                )
                power = any(
                    canonical.startswith(group + "/")
                    for group in POWER_GROUPS
                )
                _require(
                    source["proto"]
                    == ("modbus" if power else "opcua"),
                    "Incorrect lathe source protocol",
                )
                if power:
                    _require(
                        set(source) == {"proto", "ip", "host"},
                        "Invalid Modbus source shape",
                    )
                else:
                    _require(
                        set(source)
                        in (
                            {"proto", "ip", "host"},
                            {"proto", "host", "access"},
                        ),
                        "Invalid OPC UA source shape",
                    )
                    _require(
                        set(meta)
                        == {
                            "ClientHandle",
                            "ServerTimestamp",
                            "SourceTimestamp",
                            "StatusCode",
                            "VariantType",
                        }
                        and meta["StatusCode"] == "Good",
                        "Invalid OPC UA status/metadata",
                    )
                    variant = meta["VariantType"]
                    _require(
                        variant
                        in {
                            "VariantType.Double",
                            "VariantType.UInt16",
                            "VariantType.Int32",
                            "VariantType.String",
                        },
                        "Invalid VariantType",
                    )
                    _require(
                        (
                            variant == "VariantType.String"
                            and type(value) is str
                        )
                        or (
                            variant == "VariantType.Double"
                            and _number(value)
                        )
                        or (
                            variant == "VariantType.UInt16"
                            and type(value) is int
                            and 0 <= value <= 65535
                        )
                        or (
                            variant == "VariantType.Int32"
                            and type(value) is int
                            and -(2**31) <= value < 2**31
                        ),
                        "OPC VariantType/value mismatch",
                    )
                    _require(
                        type(meta["ClientHandle"]) is int
                        and meta["ClientHandle"] > 0
                        and abs(
                            _time(meta["SourceTimestamp"]) - t
                        ) < 1e-6
                        and _time(meta["ServerTimestamp"])
                        >= t - 1e-6,
                        "Invalid OPC timestamp/handle",
                    )

                frame = frames[(a["id"], t)]
                _require(
                    channel not in frame,
                    "Duplicated lathe channel/timestamp",
                )
                frame[channel] = value

        _require(
            all(a <= b for a, b in zip(times, times[1:])),
            "Sensor samples reverse within their own normalized source clock",
        )

        if name == "IRB2600":
            _require(
                len(set(times)) == 1
                and channels == ROBOT_CHANNELS,
                "Robot snapshot must have all 38 simultaneous channels",
            )
            t = times[0]
            _require(
                t in checks and t not in robot_seen,
                "Robot stream lacks a unique Check State receiving event",
            )
            robot_seen.add(t)
            a = checks[t]
            change = next(
                r for r in a["rows"]
                if r["tr"] == "dataelements/change"
            )
            state = data[change["v"]["data"]]
            _require(
                state.get("xxx", {}).get("value") == values
                and state.get("yyy", {}).get("value")
                == values["StateMachine"]
                and state.get("statemachine")
                == values["StateMachine"],
                "Robot state snapshot and unlinked sensor bundle disagree",
            )
            for key in ["xxx", "yyy"]:
                record = state.get(key)
                _require(
                    type(record) is dict
                    and set(record)
                    == {
                        "timestamp",
                        "id",
                        "name",
                        "path",
                        "source",
                        "type",
                        "value",
                    },
                    "Incomplete nested robot state record",
                )
                _require(
                    abs(_time(record["timestamp"]) - t) < 1e-6
                    and str(uuid.UUID(record["id"])) == record["id"],
                    "Invalid nested robot snapshot timestamp/ID",
                )

            _validate_robot(values, t)
            robot_samples.append((t, values))
        else:
            expected_receive = round(times[-1] + 0.002, 6)
            while expected_receive in used_receives[owner_fetch]:
                expected_receive = round(
                    expected_receive + 0.000001, 6
                )
            _require(
                expected_receive in receive_times[owner_fetch]
                and expected_receive
                not in used_receives[owner_fetch],
                "Sensor bundle lacks its unique "
                "component-linked receiving event",
            )
            used_receives[owner_fetch].add(expected_receive)

    _require(
        aware_seen and local_seen,
        "Mixed sensor timestamp encodings were lost",
    )
    _require(
        robot_seen == set(checks)
        and stream_counts["IRB2600"] == len(checks),
        "Check State/robot-stream cardinality mismatch",
    )

    for name, acquisitions in fetches.items():
        _require(
            empty_counts[name] == len(acquisitions),
            "Each acquisition must retain its empty source bundle",
        )
        _require(
            stream_counts[name]
            == sum(
                len(receive_times[a["id"]])
                for a in acquisitions
            ),
            "Fetch-receive/stream cardinality mismatch",
        )
        for a in acquisitions:
            unused = (
                receive_times[a["id"]]
                - used_receives[a["id"]]
            )
            _require(
                unused == {round(a["s"] + 0.005, 6)},
                "Missing/extra acquisition receiving event "
                "or unlinked payload",
            )

    for a in fetches["keyence"]:
        values = sorted(key_values[a["id"]])
        _require(
            len(values) > 2
            and values[0][1] == 999.99
            and values[-1][1] == 999.99
            and any(value != 999.99 for _, value in values),
            "Keyence must include valid-beam measurements "
            "and out-of-range boundary sentinels",
        )

    robot_samples.sort()
    for (ta, va), (tb, vb) in zip(
        robot_samples, robot_samples[1:]
    ):
        dt = tb - ta
        _require(
            dt > 0,
            "Repeated robot sampling instant",
        )
        for channel in ["Pos.X", "Pos.Y", "Pos.Z"]:
            _require(
                abs(vb[channel] - va[channel])
                <= 200 * dt + 0.01,
                "Discontinuous Cartesian robot trajectory",
            )
        for i in range(1, 7):
            channel = f"RobJoint.Rax_{i}"
            _require(
                abs(vb[channel] - va[channel])
                <= 60 * dt + 0.01,
                "Discontinuous robot joint trajectory",
            )

    frame_groups = defaultdict(list)
    expected_channels = set(LATHE_CHANNELS) | {
        "MaxxTurn45/" + channel
        for channel in LATHE_CHANNELS
    }
    for (aid, t), frame in frames.items():
        _require(
            set(frame) == expected_channels,
            "Incomplete or renamed duplicate-prefix lathe channels",
        )
        for channel in LATHE_CHANNELS:
            _require(
                frame[channel] == frame["MaxxTurn45/" + channel]
                and type(frame[channel])
                is type(frame["MaxxTurn45/" + channel]),
                "Prefixed source refinement contradicts its correlated channel",
            )
        frame_groups[aid].append((t, frame))

    for a in fetches["MaxxTurn45"]:
        samples = sorted(frame_groups[a["id"]])
        _require(
            len(samples) >= 3,
            "Lathe acquisition has no complete trajectory",
        )
        node = a["owner"]
        while node not in part_index:
            _require(
                node in parent,
                "Lathe acquisition is not under a scheduled part",
            )
            node = parent[node]
        i = part_index[node]

        _require(
            samples[0][1]["Spindle/turnState"] == 0
            and samples[-1][1]["Spindle/turnState"] == 0
            and any(
                frame["Spindle/turnState"] in {1, 2}
                for _, frame in samples
            ),
            "Lathe trajectory lacks start/run/stop phases",
        )
        previous = None
        for t, frame in samples:
            _validate_lathe(frame, i, previous)
            previous = frame


def _validate_robot(values, t):
    for key, expected in ROBOT_FIXED.items():
        _require(
            values[key] == expected
            and type(values[key]) is type(expected),
            "Invalid constant robot channel: " + key,
        )

    active = values["StateMachine"] == "Active"
    _require(
        values["StateMachine"] in {"Active", "Finished"},
        "Invalid StateMachine",
    )
    _require(
        values["LastTransition"]
        == (
            "FinishedToActive"
            if active
            else "ActiveToFinished"
        ),
        "Robot transition/state mismatch",
    )
    _require(
        _time(values["LastTransitionAt"]) <= t,
        "Robot transition occurs after its observation",
    )
    _require(
        values["Rapid.ExecutionStatus"]
        == ("Running" if active else "Stopped"),
        "RAPID execution/state mismatch",
    )
    _require(
        values["Task.ExecutionStatus"]
        in (
            {"Running"}
            if active
            else {"Stopped", "Ready"}
        ),
        "Task execution/state mismatch",
    )
    _require(
        values["Task.ExecutionType"] in {"Normal", "None"},
        "Invalid task execution type",
    )

    for i, (low, high) in enumerate(JOINT_LIMITS, 1):
        value = values[f"RobJoint.Rax_{i}"]
        _require(
            _number(value) and low <= value <= high,
            "Implausible joint value",
        )
    for channel, (low, high) in zip(
        ["Pos.X", "Pos.Y", "Pos.Z"],
        POS_LIMITS,
    ):
        _require(
            _number(values[channel])
            and low <= values[channel] <= high,
            "Implausible Cartesian value",
        )


def _validate_lathe(frame, i, previous):
    strings = {
        "Program/actBlock",
        "Program/msg",
        "Program/progName1",
        "State/actToolIdent",
    }
    for channel in LATHE_CHANNELS:
        _require(
            (
                type(frame[channel]) is str
                if channel in strings
                else _number(frame[channel])
            ),
            "Incorrect lathe channel value type: " + channel,
        )

    for group in POWER_GROUPS:
        phase_sum = sum(
            frame[group + "/" + phase]
            for phase in ["A", "B", "C"]
        )
        _require(
            abs(frame[group + "/Sum"] - phase_sum)
            <= max(5, abs(phase_sum) * 0.01),
            "Three-phase active-power sum mismatch",
        )

    turn = frame["Spindle/turnState"]
    _require(
        type(turn) is int and turn in {0, 1, 2},
        "Invalid spindle turnState",
    )
    speed = frame["Spindle/actSpeed"]
    command = frame["Spindle/cmdSpeed"]
    _require(
        abs(speed - command) <= max(10, abs(command) * 0.03),
        "Actual spindle speed fails to track command",
    )

    if turn == 0:
        _require(
            abs(speed) < 1
            and abs(command) < 1
            and abs(frame["Spindle/driveLoad"]) < 1
            and abs(frame["Spindle/aaPower"]) < 1
            and frame["State/progStatus"] == 4,
            "Stopped spindle has running speed/load/program status",
        )
    else:
        _require(
            abs(speed) > 0
            and frame["Spindle/driveLoad"] > 0
            and frame["Spindle/aaPower"] > 0
            and frame["State/progStatus"] == 3,
            "Running spindle has idle speed/load/status",
        )

    _require(
        type(frame["State/progStatus"]) is int
        and frame["State/resetActive"] == int(turn == 0),
        "Invalid lathe status/reset encoding",
    )

    tool = frame["State/actToolIdent"]
    _require(
        tool in TOOLS
        and type(frame["State/actTNumber"]) is int
        and frame["State/actTNumber"] == TOOLS[tool],
        "Tool identifier/number mapping mismatch",
    )
    _require(
        frame["Program/progName1"]
        in {"_N_ASUP2_SYF", "_N_TURM_PSC1_MPF"},
        "Invalid program name",
    )
    _require(
        type(frame["Program/actLineNumber"]) is int
        and frame["Program/actLineNumber"] >= 0,
        "Invalid program line number",
    )

    order, item = ITEMS[i]
    feed, depth = PARAMS[i]
    _require(
        frame["Program/msg"]
        == (
            f"Type: {order}: {item}; "
            f"Schnitttiefe: {depth}; Vorschub: {feed}"
        )
        and type(frame["State/progStarts"]) is int
        and frame["State/progStarts"] == i + 1,
        "Lathe program state contradicts the scheduled part",
    )

    for axis in "XYZ":
        prefix = f"Axes/{axis}/"
        velocity = frame[prefix + "aaVactB"]
        rate = frame[prefix + "actFeedRate"]
        _require(
            abs(rate - 60 * velocity) < 0.001,
            "Axis feed-rate/velocity mismatch",
        )
        _require(
            frame[prefix + "aaLoad"] >= 0
            and frame[prefix + "aaPower"] >= 0
            and frame[prefix + "aaTorque"] * velocity >= -1e-6,
            "Inconsistent axis load/torque/power",
        )
        if turn == 0:
            _require(
                abs(velocity) < 1e-6 and abs(rate) < 1e-6,
                "Idle axis is moving",
            )
        if previous:
            delta = (
                frame[prefix + "aaLeadP"]
                - previous[prefix + "aaLeadP"]
            )
            _require(
                delta * velocity >= -1e-7,
                "Axis velocity sign opposes position change",
            )
            if abs(velocity) < 1e-7:
                _require(
                    abs(delta) < 0.01,
                    "Axis position jumps during an idle interval",
                )


def save_ocel(ocel: dict, output_path: str) -> None:
    validate_ocel(ocel)
    path = Path(output_path)
    if path.suffix.lower() != ".jsonocel":
        raise ValueError("output_path must end in .jsonocel")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open(
        "w",
        encoding="utf-8",
        newline="\n",
    ) as file:
        json.dump(
            ocel,
            file,
            ensure_ascii=False,
            allow_nan=False,
            indent=2,
        )
        file.write("\n")


if __name__ == "__main__":
    SEED = 42
    suffix = 2
    number_of_logs = 16
    for idx in range(suffix, suffix+number_of_logs):
        ocel = generate_ocel(seed=SEED+idx, number_of_parts=8)
        save_ocel(ocel, f"log_{idx}.jsonocel")
