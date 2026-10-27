# Container Logistics O2O constraints

[generate_o2o_constraints.py](generate_o2o_constraints.py) generates one YAML file of object-to-object (O2O) cardinality constraints for each Container Logistics OCEL 2.0 log. The constraints describe the minimum and maximum relationship counts observed in that log.

These bounds are empirical: they describe the reference data, rather than establishing universal logistics business requirements. Generate them from reference logs **before deleting or masking relationships**.

## How constraints are generated

1. **Read and validate objects.** The script reads the top-level `objects` list, maps each object ID to its type, and counts the objects of each type. IDs must be unique, and IDs and types must be non-empty strings.
2. **Read explicit O2O relationships.** For each object, the script reads its `relationships` list. The containing object is the source, `objectId` identifies the target, and `qualifier` identifies the relationship's meaning. An absent `relationships` field is treated as an empty list.
3. **Group relationships.** Each distinct `(source_type, qualifier, target_type)` combination defines one rule. Direction and exact qualifier spelling are preserved. 
4. **Count relationship rows.** Within each group, every usable relationship increments the source object's outgoing count and the target object's incoming count.
5. **Compute bounds.** The script takes the minimum and maximum counts across all objects of the relevant type.
6. **Write YAML.** Rules are sorted by `(source_type, qualifier, target_type)` and written as a top-level YAML list compatible with DVIDEO's `load_cardinality_constraints`.

Incoming counts describe the same directed relationships viewed from their targets; they do not represent reverse edges.

### Objects with zero connections

All objects of the source or target type participate in the corresponding bounds, even if they have no relationships matching that rule. For example, outgoing counts of `2`, `4`, and `0` produce `min_outgoing: 0` and `max_outgoing: 4`.

## Example

Suppose the reference log contains these objects and relationships:

```text
Container C1 --[CR for TD]--> Transport Document D1
Container C2 --[CR for TD]--> Transport Document D1
Transport Document D2 has no matching incoming relationships.
```

The generated rule is:

```yaml
- source_type: Container
  qualifier: CR for TD
  target_type: Transport Document
  min_outgoing: 1
  max_outgoing: 1
  min_incoming: 0
  max_incoming: 2
```

## Exclusions and duplicate rows

- A relationship whose target ID is missing, invalid, or absent from the object list is skipped and counted in `skipped_missing_targets`.
- A relationship with a missing, null, empty, or whitespace-only qualifier is skipped and counted in `skipped_qualifiers`.
- A non-string, non-null qualifier on a relationship with a valid target raises a validation error.
- Duplicate `(source ID, qualifier, target ID)` rows are reported in `duplicate_relations` and **still counted**. Bounds therefore measure relationship rows, not distinct connected objects.
- Invalid object or relationship structures raise validation errors.

## Running the generator

The script requires Python 3.10 or newer and PyYAML. 

Run these commands from the repository root. To process logs in that directory:

```powershell
python container_logistics_constraints/generate_o2o_constraints.py
```

To specify the input and output directories:

```powershell
python container_logistics_constraints/generate_o2o_constraints.py --input-dir reference_logs --output-dir container_logistics_constraints
```

To replace existing output files:

```powershell
python container_logistics_constraints/generate_o2o_constraints.py --overwrite
```

| Option | Default | Behavior |
|---|---|---|
| `--input-dir` | Current working directory | Directory containing reference logs |
| `--output-dir` | `<input-dir>/container_logistics_constraints` | Directory for generated YAML files |
| `--overwrite` | Disabled | Replace existing YAML files instead of skipping them |

The input directory is searched directly, without recursion, for filenames matching `ContainerLogistics_<number>.json` or `ContainerLogistics_<number>.jsonocel`. Logs are processed in numeric order. Multiple files with the same numeric identifier cause an error.

Each log is processed independently; counts are not pooled across logs. For example:

```text
ContainerLogistics_1.json
  -> cardinality_constraints_ContainerLogistics_1.yaml
```

The output includes comments identifying the reference filename, explaining the bounds, and reporting excluded targets, blank qualifiers, and duplicate rows. If a valid log has no usable O2O relationships, the script writes an empty YAML list (`[]`).

## Note

**Use the original reference log before masking relations.** 

