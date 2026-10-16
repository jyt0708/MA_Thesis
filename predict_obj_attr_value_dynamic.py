"""Run object-attribute value prediction with dynamically detected targets.

This variant reuses the experiment implementation in
``predict_obj_attr_value.py`` while removing its hard-coded ``TARGET_TYPES``.
"""

from pathlib import Path

import numpy as np
import pandas as pd

import predict_obj_attr_value as base


OUTPUT_DIR = Path("Manufacturing/results_object_attr_values")

# Keep this reference before main() replaces base.make_model.
ORIGINAL_MAKE_MODEL = base.make_model


class SingleRowSafeModel:
    """Work around SAP RPT's scalar-output failure for one-row queries."""

    def __init__(self, model):
        self.model = model

    def fit(self, X, y):
        self.model.fit(X, y)
        return self

    def predict(self, X):
        if len(X) != 1:
            return self.model.predict(X)

        duplicated = pd.concat([X, X], axis=0, ignore_index=True)
        predictions = np.asarray(self.model.predict(duplicated))
        return predictions[:1]


def make_model(task_type):
    return SingleRowSafeModel(ORIGINAL_MAKE_MODEL(task_type))


def find_target_columns(df):
    """Return every obj_attr__ value column, excluding existence flags."""
    return [
        column
        for column in df.columns
        if column.startswith("obj_attr__")
        and not column.endswith(("__exists", "__exits"))
    ]


def detect_target_type(values, target):
    """Infer whether a target should use regression or classification."""
    if "date" in target.casefold():
        return "regression"

    observed = values.dropna()
    if not observed.empty:
        numeric = pd.to_numeric(observed, errors="coerce")
        if numeric.notna().all():
            return "regression"

    return "classification"


def numeric_regression_target(values, target):
    """Convert numeric targets, expressing date targets as days since epoch."""
    if "date" in target.casefold():
        datetimes = pd.to_datetime(
            values,
            dayfirst=True,
            errors="raise",
        )
        return (
            datetimes - pd.Timestamp("1970-01-01")
        ).dt.total_seconds() / 86_400

    return pd.to_numeric(values, errors="raise")


def paired_columns(target):
    """Return the possible existence/applicability flags for a target."""
    return f"{target}__exists", f"{target}__applicable"


def main():
    files = base.discover_files()

    if not files:
        raise FileNotFoundError(
            f"No files matching {base.INPUT_PATTERNS!r} found in "
            f"{base.INPUT_DIR.expanduser().resolve()}."
        )

    # Patch the reusable experiment helpers with the dynamic behavior above.
    base.OUTPUT_DIR = OUTPUT_DIR
    base.numeric_regression_target = numeric_regression_target
    base.paired_columns = paired_columns
    base.make_model = make_model

    results = []

    for file_path in files:
        print(f"\n=== {file_path.name} ===")
        df = base.read_csv_auto(file_path)
        targets = find_target_columns(df)

        target_types = {
            target: detect_target_type(df[target], target)
            for target in targets
        }
        print("Targets:", target_types)

        # run_one reads this mapping, so update it for the current input file.
        base.TARGET_TYPES = target_types

        for target, task_type in target_types.items():
            for seed in base.SEEDS:
                print(f"  {target} ({task_type}) | seed={seed}")
                result = base.run_one(
                    df,
                    file_path.name,
                    target,
                    seed,
                )
                if result is not None:
                    results.append(result)

    if not results:
        print("No valid experiments were run.")
        return

    result_df, per_file, overall = base.save_results(results)

    print("\n=== Per-run metrics ===")
    print(result_df.to_string(index=False))
    print("\n=== Per-file mean over 3 seeds ===")
    print(per_file.to_string(index=False))
    print("\n=== Overall mean +/- SD across files ===")
    print(overall)
    print(f"\nSaved to: {OUTPUT_DIR.resolve()}")


if __name__ == "__main__":
    main()
