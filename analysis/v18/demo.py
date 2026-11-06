"""Exercise the v18 feature builder and patient splits on synthetic admissions."""

import ast
from collections import Counter
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, train_test_split


NOTEBOOK = Path(__file__).with_name("BE_ICD_LABS_v18.ipynb")


def load_study_functions():
    """Load the referenced definitions without running the study notebook."""
    notebook = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    cells = {cell["id"]: "".join(cell["source"]) for cell in notebook["cells"]}
    namespace = {
        "Counter": Counter,
        "np": np,
        "pd": pd,
        "StratifiedKFold": StratifiedKFold,
        "train_test_split": train_test_split,
    }
    definitions = {
        "v10-repaired-01": {"DEFAULT_CONFIG"},
        "v10-repaired-04": {"LABS", "ADULT_RANGES", "reference_interval", "boundary_features"},
        "v10-repaired-06": {"CORE", "resolve_revisions", "draw_landmarks", "series_stats"},
        "v10-repaired-12": {"patient_folds", "inner_split"},
        "v10-repaired-19": {"_build_features_batch"},
    }
    for cell_id, names in definitions.items():
        filename = f"{NOTEBOOK.name}:{cell_id}"
        statements = []
        for node in ast.parse(cells[cell_id], filename=filename).body:
            if isinstance(node, ast.FunctionDef) and node.name in names:
                statements.append(node)
            if isinstance(node, ast.Assign) and any(
                isinstance(target, ast.Name) and target.id in names for target in node.targets
            ):
                statements.append(node)
        exec(compile(ast.Module(body=statements, type_ignores=[]), filename, "exec"), namespace)
        missing = names - namespace.keys()
        if missing:
            raise RuntimeError(f"Missing notebook definitions in {cell_id}: {sorted(missing)}")
    return namespace


def check_lab_cutoffs(study):
    admission = pd.DataFrame([{
        "patient_id": "synthetic-1", "hospital_id": 1, "label": 0, "eligible": True,
        "valid_time": True, "duration": 24.0, "year": 2015, "age": 60,
        "sex": "F", "age_topcoded": False,
    }])
    labs = pd.DataFrame([
        ("sodium", 1.0, 5.0, 135.0),
        ("creatinine", 1.25, 6.0, 1.0),  # Available at scoring.
        ("hemoglobin", 1.5, 6.25, 8.0),  # Collected in the first window, reported late.
        ("crp", 2.0, 6.0, 4.0),  # Collected exactly at the end of the first window.
        ("sodium", 8.0, 12.0, 140.0),
        ("creatinine", 8.25, 12.25, 2.0),
    ], columns=["analyte", "time", "available", "value"])
    labs = labs.assign(patient_id="synthetic-1", hospital_id=1)
    context = ({1: labs}, {"synthetic-1": labs}, {"synthetic-1": admission})
    frame, sequence, _ = study["_build_features_batch"](
        admission, labs, "eICU", study["DEFAULT_CONFIG"], context,
    )
    scores = frame.set_index("landmark")
    assert scores.score_hour.to_dict() == {1: 6.0, 2: 13.0}
    assert scores.at[1, "current__creatinine__last"] == 1.0
    for analyte, value in (("hemoglobin", 8.0), ("crp", 4.0)):
        column = f"current__{analyte}__last"
        assert pd.isna(scores.at[1, column])
        assert scores.at[2, column] == value
        channel = list(study["LABS"]).index(analyte)
        assert np.isnan(sequence["values"][0, :, channel]).all()
        assert np.nanmax(sequence["values"][1, :, channel]) == value
    print("Timing: hour 6 accepts creatinine reported at scoring, withholds late hemoglobin and cutoff CRP.")
    print("Timing: both withheld values appear in the tabular and sequence inputs at hour 13.")


def check_patient_splits(study):
    admissions = pd.DataFrame([
        (f"synthetic-{patient:02d}", int(patient < 12 and visit == 1))
        for patient in range(24)
        for visit in range(2)
    ], columns=["patient_id", "label"])
    cfg = study["DEFAULT_CONFIG"]
    n_folds, seed = cfg["outer_folds"], cfg["seed"]
    folds = study["patient_folds"](admissions, n_folds, seed)
    assigned = admissions.patient_id.map(folds)
    assert len(folds) == admissions.patient_id.nunique() == 24
    assert assigned.notna().all()
    for fold in range(n_folds):
        train = admissions.loc[assigned.ne(fold)]
        test = admissions.loc[assigned.eq(fold)]
        assert train.label.nunique() == test.label.nunique() == 2
        assert set(train.patient_id).isdisjoint(test.patient_id)
        fit_rows, holdout_rows = study["inner_split"](train, seed + fold, cfg["inner_holdout"])
        fit_ids = set(train.iloc[fit_rows].patient_id)
        holdout_ids = set(train.iloc[holdout_rows].patient_id)
        assert fit_ids.isdisjoint(holdout_ids)
        assert fit_ids | holdout_ids == set(train.patient_id)
    print(f"Splits: 48 admissions from 24 patients pass all {n_folds} outer folds and inner holdouts.")


if __name__ == "__main__":
    functions = load_study_functions()
    check_lab_cutoffs(functions)
    check_patient_splits(functions)
