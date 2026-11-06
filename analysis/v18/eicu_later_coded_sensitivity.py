
"""Reproduce the manuscript's post hoc eICU diagnosis-entry sensitivity."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


QUALIFYING_CODE = r"(?:^|[,;\s])(?:421[019](?=$|[,;\s])|I33)"
PREDICTION_COLUMNS = [
    "hospital_id", "patient_id", "label", "model", "cohort", "probability", "threshold",
]
BOOTSTRAP_DRAWS = 500


def earliest_qualifying_entry(data_root):
    unit = pd.read_csv(data_root / "eicu" / "patients.csv", usecols=[
        "patientunitstayid", "patienthealthsystemstayid",
        "hospitaladmitoffset", "hospitaldischargeoffset",
    ])
    diagnoses = pd.read_csv(data_root / "eicu" / "diagnoses.csv", usecols=[
        "patientunitstayid", "icd9code", "diagnosisoffset",
    ], dtype={"icd9code": "string"})
    tokens = diagnoses.icd9code.fillna("").str.upper().str.replace(".", "", regex=False)
    qualifying = diagnoses.loc[tokens.str.contains(QUALIFYING_CODE)].merge(
        unit, on="patientunitstayid", validate="many_to_one",
    )
    qualifying["entry_minute"] = qualifying.diagnosisoffset - qualifying.hospitaladmitoffset
    valid = qualifying.entry_minute.ge(0) & qualifying.diagnosisoffset.le(
        qualifying.hospitaldischargeoffset,
    )
    return qualifying.loc[valid].groupby("patienthealthsystemstayid").entry_minute.min() / 60


def discrimination(pos, scores, patient_ids, landmark, bootstrap=False):
    order = np.argsort(-scores, kind="stable")
    pos_sorted = pos[order]
    sorted_scores = scores[order]
    # Group tied scores before calculating the ROC and average-precision steps.
    ends = np.flatnonzero(np.r_[sorted_scores[:-1] != sorted_scores[1:], True])

    def metrics(weight):
        tp = np.cumsum(weight * pos_sorted)[ends]
        fp = np.cumsum(weight * ~pos_sorted)[ends]
        if not tp[-1] or not fp[-1]:
            raise ValueError("Discrimination requires cases and controls")
        new_tp = np.diff(tp, prepend=0)
        new_fp = np.diff(fp, prepend=0)
        precision = np.divide(tp, tp + fp, out=np.zeros(len(tp)), where=tp + fp > 0)
        auroc = np.dot(new_fp, 2 * tp - new_tp) / (2 * tp[-1] * fp[-1])
        ap = np.dot(new_tp, precision) / tp[-1]
        return float(auroc), float(ap)

    auroc, ap = metrics(np.ones(len(pos), dtype=np.int64))
    result = {"auroc": auroc, "average_precision": ap}
    if bootstrap:
        _, clusters = np.unique(patient_ids, return_inverse=True)
        sorted_clusters = clusters[order]
        n_clusters = int(clusters.max()) + 1
        rng = np.random.default_rng(2460 + landmark)
        draws = []
        for _ in range(BOOTSTRAP_DRAWS):
            cluster_weights = np.bincount(
                rng.integers(n_clusters, size=n_clusters), minlength=n_clusters,
            )
            draws.append(metrics(cluster_weights[sorted_clusters]))
        lo, hi = np.quantile(draws, [.025, .975], axis=0)
        result["auroc_ci_95"] = [float(lo[0]), float(hi[0])]
        result["average_precision_ci_95"] = [float(lo[1]), float(hi[1])]
    return result


def summarize(frame, landmark, bootstrap=False):
    pos = frame.label.to_numpy() == 1
    scores = frame.probability.to_numpy()
    decisions = scores >= frame.threshold.to_numpy()
    prevalence = float(pos.mean())
    metrics = discrimination(pos, scores, frame.patient_id.to_numpy(), landmark, bootstrap)
    return {
        "admissions": len(frame),
        "cases": int(pos.sum()),
        "prevalence": prevalence,
        **metrics,
        "average_precision_lift": metrics["average_precision"] / prevalence,
        "recall": float(decisions[pos].mean()),
        "specificity": float((~decisions[~pos]).mean()),
        "ppv": float(pos[decisions].mean()) if decisions.any() else 0.0,
        "alert_fraction": float(decisions.mean()),
    }


def run(data_root, run_dir, feature_bank=None):
    manifest = json.loads((run_dir / "primary" / "run_manifest.json").read_text(encoding="utf-8"))
    checkpoint_name = manifest["checkpoint_directory"].replace("\\", "/").rstrip("/").split("/")[-1]
    checkpoints = run_dir / "primary" / "grid_checkpoints" / checkpoint_name
    if feature_bank is None:
        bank_key = manifest["lab_feature_fingerprint"][:16]
        feature_bank = run_dir / "recovered_features" / bank_key
        if not feature_bank.is_dir():
            feature_bank = run_dir / "primary" / "feature_cache" / bank_key
    scores = pd.read_csv(feature_bank / "eICU.csv.gz", usecols=[
        "hospital_id", "label", "landmark", "score_hour", "age",
    ])
    scores = scores.loc[scores.age.ge(18)]
    first_entry = earliest_qualifying_entry(data_root)
    results = []
    for landmark in (1, 2):
        config = f"all_labs|{landmark}|lab_current+lab_history|unweighted"
        token = hashlib.sha256(config.encode()).hexdigest()[:20]
        predictions = pd.read_csv(
            checkpoints / f"{token}.csv.gz", usecols=PREDICTION_COLUMNS,
            dtype={"patient_id": str}, float_precision="round_trip",
        )
        predictions = predictions.loc[
            predictions.model.eq("XGBoost") & predictions.cohort.eq("eICU_external")
        ]
        landmark_scores = scores.loc[
            scores.landmark.eq(landmark), ["hospital_id", "label", "score_hour"]
        ]
        frame = predictions.merge(
            landmark_scores, on=["hospital_id", "label"], how="outer",
            validate="one_to_one", indicator=True,
        )
        if not frame["_merge"].eq("both").all():
            raise ValueError(f"Prediction and score-time cohorts disagree at landmark {landmark}")
        if not np.isfinite(frame[["probability", "threshold", "score_hour"]]).all().all():
            raise ValueError(f"Nonfinite prediction, threshold or score time at landmark {landmark}")
        case = frame.label.eq(1)
        entry = frame.hospital_id.map(first_entry)
        if entry.loc[~case].notna().any():
            raise ValueError(f"A code-negative admission has a qualifying code at landmark {landmark}")
        already_coded = case & entry.le(frame.score_hour)
        valid_case = case & entry.notna()
        later_coded = frame.loc[~case | entry.gt(frame.score_hour)]
        results.append({
            "assessment": "initial" if landmark == 1 else "repeat",
            "cases_with_valid_entry_time": int(valid_case.sum()),
            "cases_coded_by_score_time": int(already_coded.sum()),
            "cases_with_indeterminate_time": int((case & entry.isna()).sum()),
            "full": summarize(frame, landmark),
            "later_coded": summarize(later_coded, landmark, bootstrap=True),
        })
    return {"bootstrap_draws": BOOTSTRAP_DRAWS, "results": results}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--feature-bank", type=Path, help="Override the bank identified by the primary manifest.")
    parser.add_argument("--output", type=Path, help="Write aggregate JSON to a new file instead of stdout.")
    args = parser.parse_args()
    result = json.dumps(run(args.data_root, args.run_dir, args.feature_bank), indent=2, allow_nan=False)
    if args.output:
        with args.output.open("x", encoding="utf-8") as stream:
            stream.write(result + "\n")
    else:
        print(result)


if __name__ == "__main__":
    main()
