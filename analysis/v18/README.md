# Most recent analysis for the coded endocarditis manuscript

V18 is the analysis behind the current manuscript. It uses routine lab measurements available at two points during a hospital admission. Models were developed in MIMIC-III with patient-grouped folds and evaluated in eICU-CRD.

## Start here

- **[Study notebook](BE_ICD_LABS_v18.ipynb)** contains the cohort, features, models, and evaluation.
- **[Synthetic demo](#run-the-synthetic-demo)** runs the notebook’s lab cutoff and patient split functions without clinical data.
- **[Reproduction guide](reproducibility/README.md)** lists the input files, shows where to read the notebook, and links to the checks and run records.

## Main result

The selected XGBoost model reached **0.756 AUROC at the initial assessment** and **0.775 at repeat** in eICU. At the transferred thresholds, **1.02% and 0.64% of flagged admissions**, respectively, had the outcome code. When cases already coded by scoring were excluded, AUROC was 0.746 and 0.771. The code-entry audit measures documentation timing. The timing of clinical suspicion or workup remains unknown. [Aggregate results](reproducibility/eicu_later_coded_expected.json)

## Run the synthetic demo

From the repository root, install the demo dependencies and run the script.

```sh
python -m pip install numpy pandas scikit-learn
python analysis/v18/demo.py
```

The demo checks that a late result is withheld from both tabular and sequence inputs and that repeat admissions from the same patient stay together across all five folds and inner holdouts. [Demo source](demo.py)

## Contribution

Daniel Menzin built the v18 pipeline for the collaborative study, including cross-database cohort and lab alignment, patient-grouped evaluation, and the audit of diagnosis entry relative to scoring. [v18 analysis commit](https://github.com/SoppingBook/Precision-Medicine-BIOE5860/commit/e7295f2f21595a24755cf6b83ac249b4a9fd05f3)
