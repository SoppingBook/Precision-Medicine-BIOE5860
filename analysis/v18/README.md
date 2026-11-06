# Most recent analysis for the coded endocarditis manuscript

V18 is the analysis behind the current manuscript. It asks whether routine lab measurements available at two points during a hospital admission can distinguish admissions with coded endocarditis. Models were developed in MIMIC-III with patient-grouped folds and evaluated in eICU-CRD. The outcome is a diagnosis code recorded during the hospitalization.

## Start here

- **[Study notebook](BE_ICD_LABS_v18.ipynb)** contains the cohort, features, models, and evaluation.
- **[Synthetic demo](#run-the-synthetic-demo)** runs the notebook’s lab cutoff and patient split functions without clinical data.
- **[Reproduction guide](reproducibility/README.md)** lists the input files, shows where to read the notebook, and links to the checks and run records.

## What we found

The selected XGBoost model used current and historical lab summaries with age and sex. Alert metrics use the Youden thresholds estimated in MIMIC and applied to eICU. In the broad eICU lab cohort:

- **Initial assessment:** 439 coded cases in 148,686 admissions (0.30%). AUROC was **0.756** and average precision was **0.040**. The transferred threshold flagged 15.0% of admissions, and 1.02% of those flags had the outcome code.
- **Repeat assessment:** 412 coded cases in 128,076 admissions (0.32%). AUROC was **0.775** and average precision was **0.032**. The transferred threshold flagged 37.5% of admissions, and 0.64% of those flags had the outcome code.

In a post hoc check excluding cases already coded at scoring and one case with indeterminate timing, AUROC was 0.746 initially and 0.771 at repeat. Average precision fell to 0.025 and 0.016. The score separated coded cases from controls, but transferred probabilities overestimated risk and most flags were code-negative. Diagnosis-entry time measures documentation. The timing of clinical suspicion or workup remains unknown. [Aggregate results](reproducibility/eicu_later_coded_expected.json)

## Run the synthetic demo

From the repository root, install the demo dependencies and run the script.

```sh
python -m pip install numpy pandas scikit-learn
python analysis/v18/demo.py
```

The expected output confirms that two lab values withheld at hour 6 enter the tabular and sequence inputs at hour 13. It also confirms that repeat admissions from the same patient stay together across all five folds and inner holdouts. [Demo source](demo.py)

## Contribution

Daniel Menzin built the v18 pipeline for the collaborative study, including cross-database cohort and lab alignment, patient-grouped evaluation, and the audit of diagnosis entry relative to scoring. [v18 analysis commit](https://github.com/SoppingBook/Precision-Medicine-BIOE5860/commit/e7295f2f21595a24755cf6b83ac249b4a9fd05f3)
