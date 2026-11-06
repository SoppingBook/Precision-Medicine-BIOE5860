# Coded endocarditis analysis

The study asks whether routine laboratory measurements available early in a hospital admission distinguish admissions with coded endocarditis. Models are developed in MIMIC-III and evaluated in eICU-CRD. The broad laboratory cohort allows optional prior history. Diagnosis-history analyses use a separate subgroup with usable prior ICD codes.

## How this relates to the earlier work

- **Cohort and outcome.** The [April ICD notebook](../../Daniel_BE_ICD_Model.ipynb) built a MIMIC cohort around ICD-9 421.0 and sampled controls by admission-history counts. V18 evaluates eligible admissions at their actual index hospitalizations and uses ICD-9 421.0, 421.1 and 421.9, with I33-family codes also accepted in eICU. The endpoint is coded endocarditis.
- **Predictors and timing.** The earlier notebooks developed temporal diagnosis and procedure features, demographic inputs, and lab preprocessing. V18 uses age and sex, current labs available at each of two one-hour collection landmarks, optional prior lab summaries, and historical ICD predictors. The primary timing model assumes a four-hour reporting lag and excludes same-year separate eICU stays from prior history. The eICU history code path includes an ICD-10-to-ICD-9 crosswalk and coverage checks.
- **Training and evaluation.** The April model used fold-level chi-squared selection, case oversampling for some models, and MIMIC cross-validation. V18 uses patient-grouped MIMIC folds, training-partition preprocessing and selection, and full-MIMIC fits for cross-database evaluation. The tabular grid compares predictor blocks, unweighted and balanced loss, and a separate prevalence-matched arm that duplicates MIMIC training controls using the aggregate eICU case rate. Both evaluation cohorts keep their natural prevalence. Separate lab CNNs test whether time-binned sequences add value.
- **Interpretation.** V18 reports AUROC, average precision, calibration, and alert burden. The cohorts, outcomes and predictors changed, so metrics across notebook generations answer different questions. The post hoc eICU timing check measures performance after excluding admissions whose qualifying diagnosis code was already entered at scoring. Code-entry timing does not establish when clinical suspicion or workup began.

## Run the analysis

1. Obtain credentialed access to [MIMIC-III](https://physionet.org/content/mimiciii/) and [eICU-CRD](https://physionet.org/content/eicu-crd/). Store the extracts outside the Git working tree. The exact analyzed database releases remain pending confirmation.
2. In the data directory, place MIMIC `ADMISSIONS.csv`, `PATIENTS.csv`, `DIAGNOSES_ICD.csv` and `LABEVENTS.csv`. Place eICU `patients.csv`, `diagnoses.csv` and `lab.csv` under `eicu/`. The analysis also expects `CMS32_DESC_LONG_SHORT_DX.xlsx`, the [CMS Version 32 diagnosis dictionary](https://www.cms.gov/medicare/coding-billing/icd-10-codes/icd-9-cm-diagnosis-procedure-codes-abbreviated-and-full-code-titles).
3. Set `BE_DATA_ROOT` to that directory before launching Jupyter. Use [the reference dependencies](requirements-reference.txt) with the recorded Python 3.14.4 environment. The mapper listed there reproduced all saved token decisions, and the original mapper release remains unverified.
4. Run [the v18 notebook](BE_ICD_LABS_v18.ipynb). Its default run name is `BE_integrated_grid_v18_public_rebuild_outputs`, and its default execution builds features and fits models from local source data. A full run includes the tabular grid, CNN comparisons, and timing, age and lab-panel sensitivities.

## Reproduce the manuscript checks

- [Later-coded eICU sensitivity](eicu_later_coded_sensitivity.py) reads the primary run manifest, saved prediction vectors, and feature-bank score times. Run it with `--data-root` and `--run-dir`. Use `--feature-bank` to supply an alternate local bank. [Expected aggregate results](eicu_later_coded_expected.json) record the counts, metrics and patient-cluster intervals from the manuscript run.
- [Category coverage](coverage/compute_evidence.py) and [selected-feature coverage](coverage/selected_portability.py) read the local primary fit audit and feature bank. Set `BE_DATA_ROOT`. For a different run or bank, set `BE_RUN_DIR` and `BE_FEATURE_BANK` to their local paths.

## Run provenance

The manuscript’s primary and CNN results used audited recovery of saved stages. See the [source checksums and recovery record](provenance/analysis_provenance.json), [earlier modeling source](provenance/v15_model_source.ipynb), and [12-hour execution cells](provenance/fast12h_execution_cells.ipynb). The 12-hour cells document the cache-backed execution used for that sensitivity and depend on the original local checkpoints. The main notebook uses the standard builder for an independent run. Exact reproduction from a fresh build has not been established for the full recovered feature bank.

The repository contains source and aggregate verification results. Credentialed extracts, prediction vectors, feature banks and model checkpoints stay in local authorized storage.
