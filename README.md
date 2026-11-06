# Precision Medicine BIOE 5860

This repository contains the original MIMIC-III course experiments and a follow-up study of coded endocarditis using MIMIC-III and eICU-CRD.

## Manuscript analysis

[Start with the v18 analysis](analysis/v18/README.md) for the study design, input files, execution instructions, and the changes from the original notebooks. The [v18 notebook](analysis/v18/BE_ICD_LABS_v18.ipynb) is the entry point for a new run with credentialed data.

## Original course experiments

- [Balanced preprocessing](Balanced_Pre_Processing.ipynb) and [updated preprocessing](Pre_Processing_Updated.ipynb) build the original feature matrices.
- [Control selection comparisons](Models_Control_Selection_Comparison.ipynb) and [model helpers](Models_for_Control.py) document the control-sampling experiments.
- [Daniel’s April ICD model](Daniel_BE_ICD_Model.ipynb) is the closest predecessor to v18. It includes the original MIMIC cohort, fold-level modeling, and SHAP analysis.
