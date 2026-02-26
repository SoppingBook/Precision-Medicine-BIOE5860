# Pre_Processing Notebook - MIMIC-III Aortic Dissection Analysis

## Overview

Yo choom, this notebook's all about prepping MIMIC-III data for analyzing patients with aortic dissection against control groups. It's doing the heavy lifting of merging patient admissions, diagnoses, procedures, and lab events into clean, ready-to-analyze datasets.

## What This Code Does

### Main Workflow

1. **Loads Core MIMIC-III Data**
   - ADMISSIONS.csv
   - DIAGNOSES_ICD.csv  
   - PATIENTS.csv
   - PROCEDURES_ICD.csv
   - LABEVENTS.csv (chunked processing for memory efficiency)

2. **Identifies Patient and Control Groups**
   - **Patient Group**: Anyone diagnosed with ICD-9 codes 44100, 44101, 44102, or 44103 (aortic dissection variants)
   - **Additional Filtering**: Also catches patients with "AORTIC DISSECTION" in admission diagnosis text
   - **Control Group**: All patients NOT in either of the above categories

3. **Data Compression**
   - Consolidates multiple ICD-9 diagnosis codes per admission into single list entries
   - Consolidates procedure codes per admission into single list entries
   - Keeps data sorted by SEQ_NUM for clinical relevance

4. **Temporal Filtering for Patient Group**
   - Only includes admissions up to and including the first admission where aortic dissection was diagnosed
   - Creates `PATIENT_ADMISSION_INDEX` showing countdown to diagnosis (0 = diagnosis admission)
   - Excludes subsequent admissions to prevent outcome contamination

5. **Lab Events Processing**
   - Filters for cardiac and inflammatory markers: troponin, D-dimer, creatinine, CK-MB, BUN, urea, C-reactive protein, LDH, bilirubin, AST, ALT, WBC counts, lymphocytes, neutrophils
   - Processes LABEVENTS in 50,000-row chunks to manage memory
   - Links lab events to admissions for both groups

## Key Variables to Modify

```python
my_icd9_code = ["44100", "44101", "44102", "44103"]  # Change for different conditions
```

For bacterial endocarditis, switch to `["421"]` and update the following code accordingly.

## Output Files

Two primary merged datasets are saved:

- `PATIENT_ADMISSIONS_MERGED.csv` - (378 rows × 21 columns)
- `CONTROL_ADMISSIONS_MERGED.csv` - (58,440 rows × 21 columns)

### Output Columns

Both datasets contain:
- Patient demographics (SUBJECT_ID, admission details)
- Temporal data (ADMITTIME, DISCHTIME, DEATHTIME)
- Clinical classifications (ADMISSION_TYPE, INSURANCE, ETHNICITY, etc.)
- Compressed diagnosis lists (DIAGNOSIS (ICD_9))
- Compressed procedure lists (PROCEDURE TYPE)
- Index variable (PATIENT_ADMISSION_INDEX or ADMISSION_INDEX_PER_PATIENT)

## Memory Management Notes

The notebook includes strategic `del` statements (currently commented) to free memory after filtering:
- DIAGNOSES_ICD
- PROCEDURES_ICD  
- LAB_EVENTS

Uncomment these if running into memory constraints.

## Data Quality Checks

The notebook identifies 17 patients who have "AORTIC DISSECTION" in their admission text but lack the formal ICD-9 codes. These are excluded from the control group to prevent contamination.

## Dependencies

```python
import numpy as np
import pandas as pd
import seaborn as sns
import datetime
import matplotlib.pyplot as plt
```

## Usage Notes

1. Update file paths to match your MIMIC-III data location
2. Modify `my_icd9_code` for different disease conditions
3. Adjust `labs_of_interest` list to focus on different biomarkers
4. Run cells sequentially - the workflow depends on proper ordering

## Function Reference

### `apply_event_index_filter(PATIENT_ADMISSIONS_MERGED, CONTROL_ADMISSIONS_MERGED, AD_HADM_ID)`

Creates the temporal index for patient admissions, counting backwards from first diagnosis.

**Returns**: Filtered patient and control admission dataframes with index columns

## Data Validation

Final output check shows:
- 0 patients with "AORTIC DISSECTION" text remain in control group
- 17 patients caught by text-based filtering not in ICD-9 codes
- Patient cohort: 378 admissions
- Control cohort: 58,440 admissions