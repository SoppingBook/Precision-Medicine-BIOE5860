#!/usr/bin/env python3
"""Lab ID to name conversion with duplicate handling"""

import pandas as pd
import re
import os

print(f"Working directory: {os.getcwd()}")
print()

# Load mapping
print("Loading lab mapping...")
lab_mapping = pd.read_csv("Lab_Item_Codes.txt", sep="\t")
code_to_name = {str(code): name for code, name in zip(lab_mapping['Code'], lab_mapping['Display'])}
print(f"[OK] Loaded {len(code_to_name)} lab mappings")

def rename_labs(df):
    """Rename lab columns from numeric ID to lab name, handling duplicates"""
    rename_map = {}
    seen_names = {}
    
    for col in df.columns:
        m = re.match(r'LAB_(\d+)_(mean|min|max|last)', col)
        if m:
            code, stat = m.groups()
            if code in code_to_name:
                lab_name = code_to_name[code].replace(' ', '_').replace(',', '').replace('(', '').replace(')', '')
                new_col = f"LAB_{lab_name}_{stat}"
                
                # Handle duplicates by appending the code
                if new_col in seen_names:
                    new_col = f"LAB_{lab_name}_{code}_{stat}"
                
                seen_names[new_col] = True
                rename_map[col] = new_col
    
    print(f"  Found {len(rename_map)} lab columns to rename")
    return df.rename(columns=rename_map), len(rename_map)

# Process main matrix
print("\n1. Processing MODEL_READY_MATRIX.parquet...")
df1 = pd.read_parquet("MODEL_READY_MATRIX.parquet")
print(f"   Loaded: {df1.shape}")

df1_renamed, count1 = rename_labs(df1)
print(f"  Writing file...")
df1_renamed.to_parquet("MODEL_READY_MATRIX_NAMED_LABS.parquet", compression='snappy')
print(f"[DONE] Created: MODEL_READY_MATRIX_NAMED_LABS.parquet ({count1} renamed)")
if os.path.exists("MODEL_READY_MATRIX_NAMED_LABS.parquet"):
    size = os.path.getsize("MODEL_READY_MATRIX_NAMED_LABS.parquet") / (1024**2)
    print(f"  Size: {size:.2f} MB")

# Process balanced matrix
print("\n2. Processing MODEL_READY_MATRIX_BALANCED.parquet...")
df2 = pd.read_parquet("MODEL_READY_MATRIX_BALANCED.parquet")
print(f"   Loaded: {df2.shape}")

df2_renamed, count2 = rename_labs(df2)
print(f"  Writing file...")
df2_renamed.to_parquet("MODEL_READY_MATRIX_BALANCED_NAMED_LABS.parquet", compression='snappy')
print(f"[DONE] Created: MODEL_READY_MATRIX_BALANCED_NAMED_LABS.parquet ({count2} renamed)")
if os.path.exists("MODEL_READY_MATRIX_BALANCED_NAMED_LABS.parquet"):
    size = os.path.getsize("MODEL_READY_MATRIX_BALANCED_NAMED_LABS.parquet") / (1024**2)
    print(f"  Size: {size:.2f} MB")

print("\n[COMPLETE] Both files created successfully!")
