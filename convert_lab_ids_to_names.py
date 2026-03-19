import pandas as pd
import re

# Load the lab ID to name mapping
lab_mapping = pd.read_csv(r"Lab_Item_Codes.txt", sep="\t")
# Create a dictionary: Code (int) -> Display (name)
code_to_name = dict(zip(lab_mapping['Code'].astype(str), lab_mapping['Display']))

print(f"Loaded {len(code_to_name)} lab code mappings")
print("Sample mappings:")
for code, name in list(code_to_name.items())[:5]:
    print(f"  {code}: {name}")

def rename_lab_columns(df, code_to_name):
    """
    Rename columns with pattern LAB_<code>_<stat> to LAB_<name>_<stat>
    where code is the ITEMID and name is the lab name.
    """
    new_columns = {}
    
    for col in df.columns:
        # Match columns like LAB_50801_mean, LAB_50801_min, etc.
        match = re.match(r'LAB_(\d+)_(mean|min|max|last)', col)
        if match:
            itemid = match.group(1)
            stat = match.group(2)
            
            if itemid in code_to_name:
                # Replace underscores in lab name with spaces for readability
                lab_name = code_to_name[itemid]
                # Clean up the lab name for use as a column name
                lab_name_clean = lab_name.replace(' ', '_').replace(',', '').replace('(', '').replace(')', '')
                new_col = f"LAB_{lab_name_clean}_{stat}"
                new_columns[col] = new_col
                print(f"  {col} -> {new_col}")
            else:
                print(f"  WARNING: No mapping found for ITEMID {itemid}, keeping column as is: {col}")
        else:
            # Keep non-lab columns unchanged
            pass
    
    # Rename the columns
    df_renamed = df.rename(columns=new_columns)
    return df_renamed

# Process the main final matrix
print("\n" + "="*80)
print("Processing MODEL_READY_MATRIX.parquet...")
print("="*80)

try:
    final_matrix = pd.read_parquet(r"MODEL_READY_MATRIX.parquet")
    print(f"Loaded final matrix: shape {final_matrix.shape}")
    
    # Show original lab columns
    lab_cols = [c for c in final_matrix.columns if c.startswith('LAB_')]
    print(f"\nOriginal lab columns ({len(lab_cols)} total):")
    for col in sorted(lab_cols)[:10]:
        print(f"  {col}")
    if len(lab_cols) > 10:
        print(f"  ... and {len(lab_cols) - 10} more")
    
    # Rename columns
    print("\nRenaming lab columns...")
    final_matrix_renamed = rename_lab_columns(final_matrix, code_to_name)
    
    # Verify renamed columns
    lab_cols_renamed = [c for c in final_matrix_renamed.columns if c.startswith('LAB_')]
    print(f"\nRenamed lab columns ({len(lab_cols_renamed)} total):")
    for col in sorted(lab_cols_renamed)[:10]:
        print(f"  {col}")
    if len(lab_cols_renamed) > 10:
        print(f"  ... and {len(lab_cols_renamed) - 10} more")
    
    # Save with new name
    output_file = r"MODEL_READY_MATRIX_NAMED_LABS.parquet"
    final_matrix_renamed.to_parquet(output_file, index=False)
    print(f"\nSaved renamed matrix to: {output_file}")
    print(f"  Shape: {final_matrix_renamed.shape}")
    
except FileNotFoundError:
    print("ERROR: MODEL_READY_MATRIX.parquet not found")

# Process the balanced matrix
print("\n" + "="*80)
print("Processing MODEL_READY_MATRIX_BALANCED.parquet...")
print("="*80)

try:
    balanced_matrix = pd.read_parquet(r"MODEL_READY_MATRIX_BALANCED.parquet")
    print(f"Loaded balanced matrix: shape {balanced_matrix.shape}")
    
    # Show original lab columns
    lab_cols = [c for c in balanced_matrix.columns if c.startswith('LAB_')]
    print(f"\nOriginal lab columns ({len(lab_cols)} total):")
    for col in sorted(lab_cols)[:10]:
        print(f"  {col}")
    if len(lab_cols) > 10:
        print(f"  ... and {len(lab_cols) - 10} more")
    
    # Rename columns
    print("\nRenaming lab columns...")
    balanced_matrix_renamed = rename_lab_columns(balanced_matrix, code_to_name)
    
    # Verify renamed columns
    lab_cols_renamed = [c for c in balanced_matrix_renamed.columns if c.startswith('LAB_')]
    print(f"\nRenamed lab columns ({len(lab_cols_renamed)} total):")
    for col in sorted(lab_cols_renamed)[:10]:
        print(f"  {col}")
    if len(lab_cols_renamed) > 10:
        print(f"  ... and {len(lab_cols_renamed) - 10} more")
    
    # Save with new name
    output_file = r"MODEL_READY_MATRIX_BALANCED_NAMED_LABS.parquet"
    balanced_matrix_renamed.to_parquet(output_file, index=False)
    print(f"\nSaved renamed matrix to: {output_file}")
    print(f"  Shape: {balanced_matrix_renamed.shape}")
    
except FileNotFoundError:
    print("ERROR: MODEL_READY_MATRIX_BALANCED.parquet not found")

print("\n" + "="*80)
print("SUMMARY")
print("="*80)
print("""
The following files have been created with lab names instead of IDs:
  - MODEL_READY_MATRIX_NAMED_LABS.parquet (main matrix)
  - MODEL_READY_MATRIX_BALANCED_NAMED_LABS.parquet (balanced matrix)

Column naming convention:
  LAB_<lab_name>_<statistic>
  
  where <statistic> can be: mean, min, max, last
  
Example: 
  LAB_50809_mean  ->  LAB_Glucose_mean
  LAB_50811_max   ->  LAB_Hemoglobin_max
""")
