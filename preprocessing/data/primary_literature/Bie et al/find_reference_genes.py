import pandas as pd
from pathlib import Path

# Use pathlib to construct the file path dynamically based on the current working directory
file_path = Path.cwd() / 'spectrum.00317-23-s0002.xlsx'

# Load the spreadsheet into a pandas DataFrame with corrected header row
data = pd.ExcelFile(file_path)
df_cleaned = pd.read_excel(file_path, sheet_name='1', header=1)

# Identify columns that contain "significant" and "padj" for filtering and ranking
significant_cols = [col for col in df_cleaned.columns if "significant" in col]
padj_cols = [col for col in df_cleaned.columns if "padj" in col]
log2fc_cols = [col for col in df_cleaned.columns if "log2FoldChange" in col]
fpkm_cols = [col for col in df_cleaned.columns if "fpkm" in col]

# Debug: Print a sample of significant columns to verify their content
print("Sample values from significant columns:")
print(df_cleaned[significant_cols].head())

# Ensure all relevant columns are present in the dataset
if not significant_cols or not padj_cols or not log2fc_cols:
    raise ValueError("One or more critical column groups (significant, padj, log2FoldChange) are missing from the dataset.")

# Essential columns that must always be retained
essential_cols = ['significant(KANvsH2O)', 'significant(CIPvsH2O)']
user_defined_threshold = 250  # Default threshold for the number of rows

# Attempt filtering and adjust if necessary
while True:
    # Normalize and handle NaN values in significant columns
    significant_filtered = df_cleaned[significant_cols].apply(
        lambda x: x.fillna('MISSING').replace(False, 'FALSE').astype(str).str.strip().str.upper()
    )

    # Check for "FALSE" values across the significant columns
    filtered_rows = significant_filtered.apply(lambda x: (x == 'FALSE').all(), axis=1)
    stable_genes = df_cleaned[filtered_rows]

    # Debug: Print the number of rows that pass the filter
    print(f"Number of rows passing the filter across all significant columns: {len(stable_genes)}")

    # If rows meet the user-defined threshold, break
    if len(stable_genes) >= user_defined_threshold:
        break

    # If no rows pass, iteratively remove a non-essential column
    if len(stable_genes) == 0 and len(significant_cols) > len(essential_cols):
        non_essential_cols = [col for col in significant_cols if col not in essential_cols]
        if non_essential_cols:
            removed_col = non_essential_cols[-1]
            significant_cols.remove(removed_col)
            print(f"Removed non-essential column: {removed_col}")
    else:
        print("No more non-essential columns to remove or essential columns remain only. Exiting loop.")
        break

# Debug: Print final set of columns used for filtering
print(f"Final set of columns used for filtering: {significant_cols}")

# Rank based on high padj values and low log2FoldChange values for CIP and KAN comparisons
ranking_cols = ['padj(CIPvsH2O)', 'log2FoldChange(CIPvsH2O)', 'padj(KANvsH2O)', 'log2FoldChange(KANvsH2O)']

# Ensure the necessary ranking columns are present
for col in ranking_cols:
    if col not in df_cleaned.columns:
        raise ValueError(f"The required ranking column {col} is missing from the dataset.")

stable_genes_sorted = stable_genes.sort_values(
    by=['padj(CIPvsH2O)', 'padj(KANvsH2O)', 'log2FoldChange(CIPvsH2O)', 'log2FoldChange(KANvsH2O)'],
    ascending=[False, False, True, True]
)

# Compute average FPKM across replicates for each condition
def compute_average_fpkm(data, prefix, replicates):
    """
    Compute the average FPKM across replicates for a given prefix.

    Parameters:
        data (DataFrame): The input DataFrame containing FPKM replicate columns.
        prefix (str): Prefix for the columns to average (e.g., "KAN", "CIP", "H2O").
        replicates (list): List of replicate column suffixes (e.g., ["_2_fpkm", "_3_fpkm", "_4_fpkm"]).

    Returns:
        Series: A pandas Series containing the averaged FPKM values.
    """
    columns = [f"{prefix}{rep}" for rep in replicates]
    valid_columns = [col for col in columns if col in data.columns]
    if not valid_columns:
        raise KeyError(f"No valid columns found for prefix '{prefix}' with replicates {replicates}")
    return data[valid_columns].mean(axis=1)

# Define prefixes and replicates
conditions = ['KAN', 'CIP', 'H2O']
replicates = ['_2_fpkm', '_3_fpkm', '_4_fpkm']

for condition in conditions:
    stable_genes_sorted[f"avg_fpkm_{condition}"] = compute_average_fpkm(stable_genes_sorted, condition, replicates)

# Rank by average FPKM in descending order
top_stable_genes = stable_genes_sorted.sort_values(
    by=[f"avg_fpkm_{cond}" for cond in conditions],
    ascending=False
)

# Summarize the ontology counts in a compact format
ontology_columns = [
    # 'biological_process',
    # 'biological_process_description',
    # 'cellular_component',
    # 'cellular_component_description',
    # 'molecular_function',
    # 'molecular_function_description'
    'Description'
]

# Ensure ontology columns exist in the dataset
missing_ontology_cols = [col for col in ontology_columns if col not in top_stable_genes.columns]
if missing_ontology_cols:
    raise ValueError(f"The following ontology columns are missing: {', '.join(missing_ontology_cols)}")

# Compact summary of ontology counts
ontology_summary = {
    col: top_stable_genes[col].value_counts().nlargest(5) for col in ontology_columns
}

# Output the top stable genes for review
print("Top Stable Genes for qPCR Reference:")
print(top_stable_genes[['Gene_id', 'Description'] + [f"avg_fpkm_{cond}" for cond in conditions]].head(20))  # Display only the first 20 rows for brevity

# Output summarized ontology counts for further analysis
print("\nSummarized Ontology Counts:")
for key, value in ontology_summary.items():
    print(f"\n{key}:")
    print(value)

# Save results to CSV for better usability
top_stable_genes.to_csv("top_stable_genes.csv", index=False)
print("Results saved to 'top_stable_genes.csv'")
