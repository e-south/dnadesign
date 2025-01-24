"""
This script filters downregulated genes for specific antibiotics and outputs corresponding CSV files.

Functionality:
- Load transcriptomic data from an Excel file.
- Compute average FPKM values across replicates for antibiotics and control conditions.
- Filter and rank downregulated genes for a specific antibiotic class based on log2 fold change and significance.
- Calculate FPKM differences (antibiotic vs. control) and rank genes by both FPKM control expression and log2 fold change.
- Combine ranks to prioritize "big movers" with high control expression and strong downregulation.
- Save the processed data, including calculated metrics, to a CSV file.

Dependencies:
- pandas: For data manipulation and analysis.

"""

import pandas as pd

def load_data(file_path, sheet_name):
    """
    Load Excel data from the specified file and sheet.

    Parameters:
        file_path (str): Path to the Excel file.
        sheet_name (str): Name of the sheet to load.

    Returns:
        DataFrame: Loaded data as a pandas DataFrame.
    """
    return pd.read_excel(file_path, sheet_name=sheet_name, header=1)

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

def filter_downregulated_genes(data, log2_fc_col, significant_col, abx_fpkm_col, h2o_fpkm_col):
    """
    Filter and rank downregulated genes based on log2 fold change, significance, and effect size.

    Parameters:
        data (DataFrame): The input DataFrame containing gene data.
        log2_fc_col (str): The column name for log2 fold change values.
        significant_col (str): The column name for significance status.
        abx_fpkm_col (str): The column name for averaged antibiotic FPKM values.
        h2o_fpkm_col (str): The column name for averaged water FPKM values.

    Returns:
        DataFrame: Filtered and sorted DataFrame of downregulated genes, retaining relevant columns.
    """
    # Filter rows where log2 fold change is negative and significance is marked as 'DOWN'
    filtered = data[
        (data[log2_fc_col] < 0) & (data[significant_col] == 'DOWN')
    ][[
        'Gene_id', 'Genename', log2_fc_col, significant_col, abx_fpkm_col, h2o_fpkm_col,
        'biological_process', 'biological_process_description',
        'cellular_component', 'cellular_component_description',
        'molecular_function', 'molecular_function_description',
        'KEGG AnnotInfo', 'KEGG url', 'Description'
    ]]

    # Calculate FPKM difference as the difference between antibiotic and water FPKM averages
    filtered['FPKM_Difference'] = filtered[abx_fpkm_col] - filtered[h2o_fpkm_col]

    # Rank genes by control FPKM (water) and fold change
    filtered['Rank_FPKM'] = filtered[h2o_fpkm_col].rank(ascending=False)
    filtered['Rank_Log2FC'] = filtered[log2_fc_col].rank(ascending=True)

    # Combine ranks to calculate effect size
    filtered['Effect_Size'] = filtered['Rank_FPKM'] + filtered['Rank_Log2FC']

    # Sort by effect size (lower is better)
    return filtered.sort_values(by='Effect_Size', ascending=True)

def save_to_csv(data, output_path):
    """
    Save the provided DataFrame to a CSV file.

    Parameters:
        data (DataFrame): The DataFrame to save.
        output_path (str): Path to save the CSV file.
    """
    # Export the DataFrame to a CSV file, excluding the index column
    data.to_csv(output_path, index=False)

def main(input_file, output_kan_file, output_cipro_file):
    """
    Main function to process the input data and generate outputs for Kanamycin and Ciprofloxacin.

    Parameters:
        input_file (str): Path to the input Excel file.
        output_kan_file (str): Path to save the filtered Kanamycin CSV.
        output_cipro_file (str): Path to save the filtered Ciprofloxacin CSV.
    """
    # Load the dataset
    data = load_data(input_file, sheet_name="1")

    # Compute average FPKM values for replicates
    replicates = ["_2_fpkm", "_3_fpkm", "_4_fpkm"]
    data['KAN_avg_fpkm'] = compute_average_fpkm(data, "KAN", replicates)
    data['CIP_avg_fpkm'] = compute_average_fpkm(data, "CIP", replicates)
    data['H2O_avg_fpkm'] = compute_average_fpkm(data, "H2O", replicates)

    # Filter for Kanamycin
    kanamycin_data = filter_downregulated_genes(
        data,
        log2_fc_col="log2FoldChange(KANvsH2O)",
        significant_col="significant(KANvsH2O)",
        abx_fpkm_col="KAN_avg_fpkm",
        h2o_fpkm_col="H2O_avg_fpkm"
    )
    save_to_csv(kanamycin_data, output_kan_file)
    print(f"Filtered data for Kanamycin saved to {output_kan_file}")

    # Filter for Ciprofloxacin
    ciprofloxacin_data = filter_downregulated_genes(
        data,
        log2_fc_col="log2FoldChange(CIPvsH2O)",
        significant_col="significant(CIPvsH2O)",
        abx_fpkm_col="CIP_avg_fpkm",
        h2o_fpkm_col="H2O_avg_fpkm"
    )
    save_to_csv(ciprofloxacin_data, output_cipro_file)
    print(f"Filtered data for Ciprofloxacin saved to {output_cipro_file}")

# Uncomment below to enable execution
if __name__ == "__main__":
    main(
        input_file="spectrum.00317-23-s0002.xlsx",
        output_kan_file="output_kanamycin.csv",
        output_cipro_file="output_ciprofloxacin.csv"
    )
