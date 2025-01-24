"""
This script takes in a CSV file containing ranked fold changes of genes and generates scatter plots.
The scatter plots highlight user-specified genes in blue (default genes: malK, cyoA, nuoA, rpsL, fliC) while all other genes are gray.
Top movers based on 'Effect_Size' (an average of FPKM_Difference and log2FoldChange ranks) are highlighted in red, and point size is scaled by 'FPKM_Difference'.
Separate plots are generated for Kanamycin and Ciprofloxacin data.

Functionality:
- Load a CSV file with gene fold changes.
- Plot fold changes as scatter plots.
- Highlight user-specified genes by color and annotate them.
- Highlight top movers and scale points by 'FPKM_Difference'.
- Save the plots to separate PDFs.

Dependencies:
- pandas: For data manipulation.
- matplotlib: For plotting.

Usage Example:
    python plot_script.py input_kan.csv input_cipro.csv output_kan.pdf output_cipro.pdf
"""
"""
This script takes in a CSV file containing ranked fold changes of genes and generates scatter plots.
The scatter plots highlight user-specified genes in blue (default genes: malK, cyoA, nuoA, rpsL, fliC) while all other genes are gray.
Top movers based on 'Effect_Size' are highlighted in red, and point size is scaled by 'FPKM_Difference'.
Separate plots are generated for Kanamycin and Ciprofloxacin data.

Functionality:
- Load a CSV file with gene fold changes.
- Plot fold changes as scatter plots.
- Highlight user-specified genes by color and annotate them.
- Highlight top movers and scale points by 'FPKM_Difference'.
- Save the plots to separate PDFs.

Dependencies:
- pandas: For data manipulation.
- matplotlib: For plotting.

Usage Example:
    python plot_script.py input_kan.csv input_cipro.csv output_kan.pdf output_cipro.pdf
"""

import pandas as pd
import matplotlib.pyplot as plt

def load_csv(file_path):
    """
    Load CSV data containing fold changes.

    Parameters:
        file_path (str): Path to the CSV file.

    Returns:
        DataFrame: Loaded data as a pandas DataFrame.
    """
    return pd.read_csv(file_path)

def get_top_movers(data, effect_size_col, top_movers=10):
    """
    Get the top movers based on 'Effect_Size'.

    Parameters:
        data (DataFrame): The input DataFrame containing gene data.
        effect_size_col (str): Column name for effect size.
        top_movers (int): Number of top movers to retrieve.

    Returns:
        DataFrame: Top movers sorted by effect size.
    """
    return data.nsmallest(top_movers, effect_size_col)

def compare_top_movers(kan_data, cipro_data, effect_size_col, top_movers=15):
    """
    Compare top movers between two datasets and print unique and shared movers.

    Parameters:
        kan_data (DataFrame): Kanamycin data.
        cipro_data (DataFrame): Ciprofloxacin data.
        effect_size_col (str): Column name for effect size.
        top_movers (int): Number of top movers to consider.
    """
    kan_top = get_top_movers(kan_data, effect_size_col, top_movers)
    cipro_top = get_top_movers(cipro_data, effect_size_col, top_movers)

    kan_genes = set(kan_top['Genename'])
    cipro_genes = set(cipro_top['Genename'])

    shared = kan_genes & cipro_genes
    unique_kan = kan_genes - cipro_genes
    unique_cipro = cipro_genes - kan_genes

    print("\nTop Movers Shared Between Kanamycin and Ciprofloxacin:")
    print(shared)

    print("\nUnique Top Movers for Kanamycin:")
    print(unique_kan)
    
    print("\nUnique Top Movers for Ciprofloxacin:")
    print(unique_cipro)

def plot_fold_changes(data, output_path, title, highlight_genes=None, fold_change_col=None, effect_size_col=None, fpkm_diff_col=None, top_movers=15):
    """
    Plot fold changes as a scatter plot, highlighting user-specified genes and top movers.

    Parameters:
        data (DataFrame): The input DataFrame containing gene data.
        output_path (str): Path to save the output PDF.
        title (str): Title of the plot.
        highlight_genes (list): List of gene names to highlight in blue.
        fold_change_col (str): Column name for fold changes.
        effect_size_col (str): Column name for effect size.
        fpkm_diff_col (str): Column name for FPKM difference (for scaling point sizes).
        top_movers (int): Number of top movers based on 'Effect_Size' to highlight in red.
    """
    if highlight_genes is None:
        highlight_genes = ["malK", "cyoA", "nuoA", "rpsL", "fliC"]

    # Make gene highlighting case-insensitive
    highlight_genes = [gene.lower() for gene in highlight_genes]

    if fold_change_col not in data.columns or effect_size_col not in data.columns or fpkm_diff_col not in data.columns:
        raise KeyError("Required columns not found in the dataset.")

    # Add a 'color' column to classify genes for coloring
    data['color'] = data['Genename'].str.lower().apply(
        lambda x: 'blue' if x in highlight_genes else 'gray'
    )

    # Highlight top movers in red
    top_movers_data = data.nsmallest(top_movers, effect_size_col)  # Fix: Use nsmallest for ascending rank
    data.loc[top_movers_data.index, 'color'] = 'red'

    # Scale point sizes based on FPKM difference
    data['PointSize'] = data[fpkm_diff_col].abs() / data[fpkm_diff_col].abs().max() * 150

    # Sort data by the fold change column in ascending order
    data = data.sort_values(by=fold_change_col, ascending=True).reset_index(drop=True)

    # Create the scatter plot
    plt.figure(figsize=(14, 6))

    # Add a horizontal dashed line at y=0 for reference
    plt.axhline(0, color="lightgray", linestyle="--", linewidth=1)

    # Initialize set for occupied positions to prevent overlap
    occupied_positions = set()

    def get_adjusted_position(x, y):
        # Adjust position to prevent overlaps
        offset = 0.1
        while (x, y) in occupied_positions:
            y += offset
        occupied_positions.add((x, y))
        return x, y

    for i, row in data.iterrows():
        plt.scatter(i, row[fold_change_col], color=row['color'], s=row['PointSize'], alpha=0.6)
        if row['color'] in ['blue', 'red']:
            x, y = get_adjusted_position(i, row[fold_change_col])
            plt.text(
                x=x, 
                y=y, 
                s=row['Genename'], 
                color='red' if row['color'] == 'red' else 'black', 
                fontsize=12, 
                fontweight='bold' if row['color'] == 'blue' else 'normal',
                ha='right', 
                va='center'
            )

    # Styling
    plt.title(title, fontsize=16)
    plt.xlabel("Genes", fontsize=12)
    plt.ylabel("Log2 Fold Change", fontsize=12)
    plt.xticks([], [])  # Remove x-ticks as there are too many genes
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)

    # Save plot to a PDF
    plt.savefig(output_path, format='pdf', bbox_inches='tight')
    plt.close()

def main(input_kan_csv, input_cipro_csv, output_kan_pdf, output_cipro_pdf):
    """
    Main function to generate scatter plots for Kanamycin and Ciprofloxacin.

    Parameters:
        input_kan_csv (str): Path to the input CSV for Kanamycin.
        input_cipro_csv (str): Path to the input CSV for Ciprofloxacin.
        output_kan_pdf (str): Path to save the Kanamycin plot.
        output_cipro_pdf (str): Path to save the Ciprofloxacin plot.
    """
    # Load Kanamycin data and generate plot
    kan_data = load_csv(input_kan_csv)
      
    plot_fold_changes(
        kan_data,
        output_kan_pdf,
        title="Ranked Fold Changes for Kanamycin (vs Water)",
        fold_change_col="log2FoldChange(KANvsH2O)",
        effect_size_col="Effect_Size",
        fpkm_diff_col="FPKM_Difference"
    )
    print(f"Plot for Kanamycin saved to {output_kan_pdf}")

    # Load Ciprofloxacin data and generate plot
    cipro_data = load_csv(input_cipro_csv)
    plot_fold_changes(
        cipro_data,
        output_cipro_pdf,
        title="Ranked Fold Changes for Ciprofloxacin (vs Water)",
        fold_change_col="log2FoldChange(CIPvsH2O)",
        effect_size_col="Effect_Size",
        fpkm_diff_col="FPKM_Difference"
    )
    print(f"Plot for Ciprofloxacin saved to {output_cipro_pdf}")

    compare_top_movers(kan_data, cipro_data, effect_size_col="Effect_Size", top_movers=20)

# Uncomment below to enable execution
if __name__ == "__main__":
    main(
        input_kan_csv="output_kanamycin.csv",
        input_cipro_csv="output_ciprofloxacin.csv",
        output_kan_pdf="kanamycin_plot.pdf",
        output_cipro_pdf="ciprofloxacin_plot.pdf"
    )
