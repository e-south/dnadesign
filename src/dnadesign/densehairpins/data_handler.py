"""
--------------------------------------------------------------------------------
<dnadesign project>
/densehairpins/data_handler.py

Module Author(s): Eric J. South
Dunlop Lab
--------------------------------------------------------------------------------
"""

import pandas as pd
from pathlib import Path
from dnadesign.utils import DATA_FILES  # Assumes utils.py is in dnadesign/

def load_pancardo_dataset():
    """
    Loads the Pancardo et al. dataset from the provided DATA_FILES mapping.
    Expects an Excel file with sheet "Table 2" and columns: 'TF', 'Silenced Genes', 'Induced Genes'.
    Adds a 'Rank' column based on row order.
    """
    file_path = DATA_FILES["pancardo_et_al"]
    if not file_path.exists():
        raise FileNotFoundError(f"Pancardo dataset file not found: {file_path}")
    df = pd.read_excel(file_path, sheet_name='Table 2', header=0)
    df = df[['TF', 'Silenced Genes', 'Induced Genes']].copy()
    df['Rank'] = range(1, len(df) + 1)
    return df

def save_intermediate_csv(data, output_csv):
    """
    Saves a list of dictionaries (intermediate data) as a CSV file.
    """
    df = pd.DataFrame(data)
    df.to_csv(output_csv, index=False)

