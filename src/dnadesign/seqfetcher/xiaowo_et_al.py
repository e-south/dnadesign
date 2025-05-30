"""
--------------------------------------------------------------------------------
<dnadesign project>
seqfetcher/xiaowo_et_al.py

Module for loading data described in Xiaowo et al., which generated 24,000 synthetic 
regulatory elements and characterized them in vivo in E. coli and P. aeruginosa as 
part of developing their DeepCROSS model.

"Systematic representation and optimization enable the inverse design of cross-species 
regulatory sequences in bacteria"
DOI: 10.1038/s41467-025-57031-1

Processing:
  - For each sheet ('Lib2-EC' and 'Lib1-EC'):
      • Loads the sheet (header=0) using load_dataset().
      • Renames the first (unnamed) column to "name" and prepends the sheet name 
        to each value (e.g., "Lib2-EC-1").
      • Renames the "Sequence" column to "sequence"; all other columns are prefixed 
        with "meta_".
      • Drops any rows with null values or any cell equal to 0.
  - Concatenates data from both sheets into a tidy DataFrame.
  - Converts each row into a dictionary with only the keys:
      - id
      - name
      - sequence
      - meta_date_accessed
      - meta_source
      - meta_part_type (set to "engineered promoter")
      - meta_exp_mean(log2)
      - meta_tags
  - Saves the list of dictionaries to a dedicated subfolder in sequences/ using 
    the SequenceSaver utility.

Module Author(s): Eric J. South
Dunlop Lab
--------------------------------------------------------------------------------
"""

import datetime
import sys
import uuid
from pathlib import Path

import pandas as pd

# Adjust the Python path to import from the dnadesign project root.
current_file = Path(__file__).resolve()
src_dir = current_file.parent.parent.parent
sys.path.insert(0, str(src_dir))

from dnadesign.utils import BASE_DIR, SequenceSaver, load_dataset

# Constants for the module
DATASET_KEY = "xiaowo_et_al"
SHEETS = ["Lib2-EC", "Lib1-EC"]
OUTPUT_SUBDIR = "seqbatch_xiaowo_et_al"


def process_sheet(sheet_name: str) -> pd.DataFrame:
    """
    Process a single sheet from the xiaowo_et_al dataset.

    For the given sheet:
      - Loads the sheet (header=0) using load_dataset().
      - Renames the first (unnamed) column to "name" and prepends the sheet name
        to each value.
      - Renames the "Sequence" column (case-insensitive) to "sequence"; all other
        columns are prefixed with "meta_".
      - Drops any rows that contain null values or any cell with a 0 value.

    Returns:
        A cleaned pandas DataFrame.
    """
    try:
        df = load_dataset(DATASET_KEY, sheet_name=sheet_name, header=0)
    except Exception as e:
        raise RuntimeError(f"Error loading sheet '{sheet_name}' from dataset {DATASET_KEY}: {e}")

    # Rename the first column (which lacks a proper header) to "name"
    original_first_col = df.columns[0]
    df = df.rename(columns={original_first_col: "name"})

    # Prepend the sheet name to the 'name' column values (e.g., "Lib2-EC-1")
    df["name"] = df["name"].apply(lambda x: f"{sheet_name}-{x}")

    # Build a new columns mapping:
    # - If a column is "Sequence" (case-insensitive), rename it to "sequence"
    # - Otherwise, for all columns (except "name"), prepend "meta_"
    new_columns = {}
    for col in df.columns:
        if col == "name":
            continue
        elif col.lower() == "sequence":
            new_columns[col] = "sequence"
        else:
            new_columns[col] = f"meta_{col}"
    df = df.rename(columns=new_columns)

    # Ensure the required "sequence" column is present.
    assert "sequence" in df.columns, f"Expected 'Sequence' column missing in sheet {sheet_name}."

    # Drop rows with any null values.
    df = df.dropna()

    # Drop rows where any cell equals 0.
    def row_has_zero(row):
        for val in row:
            try:
                if float(val) == 0:
                    return True
            except (ValueError, TypeError):
                continue
        return False

    df = df[~df.apply(row_has_zero, axis=1)]
    return df


def ingest():
    """
    Ingests and processes the xiaowo_et_al dataset across all defined sheets.

    Returns:
        A list of dictionary entries representing standardized sequence data with only
        the desired keys:
            - id, name, sequence, meta_date_accessed, meta_source, meta_part_type,
              meta_exp_mean(log2), meta_tags
    """
    data_frames = []
    for sheet in SHEETS:
        df_sheet = process_sheet(sheet)
        data_frames.append(df_sheet)
    if not data_frames:
        raise ValueError("No data frames were processed from the given sheets.")

    # Concatenate the data from all sheets into a single tidy DataFrame.
    df_all = pd.concat(data_frames, ignore_index=True)

    entries = []
    for _, row in df_all.iterrows():
        # Ensure that both 'name' and 'sequence' exist.
        if "name" not in row or "sequence" not in row:
            continue

        entry = {
            "id": str(uuid.uuid4()),
            "name": row["name"],
            "sequence": row["sequence"],
            "meta_date_accessed": datetime.datetime.now().isoformat(),
            "meta_source": DATASET_KEY,
            "meta_part_type": "engineered promoter",
        }
        # Only include the allowed meta fields if they exist in the row.
        if "meta_exp_mean(log2)" in row:
            entry["meta_exp_mean(log2)"] = row["meta_exp_mean(log2)"]
        if "meta_tags" in row:
            entry["meta_tags"] = row["meta_tags"]

        entries.append(entry)
    return entries


def save_output(sequences: list):
    """
    Saves the list of sequence entries using the SequenceSaver utility.

    The output is stored in a dedicated subfolder under the sequences/ directory.
    """
    output_dir = Path(BASE_DIR) / "src" / "dnadesign" / "sequences" / OUTPUT_SUBDIR
    output_dir.mkdir(parents=True, exist_ok=True)
    saver = SequenceSaver(str(output_dir))
    additional_info = {"source_file": DATASET_KEY, "part_type": "engineered promoter"}
    saver.save_with_summary(sequences, f"{OUTPUT_SUBDIR}.pt", additional_info=additional_info)


if __name__ == "__main__":
    seq_entries = ingest()
    save_output(seq_entries)
