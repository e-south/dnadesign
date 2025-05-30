"""
--------------------------------------------------------------------------------
<dnadesign project>
seqfetcher/johns_et_al.py

Module for loading data described in Johns et al., which performed large‐scale 
analysis of diverse bacterial regulatory sequences for species‐selective gene expression.

"Metagenomic mining of regulatory elements enables programmable species-selective gene expression"
DOI: 10.1038/nmeth.4633

Ingests the Johns et al datasets from:
    DATA_FILES["johns_et_al_sequences"] and DATA_FILES["johns_et_al_labels"]

Processing:
  - Loads sequences from the "Full Constructs" sheet:
      • Uses 'Oligo ID' as the name.
      • Uses 'Regulatory Sequence + ATG + 12bp Barcode' as the sequence (trims the trailing 12 characters).
  - Loads label data from multiple sheets ("LB_exp", "NaCl_exp", "Fe_exp", "LB-stat", "M9-exp"):
      • Each sheet contains 'OLIGO ID' and 'tx_norm' (renamed to meta_observed_tx_norm_<sheet>).
  - Merges the sequence and label data on the common "name" field using inner joins.
  - Drops any entries with null or empty fields.
  - Drops rows where any label value (in tx_raw or tx_norm columns) is 0 or infinite.
  - Sets meta_part_type to "promoter".
  - Saves the standardized data along with a summary.

Module Author(s): Eric J. South
Dunlop Lab
--------------------------------------------------------------------------------
"""

import math
import sys
from pathlib import Path

import numpy as np

current_file = Path(__file__).resolve()
src_dir = current_file.parent.parent.parent
sys.path.insert(0, str(src_dir))

import datetime
import uuid

import pandas as pd

from dnadesign.utils import BASE_DIR, SequenceSaver, load_dataset

VALID_NUCLEOTIDES = set("ATCG")
LABEL_SHEETS = ["LB_exp", "NaCl_exp", "Fe_exp", "LB-stat", "M9-exp"]


def trim_barcode(seq: str) -> str:
    if not isinstance(seq, str):
        return ""
    seq = seq.strip().upper().replace(" ", "")
    return seq[:-12] if len(seq) >= 12 else ""


def validate_entry(name: str, seq: str):
    if not name or pd.isna(name):
        raise AssertionError("Empty name field.")
    if not seq:
        raise AssertionError("Empty sequence field.")
    for c in seq:
        if c not in VALID_NUCLEOTIDES:
            raise AssertionError(f"Invalid nucleotide '{c}' in sequence: {seq}")


def is_invalid_value(val) -> bool:
    """
    Returns True if the given value, when converted to a float, is 0 or infinite.
    If conversion fails, returns True to be conservative.
    """
    try:
        numeric_value = float(val)
        if numeric_value == 0 or math.isinf(numeric_value) or np.isinf(numeric_value):
            return True
        return False
    except Exception:
        return True


def ingest():
    # Load sequences
    df_seq = load_dataset("johns_et_al_sequences", sheet_name="Full Constructs", header=0)
    df_seq = df_seq.rename(columns={"Oligo ID": "name", "Regulatory Sequence + ATG + 12bp Barcode": "sequence"})
    df_seq["sequence"] = df_seq["sequence"].apply(trim_barcode)

    # Merge labels from each sheet
    df_merged = df_seq.copy()
    for sheet in LABEL_SHEETS:
        df_label = load_dataset("johns_et_al_labels", sheet_name=sheet, header=0)
        df_label = df_label.rename(columns={"OLIGO ID": "name", "tx_norm": f"meta_observed_tx_norm_{sheet}"})
        # Only keep the key and the tx_norm column
        df_label = df_label[["name", f"meta_observed_tx_norm_{sheet}"]]
        # Merge using inner join to drop non-matches
        df_merged = pd.merge(df_merged, df_label, on="name", how="inner")

    # Drop rows with any null values in key columns
    df_merged = df_merged.dropna(subset=["name", "sequence"])

    # Remove rows where any tx_raw or tx_norm value is 0 or infinite
    # Check all columns that start with "meta_observed_tx_norm_" or "meta_observed_tx_raw_"
    def row_is_invalid(row):
        for col in df_merged.columns:
            if col.startswith("meta_observed_tx_norm_") or col.startswith("meta_observed_tx_raw_"):
                if is_invalid_value(row[col]):
                    return True
        return False

    df_clean = df_merged[~df_merged.apply(row_is_invalid, axis=1)]

    sequences = []
    for idx, row in df_clean.iterrows():
        name = row["name"]
        seq = row["sequence"]
        try:
            validate_entry(name, seq)
        except AssertionError as e:
            print(f"Skipping row {idx}: {e}")
            continue
        # Collect all label meta from the row (all columns starting with meta_observed_tx_norm_)
        label_meta = {col: row[col] for col in df_clean.columns if col.startswith("meta_observed_tx_norm_")}
        entry = {
            "id": str(uuid.uuid4()),
            "name": name,
            "sequence": seq,
            "meta_source": "johns_et_al",
            "meta_date_accessed": datetime.datetime.now().isoformat(),
            "meta_part_type": "natural promoter",
        }
        entry.update(label_meta)
        sequences.append(entry)
    return sequences


def save_output(sequences):
    output_dir = Path(BASE_DIR) / "src" / "dnadesign" / "sequences" / "seqbatch_johns_et_al"
    output_dir.mkdir(parents=True, exist_ok=True)
    saver = SequenceSaver(str(output_dir))
    additional_info = {"source_file": "johns_et_al", "part_type": "promoter"}
    saver.save_with_summary(sequences, "seqbatch_johns_et_al.pt", additional_info=additional_info)


if __name__ == "__main__":
    seqs = ingest()
    save_output(seqs)
