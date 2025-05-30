"""
--------------------------------------------------------------------------------
<dnadesign project>
seqfetcher/sun_yim_et_al.py

Module for loading and data described in Sun Yim et al., which used active lysates 
from 10 bacterial species to measure transcription activities of thousands of 
regulatory sequences.

"Multiplex transcriptional characterizations across diverse bacterial species using 
cell-free systems"
DOI: 10.15252/msb.20198875

Ingests the Sun Yim et al dataset from:
    DATA_FILES["sun_yim_et_al"]

Processing:
  - Reads the Excel file from the sheet "Fig. 1D" (header=0).
  - Uses 'Oligo ID' as the name and 'Sequence+ATG+BC(rev)' as the sequence.
  - Cleans the sequence by trimming the trailing 12 characters (barcode).
  - Also extracts 'Gen_in vitro_tx' and 'Gen_in vivo_tx' as additional metadata.
  - Sets meta_part_type to "promoter".
  - Outputs a standardized list of dictionaries and saves them along with a summary.

Module Author(s): Eric J. South
Dunlop Lab
--------------------------------------------------------------------------------
"""

import sys
from pathlib import Path

current_file = Path(__file__).resolve()
src_dir = current_file.parent.parent.parent
sys.path.insert(0, str(src_dir))

import datetime
import re
import uuid

import pandas as pd

from dnadesign.utils import BASE_DIR, SequenceSaver, load_dataset

VALID_NUCLEOTIDES = set("ATCG")


def clean_sequence(seq: str) -> str:
    if not isinstance(seq, str):
        return ""
    seq = seq.strip().upper()
    seq = re.sub(r"\s+", "", seq)
    # Trim trailing 12 characters (barcode) if possible
    return seq[:-12] if len(seq) >= 12 else ""


def validate_entry(name: str, seq: str):
    if not name or pd.isna(name):
        raise AssertionError("Empty name field.")
    if not seq:
        raise AssertionError("Empty sequence field.")
    for c in seq:
        if c not in VALID_NUCLEOTIDES:
            raise AssertionError(f"Invalid nucleotide '{c}' in sequence: {seq}")


def ingest():
    df = load_dataset("sun_yim_et_al", sheet_name="Fig. 1D", header=0)
    sequences = []
    for idx, row in df.iterrows():
        name = row.get("Oligo ID")
        if pd.isna(name):
            name = f"Row_{idx}"
        raw_seq = row.get("Sequence+ATG+BC(rev)")
        if pd.isna(raw_seq):
            continue
        seq = clean_sequence(raw_seq)
        try:
            validate_entry(name, seq)
        except AssertionError as e:
            print(f"Skipping row {idx}: {e}")
            continue
        entry = {
            "id": str(uuid.uuid4()),
            "name": name,
            "sequence": seq,
            "meta_source": "sun_yim_et_al",
            "meta_date_accessed": datetime.datetime.now().isoformat(),
            "meta_gene_in_vitro_tx": row.get("Gen_in vitro_tx"),
            "meta_gene_in_vivo_tx": row.get("Gen_in vivo_tx"),
            "meta_part_type": "promoter",
        }
        sequences.append(entry)
    return sequences


def save_output(sequences):
    output_dir = Path(BASE_DIR) / "src" / "dnadesign" / "sequences" / "seqbatch_sun_yim_et_al"
    output_dir.mkdir(parents=True, exist_ok=True)
    saver = SequenceSaver(str(output_dir))
    additional_info = {"source_file": "sun_yim_et_al", "part_type": "engineered promoter"}
    saver.save_with_summary(sequences, "seqbatch_sun_yim_et_al.pt", additional_info=additional_info)


if __name__ == "__main__":
    seqs = ingest()
    save_output(seqs)
