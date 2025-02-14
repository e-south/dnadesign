"""
--------------------------------------------------------------------------------
<dnadesign project>
seqfetcher/regulondb_13_promoter_set.py

Ingests the RegulonDB 13 Promoter Set from:
    DATA_FILES["regulondb_13_promoter_set"]
File type: TSV; note that the first 28 rows are comments so the header is on row 29.
Columns of interest:
  - Name: "2)pmName"
  - Sequence: "6)pmSequence" (to be renamed to "sequence")
Sets meta_part_type to "promoter".

Module Author(s): Eric J. South
Dunlop Lab
--------------------------------------------------------------------------------
"""

import sys
from pathlib import Path

current_file = Path(__file__).resolve()
src_dir = current_file.parent.parent.parent
sys.path.insert(0, str(src_dir))

import pandas as pd
import re
import datetime
import uuid
import yaml

from dnadesign.utils import load_dataset, SequenceSaver, DATA_FILES, BASE_DIR

VALID_NUCLEOTIDES = set("ATCG")

def clean_sequence(seq: str) -> str:
    if not isinstance(seq, str):
        return ""
    seq = seq.strip().upper()
    seq = re.sub(r"\s+", "", seq)
    return "".join(c for c in seq if c in VALID_NUCLEOTIDES)

def validate_entry(name: str, seq: str):
    if not name or pd.isna(name):
        raise AssertionError("Empty name field.")
    if not seq:
        raise AssertionError("Empty sequence field.")
    for c in seq:
        if c not in VALID_NUCLEOTIDES:
            raise AssertionError(f"Invalid nucleotide '{c}' in sequence: {seq}")

def ingest():
    file_path = DATA_FILES["regulondb_13_promoter_set"]
    df = pd.read_csv(file_path, sep="\t", skiprows=29, header=0)
    sequences = []
    for idx, row in df.iterrows():
        name = row.get("2)pmName")
        seq = row.get("6)pmSequence")
        if pd.isna(name):
            name = f"Row_{idx}"
        if pd.isna(seq):
            continue
        seq = clean_sequence(seq)
        try:
            validate_entry(name, seq)
        except AssertionError as e:
            print(f"Skipping row {idx}: {e}")
            continue
        entry = {
            "id": str(uuid.uuid4()),
            "name": name,
            "sequence": seq,
            "meta_source": "regulondb_13_promoter_set",
            "meta_date_accessed": datetime.datetime.now().isoformat(),
            "meta_part_type": "promoter"
        }
        sequences.append(entry)
    return sequences

def save_output(sequences):
    output_dir = Path(BASE_DIR) / "src" / "dnadesign" / "sequences" / "seqbatch_regulondb_13_promoter_set"
    output_dir.mkdir(parents=True, exist_ok=True)
    saver = SequenceSaver(str(output_dir))
    additional_info = {
        "source_file": "regulondb_13_promoter_set",
        "part_type": "promoter"
    }
    saver.save_with_summary(sequences, "seqbatch_regulondb_13_promoter_set.pt", additional_info=additional_info)

if __name__ == "__main__":
    seqs = ingest()
    save_output(seqs)
