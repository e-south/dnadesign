"""
--------------------------------------------------------------------------------
<dnadesign project>
seqfetcher/regulondb_13_tf_ri_set.py

Ingests the RegulonDB Transcription Factor Binding Site Dataset from:
    DATA_FILES["regulondb_13_tf_ri_set"]

File type: TSV (tab-delimited)
Reading parameters:
    - The first 45 rows are comments (starting with '#')
    - The header is on the row following these comments.
Expected columns include:
    - "1)riId" (to be used as the unique name)
    - "10)tfrsSeq" (to be renamed to "sequence")
    - "4)regulatorName" (to be saved as the 'regulator' key)
Processing:
    - The sequence is cleaned (removing whitespace, uppercasing, and filtering valid nucleotides).
    - The name is taken from "1)riId" (or generated if missing).
Additional metadata:
    - 'meta_part_type' is set to "tfbs"

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
    file_path = DATA_FILES["regulondb_13_tf_ri_set"]
    if not file_path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")
    # Skip the first 45 rows (which are comments) and then use the next row as the header.
    df = pd.read_csv(file_path, sep="\t", skiprows=44, header=0)
    sequences = []
    for idx, row in df.iterrows():
        name = row.get("1)riId")
        seq = row.get("10)tfrsSeq")
        regulator = row.get("4)regulatorName")  # Retrieve regulator name
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
            "regulator": regulator,  # New key for regulator name
            "meta_source": "regulondb_13_tf_ri_set",
            "meta_date_accessed": datetime.datetime.now().isoformat(),
            "meta_part_type": "natural tfbs"
        }
        sequences.append(entry)
    return sequences

def save_output(sequences):
    output_dir = Path(BASE_DIR) / "src" / "dnadesign" / "sequences" / "seqbatch_regulondb_13_tf_ri_set"
    output_dir.mkdir(parents=True, exist_ok=True)
    saver = SequenceSaver(str(output_dir))
    additional_info = {
        "source_file": "regulondb_13_tf_ri_set",
        "part_type": "tfbs"
    }
    saver.save_with_summary(sequences, "seqbatch_regulondb_13_tf_ri_set.pt", additional_info=additional_info)

if __name__ == "__main__":
    seqs = ingest()
    save_output(seqs)
