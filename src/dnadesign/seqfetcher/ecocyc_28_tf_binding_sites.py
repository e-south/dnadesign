"""
--------------------------------------------------------------------------------
<dnadesign project>
seqfetcher/ecocyc_28_tf_binding_sites.py

Ingests the EcoCyc Transcription Factor Binding Site SmartTable from:
    DATA_FILES["ecocyc_28_tf_binding_sites"]

File type: .txt (tab-delimited)
Reading parameters:
    - header: 2 (i.e. the header is on the third row of the file)
Expected columns:
    - 'Site'
    - 'Regulator'
    - 'Left'
    - 'Right'
    - 'Strand'
    - 'Transcription-Units'
    - 'Sequence - DNA sequence'

Processing:
    - The sequence is read from "Sequence - DNA sequence", cleaned (whitespace removed, uppercase, only A, T, C, G retained).
    - The name is taken from "Site" and appended with the row number (e.g., "SiteName_23") to ensure uniqueness.
Additional metadata:
    - 'meta_part_type' is set to "tfbs"

Module Author(s): Eric J. South
Dunlop Lab
--------------------------------------------------------------------------------
"""

import pandas as pd
import re
import datetime
import uuid
import yaml
from pathlib import Path
from dnadesign.utils import SequenceSaver, DATA_FILES, BASE_DIR

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
    file_path = DATA_FILES["ecocyc_28_tf_binding_sites"]
    if not file_path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")
    # Read the file with header on the third row (0-indexed header=2), assuming tab-delimited.
    df = pd.read_csv(file_path, sep="\t", header=2)
    sequences = []
    for idx, row in df.iterrows():
        site = row.get("Site")
        seq = row.get("Sequence - DNA sequence")
        # Append row number to the site name to ensure uniqueness.
        name = f"{site}_{idx}"
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
            "meta_source": "ecocyc_28_tf_binding_sites",
            "meta_date_accessed": datetime.datetime.now().isoformat(),
            "meta_part_type": "tfbs"
        }
        sequences.append(entry)
    return sequences

def save_output(sequences):
    output_dir = Path(BASE_DIR) / "sequences" / "seqbatch_ecocyc_28_tf_binding_sites"
    output_dir.mkdir(parents=True, exist_ok=True)
    saver = SequenceSaver(str(output_dir))
    saver.save(sequences, "seqset_ecocyc_28_tf_binding_sites.pt")
    summary = {
        "date_created": datetime.datetime.now().isoformat(),
        "source_file": str(DATA_FILES["ecocyc_28_tf_binding_sites"]),
        "num_sequences": len(sequences),
        "part_type": "tfbs"
    }
    with open(output_dir / "summary.yaml", "w") as f:
        yaml.dump(summary, f)
    print("Summary saved.")

if __name__ == "__main__":
    seqs = ingest()
    save_output(seqs)
