"""
--------------------------------------------------------------------------------
<dnadesign project>
seqfetcher/ecocyc_28_promoters.py

Ingests the EcoCyc 28 Promoters SmartTable from:
    DATA_FILES["ecocyc_28_promoters"]
File type: .txt (tab-delimited)
Columns:
  - Name: "Site"
  - Sequence: "Sequence - DNA sequence" (renamed to "sequence")
Sets meta_part_type to "promoter".

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
    file_path = DATA_FILES["ecocyc_28_promoters"]
    # Read as tab-delimited text file.
    df = pd.read_csv(file_path, sep="\t", header=0)
    sequences = []
    for idx, row in df.iterrows():
        name = row.get("Site")
        seq = row.get("Sequence - DNA sequence")
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
            "meta_source": "ecocyc_28_promoters",
            "meta_date_accessed": datetime.datetime.now().isoformat(),
            "meta_part_type": "promoter"
        }
        sequences.append(entry)
    return sequences

def save_output(sequences):
    output_dir = Path(BASE_DIR) / "sequences" / "seqbatch_ecocyc_28_promoters"
    output_dir.mkdir(parents=True, exist_ok=True)
    saver = SequenceSaver(str(output_dir))
    saver.save(sequences, "seqset_ecocyc_28_promoters.pt")
    summary = {
        "date_created": datetime.datetime.now().isoformat(),
        "source_file": str(DATA_FILES["ecocyc_28_promoters"]),
        "num_sequences": len(sequences),
        "part_type": "promoter"
    }
    with open(output_dir / "summary.yaml", "w") as f:
        yaml.dump(summary, f)
    print("Summary saved.")

if __name__ == "__main__":
    seqs = ingest()
    save_output(seqs)
