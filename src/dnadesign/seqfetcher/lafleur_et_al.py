"""
--------------------------------------------------------------------------------
<dnadesign project>
seqfetcher/lafleur_et_al.py

Ingests the La Fleur et al promoter dataset from:
    DATA_FILES["lafleur_et_al"]
Sheet: "La Fleur et al (Fig 3a)"
Columns:
  - No explicit name column is provided; we generate names as "Row_X".
  - Sequence: "Promoter Sequence"
  - Additional meta: "Observed log(TX/Txref)"
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
    df = load_dataset("lafleur_et_al", sheet_name="La Fleur et al (Fig 3a)")
    sequences = []
    for idx, row in df.iterrows():
        # No name column; use row number as name.
        name = f"Row_{idx}"
        seq = row.get("Promoter Sequence")
        obs_log = row.get("Observed log(TX/Txref)")
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
            "meta_source": "lafleur_et_al",
            "meta_date_accessed": datetime.datetime.now().isoformat(),
            "meta_observed_log_RNA_over_ref": obs_log,
            "meta_part_type": "promoter"
        }
        sequences.append(entry)
    return sequences

def save_output(sequences):
    output_dir = Path(BASE_DIR) / "sequences" / "seqbatch_lafleur_et_al"
    output_dir.mkdir(parents=True, exist_ok=True)
    saver = SequenceSaver(str(output_dir))
    saver.save(sequences, "seqset_lafleur_et_al.pt")
    summary = {
        "date_created": datetime.datetime.now().isoformat(),
        "source_file": str(DATA_FILES["lafleur_et_al"]),
        "num_sequences": len(sequences),
        "part_type": "promoter"
    }
    with open(output_dir / "summary.yaml", "w") as f:
        yaml.dump(summary, f)
    print("Summary saved.")

if __name__ == "__main__":
    seqs = ingest()
    save_output(seqs)
