"""
--------------------------------------------------------------------------------
<dnadesign project>
seqfetcher/urtecho_et_al.py

Module for loading and data described in Urtecho et al., which created and 
characterized a 10,898‐variant library of bacterial promoters, dissecting 
−35, −10, UP elements, and spacers to evaluate σ70‐dependent expression.

"Systematic Dissection of Sequence Elements Controlling σ70 Promoters Using a 
Genomically Encoded Multiplexed Reporter Assay in Escherichia coli"
DOI: 10.1021/acs.biochem.7b01069

Ingests the Urtecho et al promoter dataset from:
    DATA_FILES["urtecho_et_al"]
Sheet: "Urtecho et al (Fig 3c, S7b)"
Columns:
  - Name: "ID"
  - Sequence: "Promoter Sequence"
  - Additional meta: "Observed log(TX/Txref)"
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
    df = load_dataset("urtecho_et_al", sheet_name="Urtecho et al (Fig 3c, S7b)")
    sequences = []
    for idx, row in df.iterrows():
        name = row.get("ID")
        seq = row.get("Promoter Sequence")
        obs_log = row.get("Observed log(TX/Txref)")
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
            "meta_source": "urtecho_et_al",
            "meta_date_accessed": datetime.datetime.now().isoformat(),
            "meta_observed_log_RNA_over_ref": obs_log,
            "meta_part_type": "engineered promoter"
        }
        sequences.append(entry)
    return sequences

def save_output(sequences):
    output_dir = Path(BASE_DIR) / "src" / "dnadesign" / "sequences" / "seqbatch_urtecho_et_al"
    output_dir.mkdir(parents=True, exist_ok=True)
    saver = SequenceSaver(str(output_dir))
    additional_info = {
        "source_file": "urtecho_et_al",
        "part_type": "engineered promoter"
    }
    saver.save_with_summary(sequences, "seqbatch_urtecho_et_al.pt", additional_info=additional_info)

if __name__ == "__main__":
    seqs = ingest()
    save_output(seqs)
