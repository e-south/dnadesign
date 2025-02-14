"""
--------------------------------------------------------------------------------
<dnadesign project>
seqfetcher/hernandez_et_al.py

Module for loading and processing data from Hernandez et al., which curated DNA sequences 
from the MG1655 genome (from RegulonDB v9.3.), categorizing them as promoters or non-promoters. 
This dataset was then used to train a multiclass CNN for promoter identification and classification.

"PromoterLCNN: A Light CNN-Based Promoter Prediction and Classification Model"
DOI: 10.3390/genes13071126

Ingests the Hernandez et al dataset from two FASTA files:
    - DATA_FILES["hernandez_et_al_positive"]
    - DATA_FILES["hernandez_et_al_negative"]

Processing:
  - Each file is in FASTA format.
  - For each entry, the header line (starting with '>') is processed:
      • The header is split on underscores and only the first two words are used as the name.
  - The following line is taken as the sequence.
  - Sets meta_part_type to "promoter" for the positive file and "non-promoter" for the negative file.
  - Each file is processed separately to generate its own .pt file and summary.

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

def parse_fasta(file_path: Path, meta_part_type: str, source_key: str) -> list:
    sequences = []
    with file_path.open("r") as f:
        header = None
        for line in f:
            line = line.strip()
            if line.startswith(">"):
                header = line[1:]
                # Extract only the first two underscore-delimited words
                parts = header.split("_")
                name = "_".join(parts[:2]) if len(parts) >= 2 else header
            elif header:
                seq = clean_sequence(line)
                try:
                    if not name or not seq:
                        raise AssertionError("Empty name or sequence.")
                except AssertionError as e:
                    print(f"Skipping entry with header {header}: {e}")
                    header = None
                    continue
                entry = {
                    "id": str(uuid.uuid4()),
                    "name": name,
                    "sequence": seq,
                    "meta_source": source_key,
                    "meta_date_accessed": datetime.datetime.now().isoformat(),
                    "meta_part_type": meta_part_type
                }
                sequences.append(entry)
                header = None  # reset header for next entry
    return sequences

def ingest():
    sequences_all = {}
    for key, part_type in [("hernandez_et_al_positive", "promoter"),
                           ("hernandez_et_al_negative", "non-promoter")]:
        file_path = DATA_FILES[key]
        seqs = parse_fasta(file_path, meta_part_type=part_type, source_key=key)
        sequences_all[key] = seqs
    return sequences_all

def save_output(sequences_dict: dict):
    for key, sequences in sequences_dict.items():
        output_dir = Path(BASE_DIR) / "sequences" / f"seqbatch_{key}"
        output_dir.mkdir(parents=True, exist_ok=True)
        saver = SequenceSaver(str(output_dir))
        saver.save(sequences, f"seqset_{key}.pt")
        summary = {
            "date_created": datetime.datetime.now().isoformat(),
            "source_file": str(DATA_FILES[key]),
            "num_sequences": len(sequences),
            "part_type": key.split("_")[-1]  # "positive" or "negative"
        }
        with open(output_dir / "summary.yaml", "w") as f:
            yaml.dump(summary, f)
        print(f"Summary saved for {key}.")

if __name__ == "__main__":
    seqs = ingest()
    save_output(seqs)
