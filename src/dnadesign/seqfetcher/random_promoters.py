"""
--------------------------------------------------------------------------------
<dnadesign project>
seqfetcher/random_promoters.py

Generates a user-specified number of random promoter sequences.
Default parameters:
  - Sequence length: 100 nucleotides
  - GC content range: 40% to 60%
  - Number of sequences: specified by the user

Processing:
  - Random sequences are generated until the desired count is reached,
    each meeting the GC content criteria.
  - Each entry is assigned a unique name and meta_part_type "random_promoter".
  - The standardized data is saved along with a summary.

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

VALID_NUCLEOTIDES = "ATCG"

def generate_random_sequence(length: int, gc_min: float, gc_max: float) -> str:
    while True:
        seq = "".join(random.choices(VALID_NUCLEOTIDES, k=length))
        gc_content = 100 * (seq.count("G") + seq.count("C")) / length
        if gc_min <= gc_content <= gc_max:
            return seq

def ingest(num_sequences: int = 10, length: int = 100, gc_min: float = 40, gc_max: float = 60) -> list:
    sequences = []
    for i in range(num_sequences):
        seq = generate_random_sequence(length, gc_min, gc_max)
        if not seq:
            continue
        entry = {
            "id": str(uuid.uuid4()),
            "name": f"random_promoter_{i+1}",
            "sequence": seq,
            "meta_source": "random_promoters",
            "meta_date_accessed": datetime.datetime.now().isoformat(),
            "meta_part_type": "random_promoter"
        }
        sequences.append(entry)
    return sequences

def save_output(sequences):
    output_dir = Path(BASE_DIR) / "sequences" / "seqbatch_random_promoters"
    output_dir.mkdir(parents=True, exist_ok=True)
    saver = SequenceSaver(str(output_dir))
    saver.save(sequences, "seqset_random_promoters.pt")
    summary = {
        "date_created": datetime.datetime.now().isoformat(),
        "parameters": {
            "num_sequences": len(sequences),
            "sequence_length": len(sequences[0]["sequence"]) if sequences else None,
            "gc_range": [40, 60]
        },
        "part_type": "random_promoter"
    }
    with open(output_dir / "summary.yaml", "w") as f:
        yaml.dump(summary, f)
    print("Summary saved.")

if __name__ == "__main__":
    # You may adjust these parameters as needed
    seqs = ingest(num_sequences=10, length=100, gc_min=40, gc_max=60)
    save_output(seqs)
