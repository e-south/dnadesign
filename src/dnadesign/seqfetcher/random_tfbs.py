"""
--------------------------------------------------------------------------------
<dnadesign project>
seqfetcher/random_tfbs.py

Generates a user-specified number of random transcription factor binding site (TFBS) sequences.
Default parameters:
  - Sequence length: 18 nucleotides
  - GC content range: 40% to 60%
  - Number of sequences: specified by the user

Processing:
  - Random sequences are generated until the desired count is reached,
    each meeting the GC content criteria.
  - Each entry is assigned a unique name and meta_part_type "random_tfbs".
  - The standardized data is saved along with a summary that includes the 
    user-defined parameters.
    
Module Author(s): Eric J. South
Dunlop Lab
--------------------------------------------------------------------------------
"""

import sys
from pathlib import Path

current_file = Path(__file__).resolve()
src_dir = current_file.parent.parent.parent
sys.path.insert(0, str(src_dir))

import random
import pandas as pd
import re
import datetime
import uuid
import yaml

from dnadesign.utils import SequenceSaver, BASE_DIR

VALID_NUCLEOTIDES = "ATCG"

def generate_random_sequence(length: int, gc_min: float, gc_max: float) -> str:
    while True:
        seq = "".join(random.choices(VALID_NUCLEOTIDES, k=length))
        gc_content = 100 * (seq.count("G") + seq.count("C")) / length
        if gc_min <= gc_content <= gc_max:
            return seq

def ingest(num_sequences: int = 10, length: int = 18, gc_min: float = 40, gc_max: float = 60) -> list:
    sequences = []
    for i in range(num_sequences):
        seq = generate_random_sequence(length, gc_min, gc_max)
        if not seq:
            continue
        entry = {
            "id": str(uuid.uuid4()),
            "name": f"random_tfbs_{i+1}",
            "sequence": seq,
            "meta_source": "random tfbs",
            "meta_date_accessed": datetime.datetime.now().isoformat(),
            "meta_part_type": "random tfbs"
        }
        sequences.append(entry)
    return sequences

def save_output(sequences, num_sequences: int, length: int, gc_min: float, gc_max: float):
    output_dir = Path(BASE_DIR) / "src" / "dnadesign" / "sequences" / "seqbatch_random_tfbs"
    output_dir.mkdir(parents=True, exist_ok=True)
    saver = SequenceSaver(str(output_dir))
    additional_info = {
        "source_file": "seqbatch_random_tfbs",
        "part_type": "random_tfbs",
        "parameters": {
            "num_sequences": num_sequences,
            "sequence_length": length,
            "gc_range": [gc_min, gc_max]
        }
    }
    saver.save_with_summary(sequences, "seqbatch_random_tfbs.pt", additional_info=additional_info)

if __name__ == "__main__":
    # Example parameters; adjust as needed.
    num_sequences = 5000
    length = 18
    gc_min = 40
    gc_max = 60
    seqs = ingest(num_sequences=num_sequences, length=length, gc_min=gc_min, gc_max=gc_max)
    save_output(seqs, num_sequences, length, gc_min, gc_max)
