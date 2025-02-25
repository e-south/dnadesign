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
  - Each entry is assigned a unique name and meta_part_type "random promoter".
  - Optionally, a promoter constraint (with upstream and downstream hexamers) 
    is applied in a decoupled module step.
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

import re
import uuid
import yaml
import random
import datetime
import pandas as pd

from dnadesign.utils import SequenceSaver, BASE_DIR

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
            "meta_source": "random promoters",
            "meta_date_accessed": datetime.datetime.now().isoformat(),
            "meta_part_type": "random promoter"
        }
        sequences.append(entry)
    return sequences

def apply_promoter_constraint(seq: str, constraint: dict) -> (str, dict):
    """
    Applies a promoter constraint to a given sequence.
    Inserts the upstream motif at a random position within `upstream_pos` range,
    then places the downstream motif at an offset (randomly chosen within the
    provided `spacer_length` range) from the end of the upstream motif.
    
    Returns the modified sequence and metadata about the insertion, including
    the actual upstream and downstream motifs used.
    """
    upstream = constraint["upstream"]
    downstream = constraint["downstream"]
    pos_min, pos_max = constraint["upstream_pos"]
    spacer_min, spacer_max = constraint["spacer_length"]
    seq_length = len(seq)

    # Try several times to find a valid insertion point.
    for attempt in range(10):
        pos = random.randint(pos_min, pos_max)
        spacer = random.randint(spacer_min, spacer_max)
        downstream_pos = pos + len(upstream) + spacer
        # Ensure that the entire downstream motif fits in the sequence.
        if downstream_pos + len(downstream) <= seq_length:
            seq_list = list(seq)
            # Replace bases with the upstream motif.
            seq_list[pos: pos + len(upstream)] = list(upstream)
            # Replace bases with the downstream motif.
            seq_list[downstream_pos: downstream_pos + len(downstream)] = list(downstream)
            meta = {
                "promoter_constraint": constraint["name"],
                "upstream_insertion_position": pos,
                "spacer_length_used": spacer,
                "downstream_insertion_position": downstream_pos,
                "upstream_motif": upstream,
                "downstream_motif": downstream
            }
            return "".join(seq_list), meta
    raise ValueError("Could not apply promoter constraint: sequence length or specified ranges may be insufficient.")

def process_promoter_constraints(sequences: list, constraint: dict) -> list:
    """
    Applies the promoter constraint to each sequence entry in the provided list.
    Adds a new meta key 'meta_promoter_constraint' to each entry with details.
    """
    for entry in sequences:
        seq = entry["sequence"]
        modified_seq, meta = apply_promoter_constraint(seq, constraint)
        entry["sequence"] = modified_seq
        entry["meta_promoter_constraint"] = meta
    return sequences

def save_output(sequences, num_sequences: int, length: int, gc_min: float, gc_max: float, promoter_constraint: dict = None):
    # If a promoter constraint is provided, incorporate its name into the file and directory.
    if promoter_constraint:
        out_dir_name = f"seqbatch_random_promoters_{promoter_constraint['name']}"
        out_file_name = f"seqbatch_random_promoters_{promoter_constraint['name']}.pt"
    else:
        out_dir_name = "seqbatch_random_promoters"
        out_file_name = "seqbatch_random_promoters.pt"
    output_dir = Path(BASE_DIR) / "src" / "dnadesign" / "sequences" / out_dir_name
    output_dir.mkdir(parents=True, exist_ok=True)
    saver = SequenceSaver(str(output_dir))
    additional_info = {
        "source_file": out_file_name,
        "part_type": "random_promoter",
        "parameters": {
            "num_sequences": num_sequences,
            "sequence_length": length,
            "gc_range": [gc_min, gc_max]
        }
    }
    if promoter_constraint:
        additional_info["promoter_constraint"] = promoter_constraint
    saver.save_with_summary(sequences, out_file_name, additional_info=additional_info)

if __name__ == "__main__":
    # Example parameters; adjust as needed.
    num_sequences = 3000
    length = 120
    gc_min = 40
    gc_max = 60
    seqs = ingest(num_sequences=num_sequences, length=length, gc_min=gc_min, gc_max=gc_max)
    
    # Define the promoter constraint as specified.
    # promoter_constraint = {
    #     "name": "sigma70_consensus",
    #     "upstream": "TTGACA",
    #     "upstream_pos": [10, 100],
    #     "spacer_length": [16, 18],
    #     "downstream": "TATAAT"
    # }
    
    promoter_constraint = {
    "name": "sigma70_consensus_+",
    "upstream": "TTGACA",
    "upstream_pos": [10, 100],
    "spacer_length": [14, 16],
    "downstream": "TGTATAATGCT"
   }
    
    # Apply promoter constraints in a decoupled module step.
    seqs = process_promoter_constraints(seqs, promoter_constraint)
    save_output(seqs, num_sequences, length, gc_min, gc_max, promoter_constraint)
