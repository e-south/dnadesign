"""
--------------------------------------------------------------------------------
<dnadesign project>
latdna/utils.py

Module Author(s): Eric J. South
Dunlop Lab
--------------------------------------------------------------------------------
"""

import os
import uuid
import random
import logging
from pathlib import Path
from datetime import datetime

import torch
import numpy as np
import yaml

def load_yaml(yaml_path: Path) -> dict:
    with yaml_path.open("r") as f:
        return yaml.safe_load(f)

def create_output_directory(base_dir: Path, prefix: str) -> Path:
    """
    Create a new directory with the given prefix and current UTC date stamp.
    """
    timestamp = datetime.utcnow().strftime("%Y%m%d")
    out_dir = base_dir / f"{prefix}_{timestamp}"
    out_dir.mkdir(parents=True, exist_ok=False)
    return out_dir

def read_single_pt_file_from_subdir(subdir: Path) -> list:
    """
    Read a single .pt file from a directory. Fail if not exactly one file.
    """
    pt_files = list(subdir.glob("*.pt"))
    if len(pt_files) != 1:
        raise FileNotFoundError(f"Expected exactly one .pt file in {subdir}, found {len(pt_files)}.")
    return torch.load(pt_files[0])

def write_pt_file(data, output_path: Path):
    torch.save(data, output_path)

def generate_random_dna_sequence(length: int, gc_range: tuple, seed: int = 42) -> str:
    """
    Generate a random DNA sequence of given length with GC content within gc_range (min, max).
    """
    random.seed(seed)
    np.random.seed(seed)
    bases = ['A', 'T', 'C', 'G']
    
    max_attempts = 1000
    for _ in range(max_attempts):
        seq = ''.join(random.choices(bases, k=length))
        gc_content = (seq.count('G') + seq.count('C')) / length
        if gc_range[0] <= gc_content <= gc_range[1]:
            return seq
    raise ValueError("Failed to generate random DNA sequence within GC content range after multiple attempts.")

def reverse_complement(seq: str) -> str:
    """
    Compute the reverse complement of a DNA sequence.
    """
    complement = str.maketrans("ATCGatcg", "TAGCtagc")
    return seq.translate(complement)[::-1]

def generate_uuid() -> str:
    return str(uuid.uuid4())

def current_utc_timestamp() -> str:
    return datetime.utcnow().isoformat() + "Z"