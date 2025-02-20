"""
--------------------------------------------------------------------------------
<dnadesign project>
dnadesign/evoinference/data_ingestion.py

Module Author(s): Eric J. South
Dunlop Lab
--------------------------------------------------------------------------------
"""

import os
import torch
import re
from logger import get_logger

logger = get_logger(__name__)

def list_pt_files(directory: str):
    """
    Return a list of .pt file paths in the given directory.
    """
    if not os.path.isdir(directory):
        logger.error(f"Directory not found: {directory}")
        raise NotADirectoryError(f"Directory not found: {directory}")
    return [os.path.join(directory, f) for f in os.listdir(directory) if f.endswith('.pt')]

def load_sequences(filepath: str):
    """
    Load sequences from a .pt file. The file should contain a list of dictionaries.
    """
    try:
        data = torch.load(filepath, map_location='cpu')
    except Exception as e:
        logger.error(f"Error loading file {filepath}: {str(e)}")
        raise e
    if not isinstance(data, list):
        logger.error(f"Data in {filepath} is not a list.")
        raise ValueError(f"Data in {filepath} is not a list.")
    for entry in data:
        if not isinstance(entry, dict):
            logger.error(f"One entry in {filepath} is not a dictionary.")
            raise ValueError(f"One entry in {filepath} is not a dictionary.")
    return data

def validate_sequence(entry: dict) -> bool:
    """
    Validate that the entry has a 'sequence' key and that the sequence 
    contains only A, T, G, C characters (case-insensitive).
    """
    if "sequence" not in entry:
        return False
    seq = entry["sequence"]
    if not isinstance(seq, str):
        return False
    if not re.fullmatch(r'[ATGCatgc]+', seq):
        return False
    return True
