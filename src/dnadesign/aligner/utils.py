"""
--------------------------------------------------------------------------------
<dnadesign project>
aligner/utils.py

Module Author(s): Eric J. South
Dunlop Lab
--------------------------------------------------------------------------------
"""

from typing import Any

def validate_sequence(seq: Any) -> str:
    """
    Validate and clean a nucleotide sequence.
    
    Ensures that the sequence is a non-empty string, converts it to uppercase,
    and checks that it contains only valid nucleotide characters (A, C, G, T, N).
    
    Parameters:
        seq: The input sequence (expected to be a string).
    
    Returns:
        The cleaned and validated sequence in uppercase.
    
    Raises:
        ValueError: If the sequence is invalid.
    """
    if not isinstance(seq, str) or not seq.strip():
        raise ValueError(f"Invalid sequence: must be a non-empty string, got: {repr(seq)}")
    seq = seq.upper()
    allowed = set("ACGTN")
    for base in seq:
        if base not in allowed:
            raise ValueError(f"Invalid character '{base}' in sequence: {seq}")
    return seq

def extract_sequence(item: Any, key: str = "sequence") -> str:
    """
    Extract a sequence from an item, which can be either a string or a dictionary.
    
    Parameters:
        item: A sequence string or a dict containing a sequence.
        key: The key to use for extraction if item is a dict.
    
    Returns:
        The cleaned and validated sequence.
    
    Raises:
        ValueError: If the sequence cannot be extracted.
    """
    if isinstance(item, str):
        return validate_sequence(item)
    elif isinstance(item, dict):
        seq = item.get(key)
        if seq is None:
            raise ValueError(f"Dictionary does not contain the key '{key}'.")
        return validate_sequence(seq)
    else:
        raise ValueError("Item must be a string or a dict with a sequence key.")