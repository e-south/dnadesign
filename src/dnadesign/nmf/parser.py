"""
--------------------------------------------------------------------------------
<dnadesign project>
nmf/parser.py

Module Author(s): Eric J. South
Dunlop Lab
--------------------------------------------------------------------------------
"""

import torch
from typing import Tuple, List, Dict

def robust_parse_tfbs(part: str, seq_id: str = "unknown") -> Tuple[str, str]:
    """
    Parse a meta_tfbs_parts string robustly.
    Supports two formats:
      - Colon-delimited: "tf:motif"
      - Underscore-delimited (legacy): "idx_{index}_{TF_name}_{motif}"
    Returns a tuple (tf_name, motif) with tf_name in lowercase and motif in uppercase.
    """
    if ":" in part:
        parts = part.split(":", 1)
        if len(parts) != 2:
            raise ValueError(f"Invalid format for TFBS part in sequence {seq_id}: {part}")
        tf_name, motif = parts
        tf_name = tf_name.strip().lower()
        motif = motif.strip().upper()
        if not motif.isalpha() or not all(c in "ATCG" for c in motif):
            raise ValueError(f"Motif candidate '{motif}' is not valid in sequence {seq_id}")
        return tf_name, motif

    # Legacy underscore-delimited format
    if part.startswith("idx_"):
        s = part[4:]
    else:
        s = part
    parts = s.split("_")
    if len(parts) < 2:
        raise ValueError(f"Invalid meta_tfbs_parts format in sequence {seq_id}: {part}")
    motif_candidate = parts[-1].strip().upper()
    if not motif_candidate.isalpha() or not all(c in "ATCG" for c in motif_candidate):
        raise ValueError(f"Motif candidate '{motif_candidate}' is not valid in sequence {seq_id}")
    tf_name = "_".join(parts[1:-1]).strip().lower() if len(parts) > 2 else parts[0].strip().lower()
    return tf_name, motif_candidate

def load_pt_file(pt_path: str) -> List[Dict]:
    """
    Load a .pt file that contains a list of sequence dictionaries.
    """
    try:
        data = torch.load(pt_path, map_location="cpu")
    except Exception as e:
        raise ValueError(f"Failed to load .pt file {pt_path}: {str(e)}")
    if not isinstance(data, list):
        raise ValueError(f"Expected a list in the .pt file, got {type(data)} instead")
    return data