"""
--------------------------------------------------------------------------------
<dnadesign project>
latdna/validation.py

Module Author(s): Eric J. South
Dunlop Lab
--------------------------------------------------------------------------------
"""

import logging

def validate_densegen_entry(entry: dict, index: int, source_file: str):
    """
    Validate that a dense batch entry contains all required keys.
    """
    required_keys = [
        "sequence",
        "sequence_length",
        "fixed_elements",
        "meta_tfbs_parts",
        "meta_tfbs_parts_in_array"
    ]
    missing = [key for key in required_keys if key not in entry]
    if missing:
        raise ValueError(f"Entry {index} in file '{source_file}' is missing required key(s): {missing}")

def validate_config_motifs(motifs: dict):
    """
    Validate that the motifs configuration is correct and that no duplicate motif sequences exist across TFs.
    """
    motif_map = {}
    for tf, motif_values in motifs.items():
        tf_lower = tf.lower()  # case-insensitive
        if isinstance(motif_values, str):
            motif_list = [motif_values]
        elif isinstance(motif_values, list):
            motif_list = motif_values
        else:
            raise ValueError(f"Motif for TF '{tf}' must be a string or a list of strings.")
        
        for motif in motif_list:
            if motif in motif_map:
                raise ValueError(f"Duplicate motif detected: '{motif}' is assigned to both '{motif_map[motif]}' and '{tf}'. Please disambiguate.")
            motif_map[motif] = tf_lower

def validate_analysis_entry(entry: dict, index: int, source_file: str):
    """
    Validate that an analysis entry contains the 'evo2_logits_mean_pooled' key.
    """
    if "evo2_logits_mean_pooled" not in entry:
        raise ValueError(f"Entry {index} in file '{source_file}' is missing 'evo2_logits_mean_pooled'. All entries must have been processed by Evo before analysis.")