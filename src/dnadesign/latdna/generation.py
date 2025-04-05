"""
--------------------------------------------------------------------------------
<dnadesign project>
latdna/generation.py

Module Author(s): Eric J. South
Dunlop Lab
--------------------------------------------------------------------------------
"""

import logging
from pathlib import Path
import yaml
import torch

from dnadesign.latdna import utils, validation

def get_valid_tiling_positions(sequence_length: int, motif_length: int, fixed_regions: list, step: int) -> list:
    """
    Compute valid tiling positions for a motif in a sequence of given length.
    'fixed_regions' is a list of (start, end) tuples that must be avoided.
    """
    positions = []
    for pos in range(0, sequence_length - motif_length + 1, step):
        overlap = any(pos < end and (pos + motif_length) > start for start, end in fixed_regions)
        if not overlap:
            positions.append(pos)
    if not positions:
        raise ValueError("No valid tiling positions found for motif without overlapping fixed regions.")
    return positions

def run_generation_pipeline(config: dict):
    """
    Execute the synthetic sequence generation pipeline.
    """
    logging.info("Starting generation pipeline...")
    
    # Resolve the dense batch directory from the config.
    sequences_dir = Path(__file__).parent.parent / "sequences"
    dense_batch_name = config.get("dense_array_for_generation")
    if not dense_batch_name:
        raise ValueError("Config missing 'dense_array_for_generation'")
    
    dense_batch_dir = sequences_dir / dense_batch_name
    if not dense_batch_dir.exists():
        raise FileNotFoundError(f"Dense batch directory {dense_batch_dir} does not exist.")
    
    # Load the dense PT file.
    dense_data = utils.read_single_pt_file_from_subdir(dense_batch_dir)
    logging.info(f"Loaded {len(dense_data)} entries from dense batch '{dense_batch_name}'.")
    
    # Validate each dense entry.
    for idx, entry in enumerate(dense_data):
        validation.validate_densegen_entry(entry, idx, dense_batch_name)
    
    # Validate motifs in config.
    motifs_config = config.get("motifs")
    if not motifs_config:
        raise ValueError("Config missing 'motifs' under latdna.")
    validation.validate_config_motifs(motifs_config)
    
    # Generate the base random sequence.
    seq_length = config.get("sequence_length")
    gc_range = tuple(config.get("gc_content_range", [0.4, 0.6]))
    seed = config.get("seed", 42)
    base_sequence = utils.generate_random_dna_sequence(seq_length, gc_range, seed)
    logging.info(f"Generated base random sequence of length {seq_length}.")
    
    # Fixed elements.
    fixed_config = config.get("fixed_elements", {})
    upstream_seq = fixed_config.get("upstream_seq")
    downstream_seq = fixed_config.get("downstream_seq")
    upstream_start = fixed_config.get("upstream_start")
    spacer_range = fixed_config.get("spacer_range")
    if not (upstream_seq and downstream_seq and upstream_start and spacer_range):
        raise ValueError("Fixed elements config incomplete. Specify upstream_seq, downstream_seq, upstream_start, and spacer_range.")
    
    # Determine spacer length deterministically.
    spacer_length = spacer_range[0]
    downstream_start = upstream_start + len(upstream_seq) + spacer_length
    if downstream_start + len(downstream_seq) > seq_length:
        raise ValueError("Downstream motif would exceed sequence bounds. Adjust upstream_start or spacer_range.")
    
    # Define fixed regions to avoid (upstream and downstream).
    fixed_regions = [
        (upstream_start, upstream_start + len(upstream_seq)),
        (downstream_start, downstream_start + len(downstream_seq))
    ]
    
    tiling_step = config.get("tiling", {}).get("step", 6)
    synthetic_entries = []
    
    # Iterate over each TF from the motifs configuration.
    for tf, motif_values in motifs_config.items():
        if isinstance(motif_values, str):
            motif_list = [motif_values]
        else:
            motif_list = motif_values
        
        # Check if the TF is present in the dense batch.
        tf_found = False
        for entry in dense_data:
            parts = entry.get("meta_tfbs_parts", [])
            for part in parts:
                if part.split(":")[0].lower() == tf.lower():
                    tf_found = True
                    break
            if tf_found:
                break
        if not tf_found:
            logging.warning(f"TF '{tf}' not found in dense batch. Skipping.")
            continue
        
        # Process each motif for the TF.
        for motif in motif_list:
            motif_length = len(motif)
            valid_positions = get_valid_tiling_positions(seq_length, motif_length, fixed_regions, tiling_step)
            logging.info(f"TF '{tf}', motif '{motif}': Found {len(valid_positions)} valid tiling positions.")
            
            for pos in valid_positions:
                # Work on a copy of the base sequence (as a list for mutability).
                sequence_list = list(base_sequence)
                
                # Insert fixed elements.
                sequence_list[upstream_start:upstream_start+len(upstream_seq)] = list(upstream_seq)
                sequence_list[downstream_start:downstream_start+len(downstream_seq)] = list(downstream_seq)
                
                def insert_motif(seq_list, motif, position):
                    seq_list[position:position+len(motif)] = list(motif)
                    return "".join(seq_list)
                
                # Create forward strand entry.
                seq_forward = insert_motif(sequence_list.copy(), motif, pos)
                entry_forward = {
                    "id": utils.generate_uuid(),
                    "sequence": seq_forward,
                    "sequence_length": seq_length,
                    "transcription_factor": tf,
                    "tiled_motif": motif,
                    "tiled_position": pos,
                    "motif_strand": "forward",
                    "fixed_elements": {
                        "upstream_seq": upstream_seq,
                        "upstream_start": upstream_start,
                        "downstream_seq": downstream_seq,
                        "downstream_start": downstream_start,
                        "spacer_length": spacer_length,
                    },
                    "meta_latdna_source": dense_batch_name,
                    "meta_date_generated": utils.current_utc_timestamp(),
                }
                synthetic_entries.append(entry_forward)
                
                # Create reverse complement entry.
                rev_motif = utils.reverse_complement(motif)
                seq_revcom = insert_motif(sequence_list.copy(), rev_motif, pos)
                entry_revcom = {
                    "id": utils.generate_uuid(),
                    "sequence": seq_revcom,
                    "sequence_length": seq_length,
                    "transcription_factor": tf,
                    "tiled_motif": rev_motif,
                    "tiled_position": pos,
                    "motif_strand": "revcom",
                    "fixed_elements": {
                        "upstream_seq": upstream_seq,
                        "upstream_start": upstream_start,
                        "downstream_seq": downstream_seq,
                        "downstream_start": downstream_start,
                        "spacer_length": spacer_length,
                    },
                    "meta_latdna_source": dense_batch_name,
                    "meta_date_generated": utils.current_utc_timestamp(),
                }
                synthetic_entries.append(entry_revcom)
    
    logging.info(f"Generated {len(synthetic_entries)} synthetic sequence entries.")
    
    # Handle dry-run mode.
    dry_run = config.get("dry_run", False)
    if dry_run:
        logging.info("Dry run enabled. No output files will be written.")
        return
    
    # Write output: create a new latdnabatch subdirectory under sequences.
    output_dir = utils.create_output_directory(Path(__file__).parent.parent / "sequences", "latdnabatch")
    pt_output_path = output_dir / f"{output_dir.name}.pt"
    utils.write_pt_file(synthetic_entries, pt_output_path)
    logging.info(f"Wrote synthetic sequences to {pt_output_path}")
    
    # Write generation summary YAML.
    summary = {
        "latdna_batch": output_dir.name,
        "date_generated": utils.current_utc_timestamp(),
        "source_dense_batch": dense_batch_name,
        "sequence_length": seq_length,
        "fixed_elements": {
            "upstream_seq": upstream_seq,
            "upstream_start": upstream_start,
            "downstream_seq": downstream_seq,
            "downstream_start": downstream_start,
            "spacer_length": spacer_length,
        },
        "tiling_step": tiling_step,
        "tfbs_summary": {}
    }
    for entry in synthetic_entries:
        tf = entry["transcription_factor"]
        motif = entry["tiled_motif"]
        key = f"{tf}:{motif}"
        summary["tfbs_summary"][key] = summary["tfbs_summary"].get(key, 0) + 1
    
    summary_path = output_dir / "generation_summary.yaml"
    with summary_path.open("w") as f:
        yaml.dump(summary, f)
    logging.info(f"Wrote generation summary to {summary_path}")