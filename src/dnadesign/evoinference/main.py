"""
--------------------------------------------------------------------------------
<dnadesign project>
dnadesign/evoinference/main.py

Module Author(s): Eric J. South
Dunlop Lab
--------------------------------------------------------------------------------
"""

import os
import sys
import time
import torch
from config import load_config
from data_ingestion import list_pt_files, load_sequences, validate_sequence
from model_invocation import initialize_model, tokenize_sequence, run_model
from storage import write_results, update_progress, load_progress
from aggregations import augment_outputs
from logger import get_logger

logger = get_logger(__name__)

def process_file(filepath, model, output_types, overwrite, checkpoint_every):
    """
    Process a single .pt file:
      - Load sequences from the file.
      - Validate each sequence.
      - If a sequence is valid and not already processed (or if overwrite is True),
        tokenize the sequence and run it through the model.
      - Augment model outputs by adding tensor shape information and pooled values if configured.
      - Update the entry with all model outputs.
      - Save progress (checkpoint) every 'checkpoint_every' sequences.
      - Write the updated data back to the file.
    
    Returns a dictionary summarizing the progress for this file.
    """
    logger.info(f"Processing file: {filepath}")
    data = load_sequences(filepath)
    total_sequences = len(data)
    processed = 0
    skipped = 0

    # Determine the progress file path.
    dir_path = os.path.dirname(filepath)
    progress_files = [f for f in os.listdir(dir_path) if f.endswith('.yaml')]
    if progress_files:
        progress_filepath = os.path.join(dir_path, progress_files[0])
    else:
        progress_filepath = os.path.join(dir_path, "progress_status.yaml")

    progress_data = load_progress(progress_filepath)
    run_summary = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "model": model.__class__.__name__,
        "extracted": [],
        "total_sequences": total_sequences,
        "processed_sequences": 0,
        "skipped_sequences": 0,
        "failed_sequences": []
    }

    # Record which keys will be added (for logging purposes).
    for item in output_types:
        if item.get("type") == "logits":
            run_summary["extracted"].append("logits")
        if item.get("type") == "embeddings" and "layers" in item:
            for layer in item["layers"]:
                run_summary["extracted"].append(f"embeddings_{layer}")

    # Process each sequence entry.
    for idx, entry in enumerate(data):
        # Check if entry is already processed and skipping is desired.
        already_processed = False
        for key in run_summary["extracted"]:
            full_key = f"evo2_{key}"
            if not overwrite and full_key in entry and entry[full_key] is not None:
                already_processed = True
                break
        if already_processed:
            logger.info(f"Skipping entry {idx} in {filepath}: already processed.")
            skipped += 1
            continue

        if not validate_sequence(entry):
            logger.error(f"Skipping entry {idx} in {filepath}: invalid or missing sequence.")
            skipped += 1
            run_summary["failed_sequences"].append({
                "index": idx,
                "reason": "Invalid sequence format or missing 'sequence' key"
            })
            continue

        try:
            sequence = entry["sequence"]
            input_tensor = tokenize_sequence(model, sequence)
            # Run the model to get raw outputs.
            results = run_model(model, input_tensor, output_types)
            # Augment outputs with shape information and pooling if configured.
            results = augment_outputs(results, output_types)
            # Update entry with augmented model outputs.
            for key, value in results.items():
                if not overwrite and key in entry:
                    continue
                entry[key] = value
            processed += 1
        except Exception as e:
            logger.error(f"Error processing entry {idx} in {filepath}: {str(e)}")
            run_summary["failed_sequences"].append({
                "index": idx,
                "reason": str(e)
            })
            continue

        # Save progress every checkpoint_every processed sequences.
        if processed > 0 and processed % checkpoint_every == 0:
            run_summary["processed_sequences"] = processed
            run_summary["skipped_sequences"] = skipped
            update_progress(progress_filepath, run_summary)
            logger.info(f"Checkpoint: {processed} sequences processed in {filepath}.")

    # Final progress update.
    run_summary["processed_sequences"] = processed
    run_summary["skipped_sequences"] = skipped
    update_progress(progress_filepath, run_summary)
    write_results(filepath, data, overwrite)
    return run_summary

def main():
    # Assume the config file is located at '../configs/example.yaml' relative to this file.
    config_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'configs', 'example.yaml')
    try:
        config = load_config(config_path)
    except Exception as e:
        logger.error(f"Failed to load configuration: {str(e)}")
        sys.exit(1)

    data_sources = config.get("data_sources", [])
    evo_model_config = config.get("evo_model", {})
    model_version = evo_model_config.get("version", "evo2_7b")
    output_types = evo_model_config.get("output_types", [{"type": "logits"}])
    overwrite = config.get("overwrite_existing", False)
    checkpoint_every = config.get("checkpoint_every", 100)

    # Initialize the Evo model.
    try:
        model = initialize_model(model_version)
    except Exception as e:
        logger.error(f"Failed to initialize model: {str(e)}")
        sys.exit(1)

    # Process each data source directory. The directories are assumed to be under '../sequences'.
    base_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'sequences')
    overall_progress = {}
    for source in data_sources:
        dir_name = source.get("dir")
        if not dir_name:
            logger.error("Data source entry missing 'dir' key in config.")
            continue
        data_dir = os.path.join(base_dir, dir_name)
        try:
            pt_files = list_pt_files(data_dir)
        except Exception as e:
            logger.error(f"Error listing .pt files in {data_dir}: {str(e)}")
            continue

        for pt_file in pt_files:
            try:
                summary = process_file(pt_file, model, output_types, overwrite, checkpoint_every)
                overall_progress[pt_file] = summary
            except Exception as e:
                logger.error(f"Error processing file {pt_file}: {str(e)}")
                continue

    logger.info("Processing complete. Summary:")
    logger.info(overall_progress)

if __name__ == "__main__":
    main()
