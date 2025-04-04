"""
--------------------------------------------------------------------------------
<dnadesign project>
libshuffle/output.py

Provides helper functions to generate global summary and sublibrary YAML files,
and optionally save selected subsample .pt files.

Module Author(s): Eric J. South
Dunlop Lab
--------------------------------------------------------------------------------
"""

import os
import yaml
import torch
import datetime
import platform
from pathlib import Path

def save_summary(output_dir, config, input_file, sequence_batch_id, num_sequences, run_start_time, duration):
    summary = {
        "run_time": run_start_time.isoformat(),
        "duration_seconds": duration,
        "libshuffle_config": config,
        "evo2_vector_key": "evo2_logits_mean_pooled",
        "input_pt_file": input_file.name,
        "sequence_batch_id": sequence_batch_id,
        "num_sequences": num_sequences,
        "python_version": platform.python_version()
    }
    summary_path = Path(output_dir) / "global_summary.yaml"
    with open(summary_path, "w") as f:
        yaml.dump(summary, f)
    return summary_path

def save_results(output_dir, subsamples, pca_model_info):
    results = {}
    for s in subsamples:
        results[s["subsample_id"]] = {
            "selected_indices": s["indices"],
            "selected_ids": s["selected_ids"],
            "billboard_metric": s["billboard_metric"],
            "evo2_metric": s["evo2_metric"],
            "passed_threshold": s.get("passed_threshold", False),
            "raw_billboard_vector": s.get("raw_billboard_vector")
        }
    output_dict = {
        "pca_model_info": pca_model_info,
        "sublibraries": results
    }
    results_path = Path(output_dir) / "sublibraries.yaml"
    with open(results_path, "w") as f:
        yaml.dump(output_dict, f)
    return results_path

def save_selected_subsamples(subsamples, config, original_sequences, base_output_dir):
    """
    Saves the selected subsamples (as specified by joint_selection.selected_subsample_ids)
    as new .pt files back into the input sequences directory, within a subdirectory
    named "shufflebatch_YYYYMMDD" where YYYYMMDD is the current date.
    """
    from pathlib import Path
    import datetime
    import torch

    joint_sel = config.get("joint_selection", {})
    selected_ids = joint_sel.get("selected_subsample_ids", [])
    if not selected_ids:
        return []

    # Filter subsamples matching the specified IDs.
    selected = [s for s in subsamples if s.get("subsample_id") in selected_ids]

    # Create a subdirectory with date stamp in the base_output_dir.
    date_stamp = datetime.datetime.now().strftime("%Y%m%d")
    batch_subdir = Path(base_output_dir) / f"shufflebatch_{date_stamp}"
    batch_subdir.mkdir(parents=True, exist_ok=True)

    saved_files = []
    for s in selected:
        subsample_sequences = [original_sequences[i] for i in s["indices"]]
        filename = f"{s['subsample_id']}.pt"
        output_path = batch_subdir / filename
        torch.save(subsample_sequences, output_path)
        saved_files.append(str(output_path))
    return saved_files