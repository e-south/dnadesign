"""
--------------------------------------------------------------------------------
<dnadesign project>
dnadesign/densegen/main.py

This is the CLI entry point for densegen. It loads configuration,
ingests tf2tfbs_mapping data from input sources (defined under the 
input_sources key in the configuration), and then processes each source independently.
For each input source, binding sites are sampled and sequences are generated via
an optimizer. Each source (or sub‐batch when splitting by promoter constraint)
produces its own seqbatch output folder.

Module Author(s): Eric J. South
Dunlop Lab
--------------------------------------------------------------------------------
"""

import yaml
import random
import time
from pathlib import Path
import datetime
import pandas as pd
import torch
import copy

from dnadesign.utils import BASE_DIR, ConfigLoader, SequenceSaver, generate_sequence_entry
from dnadesign.densegen.data_ingestor import data_source_factory
from dnadesign.densegen.sampler import TFSampler
from dnadesign.densegen.optimizer_wrapper import DenseArrayOptimizer, random_fill
from dnadesign.densegen.progress_tracker import ProgressTracker


def select_solver(preferred: str, fallback: str, library: list, test_length: int = 10) -> tuple[str, list]:
    import dense_arrays as da
    try:
        dummy_optimizer = da.Optimizer(library=library, sequence_length=test_length)
        _ = dummy_optimizer.optimal(solver=preferred)
        print(f"Using solver: {preferred}")
        return preferred, []
    except Exception as e:
        print(f"Preferred solver {preferred} failed with error: {e}. Falling back to {fallback}.")
        return fallback, []


def _process_single_source(source_config: dict, densegen_config: dict, output_base_folder: Path, max_sequences: int = None):
    """
    Processes a single input source (or sub-batch when already split by promoter constraint).
    When max_sequences is provided (an integer), only that many new sequences are generated during
    this call (allowing round-robin interleaving). Otherwise the full quota is generated.
    """
    # Ensure a name is set.
    if "name" not in source_config or not source_config["name"]:
        source_config["name"] = Path(source_config["path"]).stem
    source_label = source_config["name"]
    print(f"\n=== Processing source: {source_label} ===")
    
    # Load data using the data source factory.
    source_obj = data_source_factory(source_config)
    data_entries, meta_data = source_obj.load_data()

    # Process based on source type.
    if source_config["type"].lower() == "pt":
        # PT data: expect data_entries to be a list of dicts with key "sequence"
        all_sequences = [entry["sequence"] for entry in data_entries if "sequence" in entry]
        if not all_sequences:
            raise ValueError(f"PT file for source {source_label} contains no sequences.")
        subsample_size = densegen_config.get("subsample_size", 10)
        library_for_optim = [seq.strip().upper() for seq in random.sample(all_sequences, min(subsample_size, len(all_sequences)))]
        meta_tfbs_parts = []  # No TF–TFBS pairing for PT data.
    elif isinstance(meta_data, pd.DataFrame) and not meta_data.empty:
        # CSV-based data: sample binding site pairs.
        sampler = TFSampler(meta_data)
        sampled_pairs = sampler.subsample_binding_sites(
            densegen_config.get("subsample_size", 10),
            unique_tf_only=densegen_config.get("unique_tf_only", False)
        )
        library_for_optim = [pair[1] for pair in sampled_pairs]
        # Now include the index in the motif library.
        meta_tfbs_parts = [f"idx_{idx}_{tf}_{tfbs}" for idx, (tf, tfbs, _) in enumerate(sampled_pairs)]
    else:
        raise ValueError(f"Expected CSV data for source {source_label} but got none.")

    # Get configuration options.
    preferred_solver = densegen_config.get("solver", "GUROBI")
    solver_options = densegen_config.get("solver_options", ["Threads=16"])
    sequence_length = densegen_config.get("sequence_length", 100)
    quota = densegen_config.get("quota", 5)
    arrays_generated_before_resample = densegen_config.get("arrays_generated_before_resample", 1)
    fixed_elements = densegen_config.get("fixed_elements", {})
    fill_gap = densegen_config.get("fill_gap", False)
    fill_gap_end = densegen_config.get("fill_gap_end", "5prime")
    fill_gc_min = densegen_config.get("fill_gc_min", 0.40)
    fill_gc_max = densegen_config.get("fill_gc_max", 0.60)
    
    # Build output folder and file names.
    out_folder_name = f"densebatch_{source_config['type'].lower()}_{source_label}_n{quota}"
    batch_folder = output_base_folder / out_folder_name
    batch_folder.mkdir(parents=True, exist_ok=True)
    results_filename = f"densegenbatch_{source_label}_n{quota}.pt"
    progress_file = batch_folder / f"progress_status_{source_label}.yaml"
    results_file = batch_folder / results_filename

    # Load or initialize progress.
    if progress_file.exists():
        with progress_file.open("r") as f:
            progress_status = yaml.safe_load(f)
        current_total = progress_status.get("total_entries", 0)
        print(f"Resuming from checkpoint for source {source_label}: {current_total} entries processed.")
        try:
            existing_results = torch.load(results_file)
        except Exception:
            existing_results = []
    else:
        progress_status = {
            "total_entries": 0,
            "target_quota": quota,
            "last_checkpoint": None,
            "error_flags": [],
            "system_resources": {},
            "config": {},
            "meta_gap_fill_used": False,
            "source": source_label
        }
        current_total = 0
        existing_results = []
    
    # Select the solver (which might fall back).
    selected_solver, extra_solver_options = select_solver(preferred_solver, "CBC", library_for_optim or [])
    densegen_config["solver"] = selected_solver
    if extra_solver_options:
        solver_options.extend(extra_solver_options)
    
    progress_tracker = ProgressTracker(str(progress_file))
    progress_tracker.update_batch_config(densegen_config, source_label)
    
    sequence_saver = SequenceSaver(str(batch_folder))
    generated_entries = existing_results[:]
    global_generated = current_total
    new_generated = 0  # Count how many new sequences are generated in this call.
    forbidden_libraries = set()
    max_forbidden_repeats = 1

    # Generation loop.
    while global_generated < quota and (max_sequences is None or new_generated < max_sequences):
        print(f"\nSource {source_label}: New TFBS library sample; generating up to {arrays_generated_before_resample} arrays from this set...")
        if source_config["type"].lower() != "pt":
            sampled_pairs = sampler.subsample_binding_sites(
                densegen_config.get("subsample_size", 10),
                unique_tf_only=densegen_config.get("unique_tf_only", False)
            )
            library_for_optim = [pair[1] for pair in sampled_pairs]
            meta_tfbs_parts = [f"idx_{idx}_{tf}_{tfbs}" for idx, (tf, tfbs, _) in enumerate(sampled_pairs)]
        else:
            subsample_size = densegen_config.get("subsample_size", 10)
            library_for_optim = [seq.strip().upper() for seq in random.sample(all_sequences, min(subsample_size, len(all_sequences)))]
        
        fp_library = tuple(sorted(library_for_optim))
        if arrays_generated_before_resample == 1 and fp_library in forbidden_libraries:
            print(f"Source {source_label}: This library has already been used. Resampling a new library.")
            continue

        optimizer_wrapper = DenseArrayOptimizer(
            library=library_for_optim,
            sequence_length=sequence_length,
            solver=selected_solver,
            solver_options=solver_options,
            fixed_elements=fixed_elements,
            fill_gap=fill_gap,
            fill_gap_end=fill_gap_end,
            fill_gc_min=fill_gc_min,
            fill_gc_max=fill_gc_max
        )
        opt_inst = optimizer_wrapper.get_optimizer_instance()
        local_generated = 0
        local_forbidden = set()
        forbidden_repeats = 0

        # Initialize error counter for time-limit errors
        max_time_error_repeats = 1
        time_error_repeats = 0

        while local_generated < arrays_generated_before_resample and global_generated < quota and (max_sequences is None or new_generated < max_sequences):
            start_time = time.time()
            try:
                solution = opt_inst.optimal(solver=selected_solver, solver_options=solver_options)
            except Exception as e:
                error_str = str(e)
                print(f"Source {source_label}: Optimization error: {error_str}. Retrying same library...")
                if "TimeLimit" in error_str or "time limit" in error_str or error_str.strip() == "":
                    time_error_repeats += 1
                    if time_error_repeats >= max_time_error_repeats:
                        print(f"Source {source_label}: Too many time-limit errors; moving to a new library sample.")
                        break
                continue

            time_error_repeats = 0
            elapsed_time = time.time() - start_time

            solution.original_visual = str(solution)
            fingerprint = solution.sequence
            if fingerprint in local_forbidden:
                forbidden_repeats += 1
                print(f"Source {source_label}: Duplicate solution encountered. Forbidden repeat count: {forbidden_repeats}")
                if forbidden_repeats >= max_forbidden_repeats:
                    print(f"Source {source_label}: Too many duplicates; moving to a new library sample.")
                    break
                continue
            local_forbidden.add(fingerprint)
            if fill_gap and len(solution.sequence) < sequence_length:
                gap = sequence_length - len(solution.sequence)
                fill_seq = random_fill(gap, fill_gc_min, fill_gc_max)
                if fill_gap_end.lower() == "5prime":
                    solution.sequence = fill_seq + solution.sequence
                else:
                    solution.sequence = solution.sequence + fill_seq
                setattr(solution, "meta_gap_fill", True)
                setattr(solution, "meta_gap_fill_details", {
                    "fill_gap": gap,
                    "fill_end": fill_gap_end,
                    "fill_gc_range": (fill_gc_min, fill_gc_max)
                })
            try:
                opt_inst.forbid(solution)
            except Exception as e:
                print(f"Source {source_label}: Warning: Could not forbid solution: {e}")
            entry = generate_sequence_entry(solution, [source_label], meta_tfbs_parts, densegen_config)
            entry["meta_source"] = f"deg2tfbs_{source_label}"
            generated_entries.append(entry)
            global_generated += 1
            new_generated += 1
            progress_tracker.update(entry, target_quota=quota)
            print(f"\nSource {source_label}: Generated sequence {global_generated}/{quota} (this call: {new_generated}) in {elapsed_time:.2f} sec.")
            print(f"Source {source_label}: TFBS parts used: {', '.join(meta_tfbs_parts)}")
            print(f"Source {source_label}: Meta Sequence Visual:\n{entry['meta_sequence_visual']}")
            local_generated += 1

        forbidden_libraries.add(fp_library)
        sequence_saver.save(generated_entries, results_filename)

    print(f"Source {source_label}: Dense array generation complete. Total sequences: {global_generated}.")


def process_source(source_config: dict, densegen_config: dict, output_base_folder: Path):
    """
    Processes a single input source. If the densegen configuration’s fixed_elements contain
    multiple promoter constraint definitions then we “split” the source into sub-batches --
    each using exactly one promoter constraint. Each sub-batch gets its own output folder
    and its own quota (as specified in the configuration).
    
    If round_robin (Boolean flag) is True, then sub-batches are interleaved so that one sequence
    is generated per sub-batch in cyclic order.
    """
    fixed_elements = densegen_config.get("fixed_elements", {})
    promoter_constraints = fixed_elements.get("promoter_constraints")
    if promoter_constraints and isinstance(promoter_constraints, list) and len(promoter_constraints) > 1:
        # Create a list of (source_config, densegen_config) pairs—one per promoter constraint.
        sub_batches = []
        for constraint in promoter_constraints:
            sub_source_config = copy.deepcopy(source_config)
            sub_densegen_config = copy.deepcopy(densegen_config)
            # Overwrite promoter_constraints with the single constraint.
            sub_densegen_config.setdefault("fixed_elements", {})["promoter_constraints"] = [constraint]
            # Append the constraint name to the source name.
            base_name = sub_source_config.get("name", Path(sub_source_config["path"]).stem)
            constraint_name = constraint.get("name", "constraint")
            sub_source_config["name"] = f"{base_name}_{constraint_name}"
            sub_batches.append((sub_source_config, sub_densegen_config))
        # If round-robin is NOT enabled, process each sub-batch to completion sequentially.
        if not densegen_config.get("round_robin", False):
            for sub_cfg, sub_dense_cfg in sub_batches:
                _process_single_source(sub_cfg, sub_dense_cfg, output_base_folder)
        else:
            # Round-robin: interleave sub-batches so each gets one sequence per call.
            all_done = False
            while not all_done:
                all_done = True
                for sub_cfg, sub_dense_cfg in sub_batches:
                    # Determine progress by reading the sub-batch’s progress file.
                    out_folder_name = f"densebatch_{sub_cfg['type'].lower()}_{sub_cfg['name']}_n{sub_dense_cfg.get('quota')}"
                    batch_folder = output_base_folder / out_folder_name
                    progress_file = batch_folder / f"progress_status_{sub_cfg['name']}.yaml"
                    current_count = 0
                    if progress_file.exists():
                        with progress_file.open("r") as f:
                            progress_data = yaml.safe_load(f)
                        current_count = progress_data.get("total_entries", 0)
                    if current_count < sub_dense_cfg.get("quota", 5):
                        all_done = False
                        # Generate just one new sequence in this call.
                        _process_single_source(sub_cfg, sub_dense_cfg, output_base_folder, max_sequences=1)
    else:
        # If there is only one (or no) promoter constraint defined, process normally.
        _process_single_source(source_config, densegen_config, output_base_folder)


def main():
    config_path = BASE_DIR / "src" / "dnadesign" / "configs" / "example.yaml"
    config_loader = ConfigLoader(config_path)
    densegen_config = config_loader.config  
    input_source_configs = densegen_config.get("input_sources", [])
    assert input_source_configs, "No input sources defined in configuration."
    
    output_base_folder = Path(__file__).parent.parent / densegen_config.get("output_dir", "sequences")
    output_base_folder.mkdir(parents=True, exist_ok=True)
    
    for src_cfg in input_source_configs:
        process_source(src_cfg, densegen_config, output_base_folder)
    
    print("\nAll input sources processed. Dense array generation complete.")


if __name__ == "__main__":
    main()
