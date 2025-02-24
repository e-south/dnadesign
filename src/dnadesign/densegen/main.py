"""
--------------------------------------------------------------------------------
<dnadesign project>
dnadesign/densegen/main.py

This is the CLI entry point for densegen. It loads configuration,
ingests tf2tfbs_mapping data from input sources (defined under the 
input_sources key in the configuration), and then processes each source independently.
For each input source, binding sites are sampled and sequences are generated via
an optimizer. Each source produces its own seqbatch output folder.

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


def select_solver(preferred: str, fallback: str, library: list, test_length: int = 10) -> tuple:
    import dense_arrays as da
    try:
        dummy_optimizer = da.Optimizer(library=library, sequence_length=test_length)
        _ = dummy_optimizer.optimal(solver=preferred)
        print(f"Using solver: {preferred}")
        return preferred, []
    except Exception as e:
        print(f"Preferred solver {preferred} failed with error: {e}. Falling back to {fallback}.")
        return fallback, []


def _process_single_source(source_config: dict, densegen_config: dict, output_base_folder: Path):
    """
    Processes a single subbatch (an input source with one promoter constraint)
    to completion (i.e. until its quota is reached). This function is used in sequential mode.
    """
    if "name" not in source_config or not source_config["name"]:
        source_config["name"] = Path(source_config["path"]).stem
    source_label = source_config["name"]
    print(f"\n=== Processing subbatch: {source_label} ===")
    
    # Load data from the source.
    source_obj = data_source_factory(source_config)
    data_entries, meta_data = source_obj.load_data()
    
    if source_config["type"].lower() == "pt":
        all_sequences = [entry["sequence"] for entry in data_entries if "sequence" in entry]
        if not all_sequences:
            raise ValueError(f"PT file for source {source_label} contains no sequences.")
        subsample_size = densegen_config.get("subsample_size", 10)
        library_for_optim = [seq.strip().upper() for seq in random.sample(all_sequences, min(subsample_size, len(all_sequences)))]
        meta_tfbs_parts = []
    elif isinstance(meta_data, pd.DataFrame) and not meta_data.empty:
        sampler = TFSampler(meta_data)
        sampled_pairs = sampler.subsample_binding_sites(
            densegen_config.get("subsample_size", 10),
            unique_tf_only=densegen_config.get("unique_tf_only", False)
        )
        library_for_optim = [pair[1] for pair in sampled_pairs]
        meta_tfbs_parts = [f"idx_{idx}_{tf}_{tfbs}" for idx, (tf, tfbs, _) in enumerate(sampled_pairs)]
    else:
        raise ValueError(f"Expected CSV data for source {source_label} but got none.")
    
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
    
    # Build output folder for this subbatch.
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
        print(f"Resuming subbatch {source_label}: {current_total} sequences already generated.")
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
    
    selected_solver, extra_solver_options = select_solver(preferred_solver, "CBC", library_for_optim or [])
    densegen_config["solver"] = selected_solver
    if extra_solver_options:
        solver_options.extend(extra_solver_options)
    
    progress_tracker = ProgressTracker(str(progress_file))
    progress_tracker.update_batch_config(densegen_config, source_label)
    
    sequence_saver = SequenceSaver(str(batch_folder))
    generated_entries = existing_results[:]
    global_generated = current_total
    forbidden_libraries = set()
    max_forbidden_repeats = 1
    
    # Generation loop for this subbatch.
    while global_generated < quota:
        print(f"Subbatch {source_label}: New TFBS library sample; generating up to {arrays_generated_before_resample} arrays.")
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
            print(f"Subbatch {source_label}: Library already used. Resampling...")
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
        
        try:
            solution = opt_inst.optimal(solver=selected_solver, solver_options=solver_options)
        except Exception as e:
            print(f"Subbatch {source_label}: Optimization error: {e}. Retrying library sample...")
            continue
        
        fingerprint = solution.sequence
        if fingerprint in local_forbidden:
            forbidden_repeats += 1
            if forbidden_repeats >= max_forbidden_repeats:
                print(f"Subbatch {source_label}: Too many duplicates. Resampling library.")
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
        except Exception:
            pass
        entry = generate_sequence_entry(solution, [source_label], meta_tfbs_parts, densegen_config)
        entry["meta_source"] = f"deg2tfbs_{source_label}"
        generated_entries.append(entry)
        global_generated += 1
        progress_tracker.update(entry, target_quota=quota)
        print(f"Subbatch {source_label}: Generated {global_generated}/{quota} sequences.")
        sequence_saver.save(generated_entries, results_filename)
    
    print(f"Subbatch {source_label} complete. Total sequences: {global_generated}.")


def generate_one_sequence(source_config: dict, densegen_config: dict, output_base_folder: Path):
    """
    A generator that produces one new sequence at a time for a given subbatch.
    This refactors the inner loop of _process_single_source to yield control after each sequence.
    """
    if "name" not in source_config or not source_config["name"]:
        source_config["name"] = Path(source_config["path"]).stem
    source_label = source_config["name"]
    print(f"\n=== Processing subbatch (round-robin): {source_label} ===")
    
    source_obj = data_source_factory(source_config)
    data_entries, meta_data = source_obj.load_data()
    
    if source_config["type"].lower() == "pt":
        all_sequences = [entry["sequence"] for entry in data_entries if "sequence" in entry]
        if not all_sequences:
            raise ValueError(f"PT file for source {source_label} contains no sequences.")
        subsample_size = densegen_config.get("subsample_size", 10)
        library_for_optim = [seq.strip().upper() for seq in random.sample(all_sequences, min(subsample_size, len(all_sequences)))]
        meta_tfbs_parts = []
    elif isinstance(meta_data, pd.DataFrame) and not meta_data.empty:
        sampler = TFSampler(meta_data)
        sampled_pairs = sampler.subsample_binding_sites(
            densegen_config.get("subsample_size", 10),
            unique_tf_only=densegen_config.get("unique_tf_only", False)
        )
        library_for_optim = [pair[1] for pair in sampled_pairs]
        meta_tfbs_parts = [f"idx_{idx}_{tf}_{tfbs}" for idx, (tf, tfbs, _) in enumerate(sampled_pairs)]
    else:
        raise ValueError(f"Expected CSV data for source {source_label} but got none.")
    
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
    
    out_folder_name = f"densebatch_{source_config['type'].lower()}_{source_label}_n{quota}"
    batch_folder = output_base_folder / out_folder_name
    batch_folder.mkdir(parents=True, exist_ok=True)
    results_filename = f"densegenbatch_{source_label}_n{quota}.pt"
    progress_file = batch_folder / f"progress_status_{source_label}.yaml"
    
    if progress_file.exists():
        with progress_file.open("r") as f:
            progress_status = yaml.safe_load(f)
        current_total = progress_status.get("total_entries", 0)
        try:
            existing_results = torch.load(batch_folder / results_filename)
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
    
    selected_solver, extra_solver_options = select_solver(preferred_solver, "CBC", library_for_optim or [])
    densegen_config["solver"] = selected_solver
    if extra_solver_options:
        solver_options.extend(extra_solver_options)
    
    progress_tracker = ProgressTracker(str(progress_file))
    progress_tracker.update_batch_config(densegen_config, source_label)
    
    sequence_saver = SequenceSaver(str(batch_folder))
    local_generated = current_total
    forbidden_libraries = set()
    max_forbidden_repeats = 1
    
    while local_generated < quota:
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
        local_forbidden = set()
        forbidden_repeats = 0
        try:
            solution = opt_inst.optimal(solver=selected_solver, solver_options=solver_options)
        except Exception:
            continue
        
        fingerprint = solution.sequence
        if fingerprint in local_forbidden:
            forbidden_repeats += 1
            if forbidden_repeats >= max_forbidden_repeats:
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
        except Exception:
            pass
        entry = generate_sequence_entry(solution, [source_label], meta_tfbs_parts, densegen_config)
        entry["meta_source"] = f"deg2tfbs_{source_label}"
        existing_results.append(entry)
        local_generated += 1
        progress_tracker.update(entry, target_quota=quota)
        sequence_saver.save(existing_results, results_filename)
        print(f"Subbatch {source_label} (round-robin): Generated {local_generated}/{quota} sequences.")
        yield  # Yield control after one sequence generation
    print(f"Subbatch {source_label} complete (round-robin). Total sequences: {local_generated}.")


def collect_subbatches(input_source_configs: list, densegen_config: dict) -> list:
    """
    For each input source config, if fixed_elements.promoter_constraints contains multiple
    constraints, create a separate subbatch for each constraint.
    Each subbatch is a tuple: (source_config, densegen_config).
    """
    subbatches = []
    all_prom_constraints = densegen_config.get("fixed_elements", {}).get("promoter_constraints", [])
    if not all_prom_constraints:
        for src_cfg in input_source_configs:
            subbatches.append((src_cfg, densegen_config))
    else:
        for src_cfg in input_source_configs:
            for pc in all_prom_constraints:
                sub_src = src_cfg.copy()
                constraint_name = pc.get("name", "default")
                sub_src["name"] = f"{src_cfg.get('name', Path(src_cfg['path']).stem)}_{constraint_name}"
                sub_densegen = copy.deepcopy(densegen_config)
                sub_densegen["fixed_elements"]["promoter_constraints"] = [pc]
                subbatches.append((sub_src, sub_densegen))
    return subbatches


def process_subbatches_round_robin(subbatches: list, output_base_folder: Path):
    """
    Processes all subbatches in round-robin mode. Each subbatch is advanced one sequence at a time
    in a cycle until every subbatch reaches its quota.
    """
    generators = []
    completed = [False] * len(subbatches)
    for sub_src, sub_densegen in subbatches:
        gen = generate_one_sequence(sub_src, sub_densegen, output_base_folder)
        generators.append(gen)
    
    still_running = True
    while still_running:
        still_running = False
        for idx, gen in enumerate(generators):
            if completed[idx]:
                continue
            try:
                next(gen)
                still_running = True
            except StopIteration:
                completed[idx] = True
        time.sleep(0.1)  # small delay to prevent tight loop


def main():
    config_path = BASE_DIR / "src" / "dnadesign" / "configs" / "example.yaml"
    config_loader = ConfigLoader(config_path)
    densegen_config = config_loader.config
    input_source_configs = densegen_config.get("input_sources", [])
    assert input_source_configs, "No input sources defined in configuration."
    
    output_base_folder = Path(__file__).parent.parent / densegen_config.get("output_dir", "sequences")
    output_base_folder.mkdir(parents=True, exist_ok=True)
    
    # Build subbatches: each (input source × promoter constraint) is its own subbatch.
    subbatches = collect_subbatches(input_source_configs, densegen_config)
    
    if densegen_config.get("round_robin", False):
        print("Running in round-robin mode.")
        process_subbatches_round_robin(subbatches, output_base_folder)
    else:
        print("Running in sequential mode.")
        for sub_src, sub_densegen in subbatches:
            _process_single_source(sub_src, sub_densegen, output_base_folder)
    
    print("\nAll subbatches processed. Dense array generation complete.")


if __name__ == "__main__":
    main()
