"""
--------------------------------------------------------------------------------
<dnadesign project>
dnadesign/densegen/main.py

CLI entry point for densegen.
Loads configuration, ingests TF-TFBS mapping data, and processes each input source (or sub‐batch).
For each source, binding sites are sampled (using a target basepair budget),
the solver is instantiated, and then dense arrays are generated.
The solver can be run in a diversity-driven mode (if flagged in the config).

Module Author(s): Eric J. South
Dunlop Lab
--------------------------------------------------------------------------------
"""

import copy
import random
from pathlib import Path

import pandas as pd
import torch
import yaml

from dnadesign.densegen.data_ingestor import data_source_factory
from dnadesign.densegen.optimizer_wrapper import DenseArrayOptimizer, random_fill
from dnadesign.densegen.progress_tracker import ProgressTracker
from dnadesign.densegen.sampler import TFSampler
from dnadesign.utils import BASE_DIR, ConfigLoader, SequenceSaver, generate_sequence_entry


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


def _process_single_source(
    source_config: dict, densegen_config: dict, output_base_folder: Path, max_sequences: int = None
):
    """
    Processes a single input source (or sub‐batch).
    Samples binding sites using a length-budget approach,
    instantiates the solver, and then generates sequences.
    Uses the 'diverse_solution' flag in the config to choose between
    diversity-driven (solutions_diverse) and standard optimal methods.
    """
    # Ensure a name is set.
    if "name" not in source_config or not source_config["name"]:
        source_config["name"] = Path(source_config["path"]).stem
    source_label = source_config["name"]
    print(f"\n=== Processing source: {source_label} ===")

    # Load data using the data source factory.
    source_obj = data_source_factory(source_config)
    data_entries, meta_data = source_obj.load_data()

    if source_config["type"].lower() == "pt":
        all_sequences = [entry["sequence"] for entry in data_entries if "sequence" in entry]
        if not all_sequences:
            raise ValueError(f"PT file for source {source_label} contains no sequences.")
        subsample_size = densegen_config.get("subsample_size", 10)
        library_for_optim = [
            seq.strip().upper() for seq in random.sample(all_sequences, min(subsample_size, len(all_sequences)))
        ]
        meta_tfbs_parts = []  # No TF–TFBS pairing for PT data.
    elif isinstance(meta_data, pd.DataFrame) and not meta_data.empty:
        sampler = TFSampler(meta_data)
        sequence_length = densegen_config.get("sequence_length", 100)
        budget_overhead = densegen_config.get("subsample_over_length_budget_by", 30)
        library_for_optim, meta_tfbs_parts = sampler.generate_binding_site_subsample(sequence_length, budget_overhead)
    else:
        raise ValueError(f"Expected CSV data for source {source_label} but got none.")

    print(f"Feeding {len(library_for_optim)} binding sites to the solver for source {source_label}.")

    # Retrieve configuration options.
    preferred_solver = densegen_config.get("solver", "GUROBI")
    solver_options = densegen_config.get("solver_options", ["Threads=16"])
    sequence_length = densegen_config.get("sequence_length", 100)
    quota = densegen_config.get("quota", 5)
    # Consolidate subsample quota to the user-defined arrays_generated_before_resample key.
    max_solutions_per_subsample = densegen_config.get(
        "max_solutions_per_subsample", densegen_config.get("arrays_generated_before_resample", 1)
    )
    fixed_elements = densegen_config.get("fixed_elements", {})
    fill_gap = densegen_config.get("fill_gap", False)
    fill_gap_end = densegen_config.get("fill_gap_end", "5prime")
    fill_gc_min = densegen_config.get("fill_gc_min", 0.40)
    fill_gc_max = densegen_config.get("fill_gc_max", 0.60)
    use_diverse_solver = densegen_config.get("diverse_solution", False)

    out_folder_name = f"densebatch_{source_config['type'].lower()}_{source_label}_n{quota}"
    batch_folder = output_base_folder / out_folder_name
    batch_folder.mkdir(parents=True, exist_ok=True)
    results_filename = f"densegenbatch_{source_label}_n{quota}.pt"
    progress_file = batch_folder / f"progress_status_{source_label}.yaml"
    results_file = batch_folder / results_filename

    if progress_file.exists():
        with progress_file.open("r") as f:
            progress_status = yaml.safe_load(f)
        current_total = progress_status.get("total_entries", 0)
        print(f"Resuming from checkpoint for source {source_label}: {current_total} entries processed.")
        try:
            existing_results = torch.load(results_file, weights_only=True)
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
            "source": source_label,
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
    new_generated = 0
    forbidden_libraries = set()
    max_forbidden_repeats = 1

    while global_generated < quota and (max_sequences is None or new_generated < max_sequences):
        print(
            f"\nSource {source_label}: New TFBS library sample; generating up to {max_solutions_per_subsample} arrays from this set..."
        )
        if source_config["type"].lower() != "pt":
            sampler = TFSampler(meta_data)
            budget_overhead = densegen_config.get("subsample_over_length_budget_by", 30)
            library_for_optim, meta_tfbs_parts = sampler.generate_binding_site_subsample(
                sequence_length, budget_overhead
            )
        else:
            subsample_size = densegen_config.get("subsample_size", 10)
            library_for_optim = [
                seq.strip().upper() for seq in random.sample(all_sequences, min(subsample_size, len(all_sequences)))
            ]

        print(f"Subsample contains {len(library_for_optim)} binding sites.")
        fp_library = tuple(sorted(library_for_optim))
        if max_solutions_per_subsample == 1 and fp_library in forbidden_libraries:
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
            fill_gc_max=fill_gc_max,
        )
        opt_inst = optimizer_wrapper.get_optimizer_instance()
        local_generated = 0
        local_forbidden = set()
        forbidden_repeats = 0

        # Get the solver instance from the optimizer wrapper.
        opt_inst = optimizer_wrapper.get_optimizer_instance()

        # Choose a solution generator based on the diverse_solution flag.
        if use_diverse_solver:
            sol_generator = opt_inst.solutions_diverse(solver=selected_solver, solver_options=solver_options)
        else:
            sol_generator = opt_inst.solutions(solver=selected_solver, solver_options=solver_options)

        # Initialize counters and duplicate tracking.
        local_generated = 0
        consecutive_duplicates = 0
        max_consecutive_duplicates = 3  # Allow up to 3 consecutive duplicates
        local_forbidden = set()

        # Iterate over the chosen solution generator.
        for sol in sol_generator:
            # Capture the original visual representation before any gap fill changes.
            sol.original_visual = str(sol)
            tf_names = [p.split(":")[0] for p in meta_tfbs_parts]
            current_global = global_generated + 1
            global_progress_pct = (current_global / quota) * 100
            subsample_progress_pct = ((local_generated + 1) / max_solutions_per_subsample) * 100

            # Forbid the original solution (which is all uppercase) for duplicate checking.
            opt_inst.forbid(sol)
            fingerprint = sol.sequence

            if fingerprint in local_forbidden:
                consecutive_duplicates += 1
                print(
                    f"Source {source_label}: Duplicate solution encountered; consecutive duplicates: {consecutive_duplicates}"
                )
                if consecutive_duplicates >= max_consecutive_duplicates:
                    print(f"Source {source_label}: Too many consecutive duplicates; moving to a new library sample.")
                    break
                continue
            consecutive_duplicates = 0
            local_forbidden.add(fingerprint)

            # Clone the solution and apply gap fill modifications on the clone.
            final_sol = copy.deepcopy(sol)
            if fill_gap and len(final_sol.sequence) < sequence_length:
                gap = sequence_length - len(final_sol.sequence)
                fill_seq = random_fill(gap, fill_gc_min, fill_gc_max)
                if fill_gap_end.lower() == "5prime":
                    final_sol.sequence = fill_seq + final_sol.sequence
                else:
                    final_sol.sequence = final_sol.sequence + fill_seq
                setattr(final_sol, "meta_gap_fill", True)
                setattr(
                    final_sol,
                    "meta_gap_fill_details",
                    {"fill_gap": gap, "fill_end": fill_gap_end, "fill_gc_range": (fill_gc_min, fill_gc_max)},
                )

            entry = generate_sequence_entry(final_sol, [source_label], meta_tfbs_parts, densegen_config)
            entry["meta_source"] = f"deg2tfbs_{source_label}"
            generated_entries.append(entry)
            global_generated += 1
            new_generated += 1
            progress_tracker.update(entry, target_quota=quota)
            local_generated += 1

            if local_generated >= max_solutions_per_subsample:
                break

            print(f"\nGlobal Quota Progress: {current_global}/{quota} ({global_progress_pct:.1f}%)")
            print(
                f"Subsample Progress: {local_generated+1}/{max_solutions_per_subsample} ({subsample_progress_pct:.1f}%)"
            )
            print("Compression Ratio:", sol.compression_ratio)
            print("Transcription Factors:", ", ".join(tf_names))
            print("Dense Visual:\n", sol.original_visual)
            print("Sequence:\n", final_sol.sequence)

        forbidden_libraries.add(fp_library)
        sequence_saver.save(generated_entries, results_filename)

    print(f"Source {source_label}: Dense array generation complete. Total sequences: {global_generated}.")


def get_sub_batches(source_config: dict, densegen_config: dict) -> list:
    """
    Splits an input source into sub-batches based on clusters and/or promoter constraints.
    """
    sub_batches = []
    clusters = source_config.get("clusters")
    fixed_elements = densegen_config.get("fixed_elements", {})
    promoter_constraints = fixed_elements.get("promoter_constraints")

    if clusters and isinstance(clusters, list) and len(clusters) > 0:
        if promoter_constraints and isinstance(promoter_constraints, list) and len(promoter_constraints) > 1:
            for constraint in promoter_constraints:
                for cluster in clusters:
                    sub_source_config = copy.deepcopy(source_config)
                    sub_densegen_config = copy.deepcopy(densegen_config)
                    sub_source_config["clusters"] = [cluster]
                    base_name = sub_source_config.get("name", Path(sub_source_config["path"]).stem)
                    constraint_name = constraint.get("name", "constraint")
                    sub_source_config["name"] = f"{base_name}_{cluster}_{constraint_name}"
                    sub_densegen_config.setdefault("fixed_elements", {})["promoter_constraints"] = [constraint]
                    sub_batches.append((sub_source_config, sub_densegen_config))
        elif promoter_constraints and isinstance(promoter_constraints, list) and len(promoter_constraints) > 0:
            for cluster in clusters:
                sub_source_config = copy.deepcopy(source_config)
                sub_densegen_config = copy.deepcopy(densegen_config)
                sub_source_config["clusters"] = [cluster]
                base_name = sub_source_config.get("name", Path(sub_source_config["path"]).stem)
                sub_source_config["name"] = f"{base_name}_{cluster}"
                sub_batches.append((sub_source_config, sub_densegen_config))
        else:
            for cluster in clusters:
                sub_source_config = copy.deepcopy(source_config)
                sub_densegen_config = copy.deepcopy(densegen_config)
                sub_source_config["clusters"] = [cluster]
                base_name = sub_source_config.get("name", Path(sub_source_config["path"]).stem)
                sub_source_config["name"] = f"{base_name}_{cluster}"
                sub_batches.append((sub_source_config, sub_densegen_config))
    else:
        if promoter_constraints and isinstance(promoter_constraints, list) and len(promoter_constraints) > 1:
            for constraint in promoter_constraints:
                sub_source_config = copy.deepcopy(source_config)
                sub_densegen_config = copy.deepcopy(densegen_config)
                base_name = sub_source_config.get("name", Path(sub_source_config["path"]).stem)
                constraint_name = constraint.get("name", "constraint")
                sub_source_config["name"] = f"{base_name}_{constraint_name}"
                sub_densegen_config.setdefault("fixed_elements", {})["promoter_constraints"] = [constraint]
                sub_batches.append((sub_source_config, sub_densegen_config))
        else:
            sub_batches.append((source_config, densegen_config))
    print("Generated sub-batches:")
    for cfg, dens in sub_batches:
        print("  ", cfg["name"], " (type:", cfg["type"], ")")
    return sub_batches


def process_source(source_config: dict, densegen_config: dict, output_base_folder: Path):
    """
    Processes a single input source by splitting it into sub-batches.
    """
    sub_batches = get_sub_batches(source_config, densegen_config)

    # Pre-create all output directories for sub-batches.
    for sub_cfg, sub_dense_cfg in sub_batches:
        out_folder_name = f"densebatch_{sub_cfg['type'].lower()}_{sub_cfg['name']}_n{sub_dense_cfg.get('quota')}"
        batch_folder = output_base_folder / out_folder_name
        batch_folder.mkdir(parents=True, exist_ok=True)

    if not densegen_config.get("round_robin", False):
        for sub_cfg, sub_dense_cfg in sub_batches:
            _process_single_source(sub_cfg, sub_dense_cfg, output_base_folder)
    else:
        # Round-robin mode: interleave sub-batches from this input source.
        all_done = False
        while not all_done:
            all_done = True
            for sub_cfg, sub_dense_cfg in sub_batches:
                out_folder_name = (
                    f"densebatch_{sub_cfg['type'].lower()}_{sub_cfg['name']}_n{sub_dense_cfg.get('quota')}"
                )
                batch_folder = output_base_folder / out_folder_name
                progress_file = batch_folder / f"progress_status_{sub_cfg['name']}.yaml"
                current_count = 0
                if progress_file.exists():
                    with progress_file.open("r") as f:
                        progress_data = yaml.safe_load(f)
                    current_count = progress_data.get("total_entries", 0)
                if current_count < sub_dense_cfg.get("quota", 5):
                    all_done = False
                    _process_single_source(sub_cfg, sub_dense_cfg, output_base_folder, max_sequences=1)


def main():
    config_path = BASE_DIR / "src" / "dnadesign" / "configs" / "example.yaml"
    config_loader = ConfigLoader(config_path)
    densegen_config = config_loader.config
    input_source_configs = densegen_config.get("input_sources", [])
    assert input_source_configs, "No input sources defined in configuration."

    output_base_folder = Path(__file__).parent.parent / densegen_config.get("output_dir", "sequences")
    output_base_folder.mkdir(parents=True, exist_ok=True)

    if densegen_config.get("round_robin", False):
        # For round-robin, collect sub-batches from all input sources.
        all_sub_batches = []
        for src_cfg in input_source_configs:
            sub_batches = get_sub_batches(copy.deepcopy(src_cfg), copy.deepcopy(densegen_config))
            all_sub_batches.extend(sub_batches)
        # Pre-create output directories for all sub-batches.
        for sub_cfg, sub_dense_cfg in all_sub_batches:
            out_folder_name = f"densebatch_{sub_cfg['type'].lower()}_{sub_cfg['name']}_n{sub_dense_cfg.get('quota')}"
            batch_folder = output_base_folder / out_folder_name
            batch_folder.mkdir(parents=True, exist_ok=True)
        # Interleave across all sub-batches.
        all_done = False
        while not all_done:
            all_done = True
            for sub_cfg, sub_dense_cfg in all_sub_batches:
                out_folder_name = (
                    f"densebatch_{sub_cfg['type'].lower()}_{sub_cfg['name']}_n{sub_dense_cfg.get('quota')}"
                )
                batch_folder = output_base_folder / out_folder_name
                progress_file = batch_folder / f"progress_status_{sub_cfg['name']}.yaml"
                current_count = 0
                if progress_file.exists():
                    with progress_file.open("r") as f:
                        progress_data = yaml.safe_load(f)
                    current_count = progress_data.get("total_entries", 0)
                if current_count < sub_dense_cfg.get("quota", 5):
                    all_done = False
                    _process_single_source(sub_cfg, sub_dense_cfg, output_base_folder, max_sequences=1)
    else:
        # Process each input source separately.
        for src_cfg in input_source_configs:
            process_source(copy.deepcopy(src_cfg), copy.deepcopy(densegen_config), output_base_folder)

    print("\nAll input sources processed. Dense array generation complete.")


if __name__ == "__main__":
    main()
