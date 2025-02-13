"""
--------------------------------------------------------------------------------
<dnadesign project>
dnadesign/densegen/main.py

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

from dnadesign.utils import BASE_DIR, ConfigLoader, SequenceSaver, generate_sequence_entry
from dnadesign.densegen.data_ingestor import DEG2TFBSParser
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

def main():
    # Load configuration.
    config_path = BASE_DIR / "configs" / "example.yaml"
    config_loader = ConfigLoader(config_path)
    config = config_loader.config

    # Use a batch folder under "sequences" (inside the dnadesign directory).
    output_base_folder = Path(__file__).parent.parent / "sequences"
    output_base_folder.mkdir(parents=True, exist_ok=True)

    preferred_solver = config.get("solver", "CBC")
    solver_options = config.get("solver_options", ["Threads=16"])
    sequence_length = config.get("sequence_length", 100)
    quota = config.get("quota", 5)
    subsample_size = config.get("subsample_size", 15)
    fixed_elements = config.get("fixed_elements", {})
    unique_tf_only = config.get("unique_tf_only", False)
    fill_gap = config.get("fill_gap", False)
    fill_gap_end = config.get("fill_gap_end", "5prime")
    fill_gc_min = config.get("fill_gc_min", 0.40)
    fill_gc_max = config.get("fill_gc_max", 0.60)
    arrays_generated_before_resample = config.get("arrays_generated_before_resample", 1)
    source_names = config.get("sources", [])
    assert source_names, "No tf2tfbs mapping files defined in configuration."

    # Process each source independently.
    for src in source_names:
        print(f"\n=== Processing source: {src} ===")
        source_label = src.replace("tfbsbatch_", "")  # Trim prefix.
        date_str = datetime.datetime.now().strftime("%Y%m%d")
        results_filename = f"densegenbatch_{date_str}_{source_label}.pt"
        batch_folder = output_base_folder / f"densegenbatch_{date_str}_{source_label}"
        batch_folder.mkdir(parents=True, exist_ok=True)
        progress_file = batch_folder / f"progress_status_{source_label}.yaml"
        results_file = batch_folder / results_filename

        # Crash recovery.
        if progress_file.exists():
            with progress_file.open("r") as f:
                progress_status = yaml.safe_load(f)
            current_total = progress_status.get("total_entries", 0)
            print(f"Resuming from checkpoint for source {src}: {current_total} entries processed.")
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
                "source": ""
            }
            current_total = 0
            existing_results = []

        progress_tracker = ProgressTracker(str(progress_file))
        progress_tracker.update_batch_config(config, source_label)

        # Ingest data for this specific source.
        parser = DEG2TFBSParser(config.get("input_dir", "."))
        try:
            pairs, meta_df = parser.parse_tfbs_file(src)
        except AssertionError as ae:
            print(f"Error processing source {src}: {ae}. Skipping this source.")
            continue

        if meta_df.empty:
            print(f"No data found for source {src}. Skipping.")
            continue

        sampler = TFSampler(meta_df)
        sampled_pairs = sampler.subsample_binding_sites(subsample_size, unique_tf_only=unique_tf_only)
        library_for_optim = [pair[1] for pair in sampled_pairs]
        meta_tfbs_parts = [f"{tf}_{tfbs}" for tf, tfbs, _ in sampled_pairs]

        selected_solver, extra_solver_options = select_solver(preferred_solver, "CBC", library_for_optim)
        if extra_solver_options:
            solver_options.extend(extra_solver_options)

        sequence_saver = SequenceSaver(str(batch_folder))
        generated_entries = existing_results[:]  # Start with any loaded results.
        global_generated = current_total
        forbidden_libraries = set()
        max_forbidden_repeats = 5

        # Outer loop.
        while global_generated < quota:
            print(f"\nSource {src}: New TFBS library sample; generating up to {arrays_generated_before_resample} arrays...")
            sampled_pairs = sampler.subsample_binding_sites(subsample_size, unique_tf_only=unique_tf_only)
            library_for_optim = [pair[1] for pair in sampled_pairs]
            tfs_used = [pair[0] for pair in sampled_pairs]
            meta_tfbs_parts = [f"{tf}_{tfbs}" for tf, tfbs, _ in sampled_pairs]
            fp_library = tuple(sorted(library_for_optim))
            if arrays_generated_before_resample == 1 and fp_library in forbidden_libraries:
                print(f"Source {src}: This library has already been used. Resampling a new library.")
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

            # Inner loop.
            while local_generated < arrays_generated_before_resample and global_generated < quota:
                start_time = time.time()
                try:
                    solution = opt_inst.optimal(solver=selected_solver, solver_options=solver_options)
                except Exception as e:
                    print(f"Source {src}: Optimization error: {e}. Retrying same library...")
                    continue
                elapsed_time = time.time() - start_time

                fingerprint = solution.sequence
                if fingerprint in local_forbidden:
                    forbidden_repeats += 1
                    print(f"Source {src}: Duplicate solution encountered. Forbidden repeat count: {forbidden_repeats}")
                    if forbidden_repeats >= max_forbidden_repeats:
                        print(f"Source {src}: Too many duplicate solutions; moving to a new library sample.")
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
                    print(f"Source {src}: Warning: Could not forbid solution: {e}")

                entry = generate_sequence_entry(solution, [src], meta_tfbs_parts, config)
                entry["tfs_used"] = [f"{tf}_{tfbs}" for tf, tfbs, _ in sampled_pairs]
                entry["meta_source"] = f"deg2tfbs_{src}"
                generated_entries.append(entry)
                global_generated += 1
                progress_tracker.update(entry, target_quota=quota)
                print(f"\nSource {src}: Generated sequence {global_generated}/{quota} in {elapsed_time:.2f} seconds.")
                print(f"Source {src}: TFs used: {', '.join(tfs_used)}")
                print(f"Source {src}: Meta Sequence Visual:\n{entry['meta_sequence_visual']}")
                local_generated += 1

            forbidden_libraries.add(fp_library)
            sequence_saver.save(generated_entries, results_filename)

        print(f"Source {src}: Dense array generation complete. Total sequences generated: {global_generated}.")

    print("\nAll sources processed. Dense array generation complete.")

if __name__ == "__main__":
    main()
