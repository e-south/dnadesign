# dnadesign/densegen/main.py
import yaml
import random
import time
from pathlib import Path
import datetime
import pandas as pd
import torch

from dnadesign.utils import BASE_DIR, ConfigLoader, SequenceSaver, generate_sequence_entry
from dnadesign.densegen.data_ingestor import DEG2TFBSParser  # use the parser directly for one source at a time
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

    input_dir = config.get("input_dir", ".")
    batch_base_folder = Path(__file__).parent / "batches"
    batch_base_folder.mkdir(parents=True, exist_ok=True)

    preferred_solver = config.get("solver", "CBC")
    solver_options = config.get("solver_options", ["Threads=16"])
    sequence_length = config.get("sequence_length", 100)
    quota = config.get("quota", 5)  # now interpreted as a per‐source quota
    subsample_size = config.get("subsample_size", 15)
    fixed_elements = config.get("fixed_elements", {})
    unique_tf_only = config.get("unique_tf_only", False)
    fill_gap = config.get("fill_gap", False)
    fill_gap_end = config.get("fill_gap_end", "5prime")
    fill_gc_min = config.get("fill_gc_min", 0.40)
    fill_gc_max = config.get("fill_gc_max", 0.60)
    arrays_generated_before_resample = config.get("arrays_generated_before_resample", 1)
    source_names = config.get("sources", [])
    assert source_names, "No sources defined in configuration."

    # Process each source independently.
    for src in source_names:
        print(f"\n=== Processing source: {src} ===")
        date_str = datetime.datetime.now().strftime("%Y-%m-%d")
        timestamp = int(time.time())
        # Create a batch folder for this source.
        batch_folder = batch_base_folder / f"seqbatch_{src}_{date_str}"
        batch_folder.mkdir(parents=True, exist_ok=True)
        progress_file = batch_folder / f"progress_status_{src}.yaml"
        results_filename = f"seqbatch_{src}_{date_str}_{timestamp}.pt"
        results_file = batch_folder / results_filename

        # Crash recovery: if a progress file exists, load checkpointed totals.
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
                "system_resources": {}
            }
            current_total = 0
            existing_results = []

        # Ingest data for this specific source.
        parser = DEG2TFBSParser(input_dir)
        try:
            pairs, meta_df = parser.parse_tfbs_file(src)
        except AssertionError as ae:
            print(f"Error processing source {src}: {ae}. Skipping this source.")
            continue

        if meta_df.empty:
            print(f"No data found for source {src}. Skipping.")
            continue

        sampler = TFSampler(meta_df)
        # Initial sampling for solver selection.
        sampled_pairs = sampler.subsample_binding_sites(subsample_size, unique_tf_only=unique_tf_only)
        library_for_optim = [pair[1] for pair in sampled_pairs]
        tfs_sample = [pair[0] for pair in sampled_pairs]
        meta_tfbs_parts = [f"{tf}_{tfbs}" for tf, tfbs, _ in sampled_pairs]

        selected_solver, extra_solver_options = select_solver(preferred_solver, "CBC", library_for_optim)
        if extra_solver_options:
            solver_options.extend(extra_solver_options)

        progress_tracker = ProgressTracker(str(progress_file))
        sequence_saver = SequenceSaver(str(batch_folder))

        generated_entries = existing_results[:]  # start with any loaded results
        global_generated = current_total
        forbidden_libraries = set()
        max_forbidden_repeats = 5

        # Outer loop: continue until the per‐source quota is met.
        while global_generated < quota:
            print(f"\nSource {src}: New tfbs library sample; generating up to {arrays_generated_before_resample} arrays...")
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

            # Inner loop: generate arrays from the current library sample.
            while local_generated < arrays_generated_before_resample and global_generated < quota:
                start_time = time.time()
                try:
                    solution = opt_inst.optimal(solver=selected_solver, solver_options=solver_options)
                except Exception as e:
                    print(f"Source {src}: Optimization error: {e}. Retrying same library...")
                    continue
                elapsed_time = time.time() - start_time

                fingerprint = solution.sequence  # use the sequence string as a fingerprint
                if fingerprint in local_forbidden:
                    forbidden_repeats += 1
                    print(f"Source {src}: Duplicate solution encountered. Forbidden repeat count: {forbidden_repeats}")
                    if forbidden_repeats >= max_forbidden_repeats:
                        print(f"Source {src}: Too many duplicate solutions; moving to a new library sample.")
                        break
                    continue
                local_forbidden.add(fingerprint)
                # Apply gap fill if needed.
                sol_seq = str(solution)
                if fill_gap and len(sol_seq) < sequence_length:
                    gap = sequence_length - len(sol_seq)
                    fill_seq = random_fill(gap, fill_gc_min, fill_gc_max)
                    if fill_gap_end.lower() == "5prime":
                        sol_seq = fill_seq + sol_seq
                    else:
                        sol_seq = sol_seq + fill_seq
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

                # Build the entry.
                entry = generate_sequence_entry(solution, [src], meta_tfbs_parts)
                entry["tfs_used"] = [f"{tf}_{tfbs}" for tf, tfbs, _ in sampled_pairs]
                entry["meta_source"] = f"deg2tfbs_{src}"
                entry["meta_nb_motifs"] = getattr(solution, "nb_motifs", None)
                entry["meta_gap_fill"] = getattr(solution, "meta_gap_fill", False)
                entry["meta_gap_fill_details"] = getattr(solution, "meta_gap_fill_details", None)
                entry["meta_offsets"] = (solution.offset_indices_in_order() if hasattr(solution, "offset_indices_in_order") else None)
                entry["meta_compression_ratio"] = getattr(solution, "compression_ratio", None)
                entry["sequence"] = solution.sequence

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
