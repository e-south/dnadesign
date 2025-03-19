"""
--------------------------------------------------------------------------------
<dnadesign project>
/densehairpins/main.py

Module Author(s): Eric J. South
Dunlop Lab
--------------------------------------------------------------------------------
"""

import os
import shutil
import random
import datetime
from pathlib import Path
import pandas as pd
import yaml

from dnadesign.utils import BASE_DIR, DATA_FILES
from dnadesign.densehairpins.config_loader import ConfigLoader
from dnadesign.densehairpins.data_handler import load_pancardo_dataset, save_intermediate_csv
from dnadesign.densehairpins.parser import parse_meme_file
from dnadesign.densehairpins.solver import run_solver_iteratively, save_solver_output
from dnadesign.densehairpins.scoring import add_scores_to_solutions
from dnadesign.densehairpins.visualization import save_ranked_csv, plot_scatter

def create_batch_folder(base_dir, batch_name, batch_label=None, use_timestamp=True, overwrite=True):
    # If use_timestamp is False, then we don't append a timestamp.
    if use_timestamp:
        timestamp = datetime.datetime.now().strftime("%Y%m%d")
        folder_name = f"{batch_name}_{batch_label}_{timestamp}" if batch_label else f"{batch_name}_{timestamp}"
    else:
        folder_name = f"{batch_name}_{batch_label}" if batch_label else batch_name
    batch_folder = Path(base_dir) / folder_name
    if batch_folder.exists() and overwrite:
        shutil.rmtree(batch_folder)
    batch_folder.mkdir(parents=True, exist_ok=True)
    for sub in ["csvs", "plots"]:
        (batch_folder / sub).mkdir(exist_ok=True)
    return batch_folder

def deduplicate_entries(entries: list) -> list:
    raw_count = len(entries)
    unique_entries = {(entry['TF'].strip(), entry['sequence'].strip()): entry for entry in entries}
    unique_list = list(unique_entries.values())
    unique_count = len(unique_list)
    assert unique_count > 0, "No unique binding site entries found after deduplication."
    if raw_count != unique_count:
        print(f"Note: Deduplicated {raw_count - unique_count} duplicate entries; using {unique_count} unique binding site entries for library building.")
    return unique_list

def sample_library(unique_list: list, subsample_size: int = None) -> list:
    unique_count = len(unique_list)
    if subsample_size:
        if subsample_size > unique_count:
            print(f"Warning: subsample_size ({subsample_size}) is larger than available unique entries ({unique_count}). Using all available entries.")
            return unique_list
        return random.sample(unique_list, subsample_size)
    return unique_list

def save_solver_csv_iteratively(solutions, output_file):
    """
    Save the current solutions list to CSV.
    If output_file exists, load existing rows and merge with new ones (avoiding duplicates).
    """
    df_new = pd.DataFrame(solutions)
    if output_file.exists():
        df_existing = pd.read_csv(output_file)
        # Merge by entry_id (assuming uniqueness)
        df_merged = pd.concat([df_existing, df_new]).drop_duplicates(subset=["entry_id"])
    else:
        df_merged = df_new
    df_merged.to_csv(output_file, index=False)
    print(f"Iterative solver output updated at {output_file}")

def main():
    random_seed = 42
    random.seed(random_seed)

    config_path = Path(__file__).resolve().parent.parent / "configs/example.yaml"
    config_loader = ConfigLoader(config_path)
    config = config_loader.config

    consensus_flag = config.get("consensus_only", True)
    run_post_solve = config.get("run_post_solve", False)
    # If run_post_solve is True, preserve existing folder (do not overwrite).
    overwrite_folder = False if run_post_solve else True

    batch_results_dir = Path(__file__).resolve().parent / "batch_results"
    batch_results_dir.mkdir(exist_ok=True)
    batch_name = config.get("batch_name", "default_batch")
    batch_label = config.get("batch_label", None)
    use_timestamp = False if run_post_solve else True
    batch_folder = create_batch_folder(batch_results_dir, batch_name, batch_label, use_timestamp=use_timestamp, overwrite=not run_post_solve)

    pancardo_df = load_pancardo_dataset()

    omalley_dir = DATA_FILES["omalley_et_al"]
    if not omalley_dir.exists():
        raise FileNotFoundError(f"O'Malley directory not found: {omalley_dir}")

    tidy_rows = []
    total_matches = 0
    total_possible = len(pancardo_df)

    # Pre-index available motif files.
    motif_files = {f.stem.lower(): f for f in omalley_dir.glob("*.txt")}

    for idx, row in pancardo_df.iterrows():
        tf = row["TF"]
        file_path = motif_files.get(tf.lower())
        if file_path is None or not file_path.exists():
            print(f"Warning: File for TF {tf} not found. Skipping.")
            continue

        parsed = parse_meme_file(file_path, consensus_only=consensus_flag)
        if parsed:
            if consensus_flag:
                consensus = parsed
                others = []
            else:
                consensus = parsed.get("consensus")
                others = parsed.get("others", [])
            if consensus:
                tidy_rows.append({
                    "TF": tf,
                    "sequence": consensus,
                    "type": "consensus",
                    "Silenced_Genes": row["Silenced Genes"],
                    "Induced_Genes": row["Induced Genes"],
                    "Rank": row["Rank"]
                })
            for site in others:
                tidy_rows.append({
                    "TF": tf,
                    "sequence": site["sequence"],
                    "type": site["type"],
                    "Silenced_Genes": row["Silenced Genes"],
                    "Induced_Genes": row["Induced Genes"],
                    "Rank": row["Rank"]
                })
            total_matches += 1

    percentage = (total_matches / total_possible) * 100
    print(f"Found binding sites for {percentage:.2f}% of TFs in Pancardo dataset.")

    intermediate_csv_path = batch_folder / "csvs" / "intermediate.csv"
    pd.DataFrame(tidy_rows).to_csv(intermediate_csv_path, index=False)

    if consensus_flag:
        entries_for_library = [entry for entry in tidy_rows if entry["type"] == "consensus" and entry["sequence"]]
    else:
        entries_for_library = [entry for entry in tidy_rows if entry["sequence"]]
    if not entries_for_library:
        raise ValueError("No binding site sequences found to build the library.")

    unique_entries = deduplicate_entries(entries_for_library)
    subsample_size = config.get("subsample_size", None)
    sampled_data = sample_library(unique_entries, subsample_size)
    library = [{"TF": entry["TF"], "sequence": entry["sequence"]} for entry in sampled_data]

    quota = config.get("quota", 10)
    solver = config.get("solver", "CBC")
    solver_options = config.get("solver_options", [])
    sequence_length = config.get("sequence_length", 30)
    subsample_size = config.get("subsample_size", None)
    random_subsample_per_solve = config.get("random_subsample_per_solve", False)
    solver_output_file = batch_folder / "csvs" / "solver_output.csv"

    if run_post_solve and solver_output_file.exists():
        print("run_post_solve is true and existing solver output found. Skipping solver.")
        solution_dicts = pd.read_csv(solver_output_file).to_dict(orient="records")
    else:
        def save_callback(solutions):
            save_solver_csv_iteratively(solutions, solver_output_file)
        solution_dicts = run_solver_iteratively(library, sequence_length, solver, solver_options, quota,
                                                  save_callback=save_callback,
                                                  random_subsample_per_solve=random_subsample_per_solve,
                                                  subsample_size=subsample_size)

    # Compute scores, etc.
    score_weights = config.get("score_weights", {"silenced_genes": 1, "induced_genes": 1})
    solution_dicts = add_scores_to_solutions(solution_dicts, pancardo_df, score_weights)

    # Save final CSV outputs.
    solver_df = pd.DataFrame(solution_dicts)
    solver_df.to_csv(solver_output_file, index=False)
    ranked_csv_path = batch_folder / "csvs" / "ranked_solutions.csv"
    ranked_df = save_ranked_csv(solution_dicts, ranked_csv_path)
    print(f"Solver output saved to {solver_output_file}")
    print(f"Ranked solutions saved to {ranked_csv_path}")

    scatter_plot_path = batch_folder / "plots" / "scatter_score_vs_length.png"
    plot_scatter(solution_dicts, scatter_plot_path)

    if solution_dicts:
        best_solution = max(solution_dicts, key=lambda sol: sol.get("Cumulative_Score", 0))
        print("\nBest performing solution meta_visual:")
        print(best_solution.get("meta_visual", "N/A"))

    print("\ndensehairpins pipeline completed successfully.")

if __name__ == "__main__":
    main()
