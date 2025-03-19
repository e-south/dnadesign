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


def create_batch_folder(base_dir, batch_name, batch_label=None):
    # Use current date only for folder naming.
    timestamp = datetime.datetime.now().strftime("%Y%m%d")
    folder_name = f"{batch_name}_{batch_label}_{timestamp}" if batch_label else f"{batch_name}_{timestamp}"
    batch_folder = Path(base_dir) / folder_name
    if batch_folder.exists():
        shutil.rmtree(batch_folder)
    batch_folder.mkdir(parents=True)
    # Create subfolders for CSV outputs and plots.
    for sub in ["csvs", "plots"]:
        (batch_folder / sub).mkdir()
    return batch_folder


def deduplicate_entries(entries: list) -> list:
    """Deduplicate binding site entries based on (TF, sequence) pair."""
    raw_count = len(entries)
    unique_entries = {(entry['TF'].strip(), entry['sequence'].strip()): entry for entry in entries}
    unique_list = list(unique_entries.values())
    unique_count = len(unique_list)
    assert unique_count > 0, "No unique binding site entries found after deduplication."
    if raw_count != unique_count:
        print(f"Note: Deduplicated {raw_count - unique_count} duplicate entries; using {unique_count} unique binding site entries for library building.")
    return unique_list


def sample_library(unique_list: list, subsample_size: int = None) -> list:
    """
    Sample the binding site library based on subsample_size.
    If subsample_size is None or exceeds the available count, use all entries.
    """
    unique_count = len(unique_list)
    if subsample_size:
        if subsample_size > unique_count:
            print(f"Warning: subsample_size ({subsample_size}) is larger than available unique entries ({unique_count}). Using all available entries.")
            sampled_data = unique_list
        else:
            sampled_data = random.sample(unique_list, subsample_size)
            # Assert that sampling returns the requested number of entries.
            assert len(sampled_data) == subsample_size, "Sampled data does not match requested subsample_size."
    else:
        sampled_data = unique_list
    return sampled_data


def main():
    # Set a fixed random seed for reproducibility.
    random_seed = 42
    random.seed(random_seed)

    config_path = Path(__file__).resolve().parent.parent / "configs/example.yaml"
    config_loader = ConfigLoader(config_path)
    config = config_loader.config

    # Get consensus flag from config.
    consensus_flag = config.get("consensus_only", True)

    # Create batch folder relative to this file.
    batch_results_dir = Path(__file__).resolve().parent / "batch_results"
    batch_results_dir.mkdir(exist_ok=True)
    batch_name = config.get("batch_name", "default_batch")
    batch_label = config.get("batch_label", None)
    batch_folder = create_batch_folder(batch_results_dir, batch_name, batch_label)

    pancardo_df = load_pancardo_dataset()

    omalley_dir = DATA_FILES["omalley_et_al"]
    if not omalley_dir.exists():
        raise FileNotFoundError(f"O'Malley directory not found: {omalley_dir}")

    tidy_rows = []
    total_matches = 0
    total_possible = len(pancardo_df)

    # Loop over the Pancardo dataset and parse each corresponding MEME file.
    for idx, row in pancardo_df.iterrows():
        tf = row["TF"]
        file_name = f"{tf.lower()}.txt"
        file_path = omalley_dir / file_name
        if not file_path.exists():
            print(f"Warning: File for TF {tf} not found at {file_path}. Skipping.")
            continue

        parsed = parse_meme_file(file_path, consensus_only=consensus_flag)
        if parsed:
            if consensus_flag:
                # When consensus_only is true, parsed returns a single consensus string.
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
            # When consensus_only is false, add additional sites.
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

    # Save the intermediate CSV.
    intermediate_csv_path = batch_folder / "csvs" / "intermediate.csv"
    pd.DataFrame(tidy_rows).to_csv(intermediate_csv_path, index=False)

    # Build the library using the entries from the intermediate data.
    if consensus_flag:
        entries_for_library = [entry for entry in tidy_rows if entry["type"] == "consensus" and entry["sequence"]]
    else:
        entries_for_library = [entry for entry in tidy_rows if entry["sequence"]]

    if not entries_for_library:
        raise ValueError("No binding site sequences found to build the library.")

    # Deduplicate and sample the entries.
    unique_entries = deduplicate_entries(entries_for_library)
    subsample_size = config.get("subsample_size", None)
    sampled_data = sample_library(unique_entries, subsample_size)

    # Build the library as "TF:sequence" strings.
    library = [f"{entry['TF']}:{entry['sequence']}" for entry in sampled_data]
    assert all(":" in item for item in library), "Library items must contain ':' separator."

    quota = config.get("quota", 10)
    solver = config.get("solver", "CBC")
    solver_options = config.get("solver_options", [])
    sequence_length = config.get("sequence_length", 30)

    # Call the solver iteratively.
    solution_dicts = run_solver_iteratively(library, sequence_length, solver, solver_options, quota)

    # Compute scores, sequence lengths, and TF counts.
    score_weights = config.get("score_weights", {"silenced_genes": 1, "induced_genes": 1})
    solution_dicts = add_scores_to_solutions(solution_dicts, pancardo_df, score_weights)

    # Save CSV outputs.
    solver_csv_path = batch_folder / "csvs" / "solver_output.csv"
    ranked_csv_path = batch_folder / "csvs" / "ranked_solutions.csv"
    pd.DataFrame(solution_dicts).to_csv(solver_csv_path, index=False)
    ranked_df = save_ranked_csv(solution_dicts, ranked_csv_path)
    print(f"Solver output saved to {solver_csv_path}")
    print(f"Ranked solutions saved to {ranked_csv_path}")

    # Generate scatter plot: X-axis: Cumulative_Score, Y-axis: Sequence_Length, point size: TF_Count.
    scatter_plot_path = batch_folder / "plots" / "scatter_score_vs_length.png"
    plot_scatter(solution_dicts, scatter_plot_path)

    # Print the meta_visual of the best performing solution.
    if solution_dicts:
        best_solution = max(solution_dicts, key=lambda sol: sol.get("Cumulative_Score", 0))
        print("\nBest performing solution meta_visual:")
        print(best_solution.get("meta_visual", "N/A"))

    print("\ndensehairpins pipeline completed successfully.")


if __name__ == "__main__":
    main()
