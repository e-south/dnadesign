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
    """Create a properly named batch directory with subdirectories."""
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
    """Remove duplicate TF-sequence entries while preserving metadata."""
    raw_count = len(entries)
    unique_entries = {(entry['TF'].strip(), entry['sequence'].strip()): entry for entry in entries}
    unique_list = list(unique_entries.values())
    unique_count = len(unique_list)
    
    assert unique_count > 0, "No unique binding site entries found after deduplication."
    
    if raw_count != unique_count:
        print(f"Note: Deduplicated {raw_count - unique_count} duplicate entries; using {unique_count} unique binding site entries for library building.")
    
    return unique_list

def sample_library(unique_list: list, subsample_size: int = None) -> list:
    """Sample from the library, either taking all entries or a random subset."""
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

def get_all_solver_outputs(batch_results_dir):
    """Find all solver_output.csv files across all batch directories."""
    solver_outputs = []
    for batch_folder in batch_results_dir.glob("*"):
        if batch_folder.is_dir():
            solver_file = batch_folder / "csvs" / "solver_output.csv"
            if solver_file.exists():
                solver_outputs.append(solver_file)
    return solver_outputs

def merge_solver_outputs(csv_files, output_file=None):
    """
    Merge multiple solver_output.csv files into a single DataFrame.
    Adds a Source_Batch column to track where each entry originated.
    """
    dfs = []
    for file in csv_files:
        try:
            df = pd.read_csv(file)
            # Add source tracking
            df['Source_Batch'] = file.parent.parent.name
            dfs.append(df)
        except Exception as e:
            print(f"Error reading {file}: {e}")
    
    if not dfs:
        print("No valid CSV files found to merge.")
        return pd.DataFrame()
    
    # Concatenate all dataframes
    merged_df = pd.concat(dfs, ignore_index=True)
    
    # Remove duplicates based on entry_id if it exists
    if 'entry_id' in merged_df.columns:
        merged_df = merged_df.drop_duplicates(subset=['entry_id'])
    
    # Save to output file if specified
    if output_file:
        merged_df.to_csv(output_file, index=False)
        print(f"Merged solver output saved to {output_file}")
    
    return merged_df

def find_batch_folder(batch_results_dir, target_batch):
    """Find a specific batch folder by name."""
    for folder in batch_results_dir.glob("*"):
        if folder.is_dir() and target_batch in folder.name:
            return folder
    return None

def process_post_solve_aggregate(batch_results_dir, config):
    """Process aggregate analysis in post-solve mode."""
    print("Running post-solve aggregate analysis...")
    
    # Create aggregate analysis directory
    aggregate_dir = batch_results_dir / "aggregate_analysis"
    aggregate_dir.mkdir(exist_ok=True)
    aggregate_csvs_dir = aggregate_dir / "csvs"
    aggregate_plots_dir = aggregate_dir / "plots"
    aggregate_csvs_dir.mkdir(exist_ok=True)
    aggregate_plots_dir.mkdir(exist_ok=True)
    
    # Find and merge all solver outputs
    solver_outputs = get_all_solver_outputs(batch_results_dir)
    if not solver_outputs:
        print("Warning: No solver output files found to analyze.")
        return
        
    print(f"Found {len(solver_outputs)} solver output files to merge.")
    
    # First, create a merged file without scoring
    merged_output_file = aggregate_csvs_dir / "merged_solver_output.csv"
    merged_df = merge_solver_outputs(solver_outputs, merged_output_file)
    
    if merged_df.empty:
        print("Warning: No data found to analyze after merging solver outputs.")
        return
    
    print(f"Successfully merged {len(merged_df)} entries from all batches.")
    
    # Load Pancardo dataset for scoring
    pancardo_df = load_pancardo_dataset()
    
    # Score the merged solutions
    score_weights = config.get("score_weights", {
        "silenced_genes": 1, 
        "induced_genes": 1,
        "tf_diversity": 1  # Default diversity weight
    })
    
    print(f"Applying scoring with weights: {score_weights}")
    
    # Convert DataFrame rows to dictionaries for scoring
    solution_dicts = merged_df.to_dict(orient="records")
    
    # Check if any entries are missing TF rosters
    missing_rosters = sum(1 for sol in solution_dicts if "tf_roster" not in sol or not sol["tf_roster"])
    if missing_rosters > 0:
        print(f"Warning: {missing_rosters} entries are missing TF roster data.")
        
        # Try to repair if the data is in string format
        for sol in solution_dicts:
            if "tf_roster" in sol and isinstance(sol["tf_roster"], str):
                try:
                    # Handle common string representations
                    if sol["tf_roster"].startswith("['") and sol["tf_roster"].endswith("']"):
                        sol["tf_roster"] = eval(sol["tf_roster"])
                    elif sol["tf_roster"].startswith("[") and sol["tf_roster"].endswith("]"):
                        sol["tf_roster"] = eval(sol["tf_roster"])
                except Exception as e:
                    print(f"Error parsing tf_roster: {e}")
    
    # Apply scoring
    solution_dicts = add_scores_to_solutions(solution_dicts, pancardo_df, score_weights)
        
    # Remove detailed score components before saving ranked solutions to avoid overly large CSVs
    for sol in solution_dicts:
        if "TF_Score_Details" in sol:
            del sol["TF_Score_Details"]
    
    # Save the ranked solutions
    ranked_csv_path = aggregate_csvs_dir / "ranked_solutions.csv"
    ranked_df = save_ranked_csv(solution_dicts, ranked_csv_path)
    
    # Generate scatter plot
    scatter_plot_path = aggregate_plots_dir / "scatter_score_vs_length.png"
    plot_scatter(solution_dicts, scatter_plot_path)
    
    # Report best solution
    if solution_dicts:
        best_solution = max(solution_dicts, key=lambda sol: sol.get("Cumulative_Score", 0))
        print("\nBest performing solution meta_visual:")
        print(best_solution.get("meta_visual", "N/A"))
        print(f"From batch: {best_solution.get('Source_Batch', 'unknown')}")
        print(f"Score: {best_solution.get('Cumulative_Score', 0)}")
        print(f"Unique TFs: {best_solution.get('Unique_TF_Count', 0)}")
        print(f"Total TFs: {best_solution.get('TF_Count', 0)}")
        print(f"Sequence Length: {best_solution.get('Sequence_Length', 0)}")
    
    print("\nAggregate analysis completed successfully.")

def process_post_solve_per_batch(batch_results_dir, config):
    """Process per-batch analysis in post-solve mode."""
    target_batch = config.get("target_batch")
    if not target_batch:
        raise ValueError("Target batch must be specified for per_batch analysis with run_post_solve=true.")
    
    # Find the target batch folder
    batch_folder = find_batch_folder(batch_results_dir, target_batch)
    if not batch_folder:
        raise ValueError(f"Target batch '{target_batch}' not found in batch_results directory.")
    
    print(f"Running post-solve analysis on batch: {batch_folder.name}")
    
    # Load existing solver output
    solver_output_file = batch_folder / "csvs" / "solver_output.csv"
    if not solver_output_file.exists():
        raise FileNotFoundError(f"Solver output file not found: {solver_output_file}")
    
    solution_df = pd.read_csv(solver_output_file)
    solution_dicts = solution_df.to_dict(orient="records")
    
    # Load Pancardo dataset for scoring
    pancardo_df = load_pancardo_dataset()
    
    # Score the solutions
    score_weights = config.get("score_weights", {
        "silenced_genes": 1, 
        "induced_genes": 1,
        "tf_diversity": 1
    })
    
    solution_dicts = add_scores_to_solutions(solution_dicts, pancardo_df, score_weights)
    
    # Save the ranked solutions
    ranked_csv_path = batch_folder / "csvs" / "ranked_solutions.csv"
    ranked_df = save_ranked_csv(solution_dicts, ranked_csv_path)
    
    # Generate scatter plot
    scatter_plot_path = batch_folder / "plots" / "scatter_score_vs_length.png"
    plot_scatter(solution_dicts, scatter_plot_path)
    
    # Report best solution
    if solution_dicts:
        best_solution = max(solution_dicts, key=lambda sol: sol.get("Cumulative_Score", 0))
        print("\nBest performing solution meta_visual:")
        print(best_solution.get("meta_visual", "N/A"))
    
    print(f"\nPost-solve analysis for batch {batch_folder.name} completed successfully.")

def main():
    random_seed = 42
    random.seed(random_seed)

    config_path = Path(__file__).resolve().parent.parent / "configs/example.yaml"
    config_loader = ConfigLoader(config_path)
    config = config_loader.config

    consensus_flag = config.get("consensus_only", True)
    run_post_solve = config.get("run_post_solve", False)
    analysis_style = config.get("analysis_style", "per_batch")  # Default to per_batch
    
    # Add warning about aggregate analysis
    if analysis_style == "aggregate" and not run_post_solve:
        print("\nWARNING: You've selected 'aggregate' analysis style but 'run_post_solve' is False.")
        print("Aggregate analysis is only performed when 'run_post_solve' is True.")
        print("The current run will generate solver output for a single batch only.")
        print("To perform aggregate analysis across all batches, set 'run_post_solve: true' in your config.\n")
    
    batch_results_dir = Path(__file__).resolve().parent / "batch_results"
    batch_results_dir.mkdir(exist_ok=True)
    
    # Handle post-solve analysis mode first
    if run_post_solve:
        if analysis_style == "aggregate":
            process_post_solve_aggregate(batch_results_dir, config)
        else:  # per_batch
            process_post_solve_per_batch(batch_results_dir, config)
        return
    
    # If we're here, we're not in post-solve mode, so run the full pipeline
    batch_name = config.get("batch_name", "default_batch")
    batch_label = config.get("batch_label", None)
    batch_folder = create_batch_folder(batch_results_dir, batch_name, batch_label, use_timestamp=True, overwrite=True)

    pancardo_df = load_pancardo_dataset()

    omalley_dir = DATA_FILES["omalley_et_al"]
    if not omalley_dir.exists():
        raise FileNotFoundError(f"O'Malley directory not found: {omalley_dir}")

    # Process TF binding site data
    tidy_rows = []
    total_matches = 0
    total_possible = len(pancardo_df)
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

    # Save intermediate data
    intermediate_csv_path = batch_folder / "csvs" / "intermediate.csv"
    pd.DataFrame(tidy_rows).to_csv(intermediate_csv_path, index=False)

    # Prepare library
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

    # Configure solver
    quota = config.get("quota", 10)
    solver = config.get("solver", "CBC")
    solver_options = config.get("solver_options", [])
    sequence_length = config.get("sequence_length", 30)
    random_subsample_per_solve = config.get("random_subsample_per_solve", False)
    solver_output_file = batch_folder / "csvs" / "solver_output.csv"

    # Run solver
    def save_callback(solutions):
        save_solver_csv_iteratively(solutions, solver_output_file)
    
    solution_dicts = run_solver_iteratively(
        library, 
        sequence_length, 
        solver, 
        solver_options, 
        quota,
        save_callback=save_callback,
        random_subsample_per_solve=random_subsample_per_solve,
        subsample_size=subsample_size
    )

    # Score solutions
    score_weights = config.get("score_weights", {
        "silenced_genes": 1, 
        "induced_genes": 1,
        "tf_diversity": 1  # Default to 1 for diversity weight
    })
    solution_dicts = add_scores_to_solutions(solution_dicts, pancardo_df, score_weights)

    # Save final outputs
    solver_df = pd.DataFrame(solution_dicts)
    solver_df.to_csv(solver_output_file, index=False)
    ranked_csv_path = batch_folder / "csvs" / "ranked_solutions.csv"
    ranked_df = save_ranked_csv(solution_dicts, ranked_csv_path)
    print(f"Solver output saved to {solver_output_file}")
    print(f"Ranked solutions saved to {ranked_csv_path}")

    # Generate visualization
    scatter_plot_path = batch_folder / "plots" / "scatter_score_vs_length.png"
    plot_scatter(solution_dicts, scatter_plot_path)

    # Report best solution
    if solution_dicts:
        best_solution = max(solution_dicts, key=lambda sol: sol.get("Cumulative_Score", 0))
        print("\nBest performing solution meta_visual:")
        print(best_solution.get("meta_visual", "N/A"))

    print("\ndensehairpins pipeline completed successfully.")

if __name__ == "__main__":
    main()
