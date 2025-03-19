"""
--------------------------------------------------------------------------------
<dnadesign project>
/densehairpins/visualization.py

Module Author(s): Eric J. South
Dunlop Lab
--------------------------------------------------------------------------------
"""

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

def save_ranked_csv(solution_dicts, ranked_csv_path):
    """
    Saves the ranked solution data to CSV after sorting by 'Cumulative_Score'.
    If no 'Cumulative_Score' column is present, a warning is printed and data is saved unsorted.
    """
    # Ensure solution_dicts is defined.
    if solution_dicts is None:
        print("Warning: No solution data provided to save_ranked_csv.")
        return pd.DataFrame()
    
    df = pd.DataFrame(solution_dicts)
    if df.empty:
        print("Warning: No solution data available to rank. Skipping ranked CSV saving.")
        return df
    if "Cumulative_Score" not in df.columns:
        print("Warning: 'Cumulative_Score' column not found in solution data. Ranked CSV will be saved unsorted.")
    else:
        df = df.sort_values(by="Cumulative_Score", ascending=False)
    df.to_csv(ranked_csv_path, index=False)
    print(f"Ranked CSV saved to {ranked_csv_path}")
    return df


def plot_scatter(solution_dicts, scatter_plot_path):
    """
    Generates and saves a scatter plot where the X-axis is Cumulative_Score,
    Y-axis is Sequence_Length, and point size is proportional to TF_Count.
    """
    import matplotlib.pyplot as plt

    # Create DataFrame from solutions.
    df = pd.DataFrame(solution_dicts)
    if df.empty:
        print("Warning: No solution data available to plot scatter chart.")
        return

    # Check required columns.
    for col in ["Cumulative_Score", "Sequence_Length", "TF_Count"]:
        if col not in df.columns:
            print(f"Warning: Column {col} is missing in solution data; cannot plot scatter chart.")
            return

    plt.figure()
    plt.scatter(df["Cumulative_Score"], df["Sequence_Length"], s=df["TF_Count"]*20, alpha=0.6)
    plt.xlabel("Cumulative Score")
    plt.ylabel("Sequence Length")
    plt.title("Scatter Plot of Solutions")
    plt.savefig(scatter_plot_path)
    plt.close()
    print(f"Scatter plot saved to {scatter_plot_path}")
