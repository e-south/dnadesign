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

def save_ranked_csv(solutions, output_csv):
    """
    Saves a CSV file with the following columns:
      Entry_ID, Sequence_Length, TF_Count, Cumulative_Score, TFs
    Sorted by Cumulative_Score descending.
    """
    data = []
    for sol in solutions:
        data.append({
            "Entry_ID": sol.get("entry_id"),
            "Sequence_Length": sol.get("Sequence_Length"),
            "TF_Count": sol.get("TF_Count"),
            "Cumulative_Score": sol.get("Cumulative_Score"),
            "TFs": ", ".join(sol.get("tf_roster", []))
        })
    df = pd.DataFrame(data)
    df = df.sort_values(by="Cumulative_Score", ascending=False)
    df.to_csv(output_csv, index=False)
    return df

def plot_scatter(solutions, output_path):
    """
    Creates a scatter plot where:
      - X-axis: Cumulative_Score
      - Y-axis: Sequence_Length
      - Point size: TF_Count
      - Color: viridis palette
      - Alpha: 0.5
    Annotates the top 5 solutions.
    """
    data = []
    for sol in solutions:
        data.append({
            "Entry_ID": sol.get("entry_id"),
            "Sequence_Length": sol.get("Sequence_Length"),
            "TF_Count": sol.get("TF_Count"),
            "Cumulative_Score": sol.get("Cumulative_Score")
        })
    df = pd.DataFrame(data)
    
    plt.figure(figsize=(8, 6))
    sns.scatterplot(
        data=df,
        x="Cumulative_Score",
        y="Sequence_Length",
        size="TF_Count",
        hue="Cumulative_Score",
        palette="viridis",
        alpha=0.5,
        legend=False
    )
    sns.despine()
    
    top5 = df.nlargest(5, 'Cumulative_Score')
    for _, row in top5.iterrows():
        plt.text(row["Cumulative_Score"], row["Sequence_Length"], str(int(row["Cumulative_Score"])),
                 fontsize=9, color="black")
    
    plt.xlabel("Cumulative Score")
    plt.ylabel("Sequence Length")
    plt.title("Dense Array Solutions: Score vs Sequence Length")
    plt.tight_layout()
    plt.savefig(output_path, dpi=600)
    plt.close()
