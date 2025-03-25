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

def save_ranked_csv(solution_dicts, ranked_csv_path):
    """
    Saves the ranked solution data to CSV after sorting by 'Cumulative_Score'.
    If no 'Cumulative_Score' column is present, a warning is printed and data is saved unsorted.
    """
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
    Generates and saves a scatter plot where:
    - X-axis: Cumulative_Score
    - Y-axis: Sequence_Length
    - Point size: Proportional to TF_Count
    - Point color: Based on Unique_TF_Count
    - Annotations: Top 5 sequences by combined higher score and shorter length
    """
    # Create DataFrame from solutions
    df = pd.DataFrame(solution_dicts)
    if df.empty:
        print("Warning: No solution data available to plot scatter chart.")
        return

    # Check required columns
    for col in ["Cumulative_Score", "Sequence_Length", "TF_Count"]:
        if col not in df.columns:
            print(f"Warning: Column {col} is missing in solution data; cannot plot scatter chart.")
            return
    
    # Fill NAs with calculated values
    if df["Cumulative_Score"].isna().any():
        print(f"Warning: {df['Cumulative_Score'].isna().sum()} entries missing Cumulative_Score values")
        
    if df["Sequence_Length"].isna().any():
        # Fill sequence length from actual sequence lengths
        for idx, row in df[df["Sequence_Length"].isna()].iterrows():
            if "sequence" in row and isinstance(row["sequence"], str):
                df.at[idx, "Sequence_Length"] = len(row["sequence"])
    
    # Drop rows with missing required values
    valid_df = df.dropna(subset=["Cumulative_Score", "Sequence_Length", "TF_Count"])
    if len(valid_df) < len(df):
        print(f"Warning: {len(df) - len(valid_df)} entries dropped from plot due to missing values")

    # Set up styling
    sns.set_style("ticks")
    plt.rcParams['font.family'] = 'Arial'
    plt.rcParams['font.size'] = 14
    
    plt.figure(figsize=(12, 10))
    
    # Create scatter plot with alpha=0.5
    if "Unique_TF_Count" in valid_df.columns:
        # Convert to integer to ensure discrete color mapping
        valid_df["Unique_TF_Count"] = valid_df["Unique_TF_Count"].fillna(0).astype(int)
        unique_counts = valid_df["Unique_TF_Count"]
        
        # Get all unique values that actually appear in the data
        unique_values = sorted(unique_counts.unique())
        min_unique = min(unique_values)
        max_unique = max(unique_values)
        
        print(f"Plotting Unique TF counts from {min_unique} to {max_unique}")
        print(f"Unique TF count values present: {unique_values}")
        
        # Use a discrete colormap with exact range
        n_colors = len(unique_values)
        cmap = plt.cm.get_cmap('viridis', n_colors)
        
        # Create a mapping from unique values to color indices
        value_to_index = {val: i for i, val in enumerate(unique_values)}
        color_indices = [value_to_index[val] for val in unique_counts]
        
        scatter = plt.scatter(valid_df["Cumulative_Score"], valid_df["Sequence_Length"], 
                             s=valid_df["TF_Count"]*25,  # Size based on total TF count
                             c=color_indices,           # Color based on unique TF count
                             alpha=0.5, 
                             cmap=cmap,
                             vmin=0, vmax=n_colors-1)
        
        # Create discrete colorbar with all present values
        cbar = plt.colorbar(ticks=range(n_colors))
        cbar.set_label("Unique TF Count", fontsize=16)
        cbar.ax.set_yticklabels(unique_values)
        cbar.ax.tick_params(labelsize=14)
    else:
        scatter = plt.scatter(valid_df["Cumulative_Score"], valid_df["Sequence_Length"], 
                             s=valid_df["TF_Count"]*25, 
                             alpha=0.5)
    
    # Identify top 5 sequences by combined score-to-length ratio (higher is better)
    if len(valid_df) > 0:
        # Calculate a combined metric that rewards higher scores and shorter sequences
        valid_df['Score_Length_Ratio'] = valid_df['Cumulative_Score'] / valid_df['Sequence_Length']
        
        # Get top 5 by this combined ratio
        top_5 = valid_df.nlargest(5, 'Score_Length_Ratio')
        
        # Annotate the top 5
        for idx, row in top_5.iterrows():
            entry_id = row.get('entry_id', 'Unknown')
            score = row['Cumulative_Score']
            length = row['Sequence_Length']
            plt.annotate(f"ID: {entry_id}",
                        (score, length),
                        xytext=(10, 0),
                        textcoords='offset points',
                        fontsize=12,
                        arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=.2'))
    
    # Increase font sizes
    plt.xlabel("Cumulative Score", fontsize=16)
    plt.ylabel("Sequence Length", fontsize=16)
    plt.title("Scatter Plot of DNA Design Solutions", fontsize=18)
    
    # Increase tick label sizes
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    
    # Remove top and right spines
    ax = plt.gca()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    # Use seaborn style ticks
    sns.despine()
    
    plt.tight_layout()
    plt.savefig(scatter_plot_path, dpi=300)
    plt.close()
    
    print(f"Scatter plot saved to {scatter_plot_path}")
