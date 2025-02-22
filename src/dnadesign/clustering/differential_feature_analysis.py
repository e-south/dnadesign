"""
--------------------------------------------------------------------------------
<dnadesign project>
dnadesign/clustering/differential_feature_analysis.py

Module Author(s): Eric J. South
Dunlop Lab
--------------------------------------------------------------------------------
"""

import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
import pandas as pd

def identify_markers(data_entries, save_csv=None):
    """
    Identify over-represented input_sources (markers) per cluster and save as CSV.
    """
    print("Starting Differential Feature Analysis...")
    records = []
    for entry in data_entries:
        cluster = entry.get("leiden_cluster")
        group = entry.get("meta_input_source", "unknown")
        records.append({"cluster": cluster, "group": group})
    df = pd.DataFrame(records)
    
    overall_freq = df["group"].value_counts(normalize=True)
    markers = {}
    for cl, group_df in df.groupby("cluster"):
        cluster_freq = group_df["group"].value_counts(normalize=True)
        fold_changes = cluster_freq / overall_freq
        marker_groups = fold_changes[fold_changes > 1.5].index.tolist()
        markers[cl] = marker_groups
    
    markers = {cl: ", ".join(marker_list) for cl, marker_list in markers.items()}
    markers_df = pd.DataFrame.from_dict(markers, orient='index', columns=["markers"])
    print("Differential Feature Analysis - Marker Identification:")
    print(markers_df)
    
    if save_csv:
        markers_df.to_csv(save_csv)
        print(f"Differential feature analysis results saved as CSV to {save_csv}")
