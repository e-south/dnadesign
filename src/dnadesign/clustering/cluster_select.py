"""
--------------------------------------------------------------------------------
<dnadesign project>
dnadesign/clustering/cluster_select.py

Module Author(s): Eric J. South
Dunlop Lab
--------------------------------------------------------------------------------
"""


def select_clusters(data_entries, selection_config):
    """
    Select clusters to export based on configuration.
    Expected selection_config keys:
      - num_files: int (number of clusters to save; if 0, then no clusters are saved)
      - order: "max", "min", or "custom"
      - custom_clusters: list (if order is "custom")
    """
    num_files = selection_config.get("num_files", None)
    order = selection_config.get("order", "max")
    custom_clusters = selection_config.get("custom_clusters", [])

    cluster_sizes = {}
    for entry in data_entries:
        cl = entry.get("leiden_cluster")
        if cl is None:
            continue
        cluster_sizes[cl] = cluster_sizes.get(cl, 0) + 1
    
    if order == "custom":
        selected = custom_clusters
    else:
        sorted_clusters = sorted(cluster_sizes.items(),
                                 key=lambda x: x[1],
                                 reverse=(order == "max"))
        if num_files == 0:
            selected = []  # If num_files is 0, do not select any clusters.
        elif num_files:
            selected = [cl for cl, size in sorted_clusters[:num_files]]
        else:
            selected = [cl for cl, size in sorted_clusters]
    
    print(f"Selected clusters: {selected}")
    return selected
