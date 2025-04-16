"""
--------------------------------------------------------------------------------
<dnadesign project>

Module Author(s): Eric J. South
Dunlop Lab
--------------------------------------------------------------------------------
"""

from .cluster_select import select_clusters
from .plotter import plot_umap
from .normalization import normalize_array, extract_numeric_hue, extract_type_hue
from .diversity_analysis import assess
from .differential_feature_analysis import identify_markers
from .resolution_sweep import run_resolution_sweep
from .intra_cluster_similarity_analysis import compute_intra_cluster_similarity, plot_intra_cluster_similarity