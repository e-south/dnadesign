"""
--------------------------------------------------------------------------------
<dnadesign project>

Module Author(s): Eric J. South
Dunlop Lab
--------------------------------------------------------------------------------
"""

from .cluster_select import select_clusters
from .differential_feature_analysis import identify_markers
from .diversity_analysis import assess
from .intra_cluster_similarity_analysis import compute_intra_cluster_similarity, plot_intra_cluster_similarity
from .normalization import extract_numeric_hue, extract_type_hue, normalize_array
from .plotter import plot_umap
from .resolution_sweep import run_resolution_sweep
