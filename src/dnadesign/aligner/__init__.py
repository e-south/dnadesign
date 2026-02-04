"""
--------------------------------------------------------------------------------
<dnadesign project>
aligner/__init__.py

Provides high-level functions for computing Needleman-Wunsch alignment scores.

Module Author(s): Eric J. South
Dunlop Lab
--------------------------------------------------------------------------------
"""

from .metrics import (
    compute_alignment_scores as compute_alignment_scores,
)
from .metrics import (
    mean_pairwise as mean_pairwise,
)
from .metrics import (
    score_pairwise as score_pairwise,
)
