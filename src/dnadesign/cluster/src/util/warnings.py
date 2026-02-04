"""
--------------------------------------------------------------------------------
<dnadesign project>
src/dnadesign/cluster/src/util/warnings.py

Keep third-party warning noise contained.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import warnings


def configure(verbose: bool = False) -> None:
    if not verbose:
        # anndata deprecation redirects (import surface noise)
        # `anndata` emits deprecations from the top-level module ("anndata"), not only submodules.
        warnings.filterwarnings("ignore", category=FutureWarning, module=r"^anndata")
        # leidenalg docstring escape sequences
        warnings.filterwarnings("ignore", category=SyntaxWarning, module=r"^leidenalg")
    # We avoid Scanpy's 'future default backend' chatter by selecting the backend explicitly.
