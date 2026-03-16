"""
--------------------------------------------------------------------------------
<dnadesign project>
src/dnadesign/cluster/src/methods/__init__.py

Public clustering-method surface.

Module Author(s): Codex
--------------------------------------------------------------------------------
"""

from .contracts import ClusteringMethod, parse_method_param_assignments
from .registry import get_method, supported_method_ids

__all__ = ["ClusteringMethod", "get_method", "parse_method_param_assignments", "supported_method_ids"]
