"""
--------------------------------------------------------------------------------
<dnadesign project>
dnadesign/permuter/permute_record.py

Responsible for dispatching a single reference entry to the chosen permutation
protocol module.  This layer keeps the main pipeline decoupled from individual
protocol implementations.

Module Author(s): Eric J. South
Dunlop Lab
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import importlib
from typing import Dict, List


def permute_record(
    ref_entry: Dict,
    protocol: str,
    params: Dict,
    regions: List | None,
    lookup_tables: List[str] | None,
) -> List[Dict]:
    """
    Look up and call the protocol-specific `generate_variants()` function.

    Args:
      ref_entry: {
        "id": str,
        "ref_name": str,
        "protocol": str,
        "sequence": str,
        "modifications": List[str],
        "round": int
      }
      protocol: name of the protocol module under `protocols/`
      params: protocol-specific parameters
      regions: list of [start,end) ranges to mutate (empty → full length)
      lookup_tables: optional auxiliary tables (e.g. codon maps)

    Returns:
      List of new variant dicts, each containing:
        - a unique "id"
        - the updated "sequence"
        - a "modifications" list describing the edit(s)
    """
    # Dynamically import the requested protocol module
    module = importlib.import_module(f"dnadesign.permuter.protocols.{protocol}")
    # Sanity check: ensure the module exposes the expected API
    if not hasattr(module, "generate_variants"):  # pragma: no cover
        raise AttributeError(f"Protocol '{protocol}' lacks generate_variants()")
    # Delegate to the protocol’s generator
    return module.generate_variants(ref_entry, params, regions, lookup_tables)
