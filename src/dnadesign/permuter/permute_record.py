"""
--------------------------------------------------------------------------------
<dnadesign project>
dnadesign/permuter/permute_record.py

Responsible for dispatching a single reference entry to the chosen permutation
protocol class via the Protocol registry. This layer keeps the main pipeline
decoupled from individual protocol implementations.

Module Author(s): Eric J. South
Dunlop Lab
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from typing import Dict, Iterable

import numpy as np

from dnadesign.permuter.protocols import get_protocol


def permute_record(
    ref_entry: Dict,
    protocol: str,
    params: Dict,
) -> Iterable[Dict]:
    """
    Look up and call the protocol's generate() method (streaming).

    Args:
      ref_entry: seed variant dict with fields:
        - var_id, ref_name, protocol, sequence, modifications, round
      protocol: protocol id (e.g., "scan_dna", "scan_codon", "scan_stem_loop")
      params: protocol-specific parameters (validated by the protocol)

    Returns:
      Iterable of new variant dicts. Each includes:
        - 'sequence': new full sequence
        - 'modifications': [summary, ...]
        - protocol-specific flat keys (e.g., nt_*, codon_*, hp_*, gen_*)
      (IDs are assigned upstream deterministically.)
    """
    proto = get_protocol(protocol)
    proto.validate_cfg(params=params)
    base_seed = params.get("_derived_seed")
    rng = (
        np.random.default_rng(base_seed)
        if base_seed is not None
        else np.random.default_rng()
    )
    return proto.generate(ref_entry=ref_entry, params=params, rng=rng)
