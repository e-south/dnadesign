"""
--------------------------------------------------------------------------------
<dnadesign project>
dnadesign/permuter/permute_record.py

Generate variant sequences given a reference entry and a protocol.

Module Author(s): Eric J. South
Dunlop Lab
--------------------------------------------------------------------------------
"""

import importlib
from typing import Dict, List

from logger import get_logger

logger = get_logger(__name__)


def permute_record(
    ref_entry: Dict,
    protocol: str,
    params: Dict,
    regions: List = None,
    lookup_tables: List[str] = None,
) -> List[Dict]:
    assert isinstance(ref_entry, dict), "ref_entry must be a dict"
    regions = regions or []
    lookup_tables = lookup_tables or []

    module = importlib.import_module(f"permuter.protocols.{protocol}")
    assert hasattr(
        module, "generate_variants"
    ), f"Protocol '{protocol}' missing generate_variants"
    variants = module.generate_variants(ref_entry, params, regions, lookup_tables)
    for v in variants:
        # inherit metadata
        v.setdefault("ref_name", ref_entry["ref_name"])
        v.setdefault("protocol", ref_entry["protocol"])
        v.setdefault("meta_source", ref_entry["meta_source"])
        v.setdefault("meta_date_accessed", ref_entry["meta_date_accessed"])
        v.setdefault("meta_type", "variant")
    return variants
