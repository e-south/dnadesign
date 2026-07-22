"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/studies/units/retron_hairpin_design/catalog/strict_mapping_io.py

Strict mapping loaders for Retron MSD catalog and spec files.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml
from yaml.resolver import BaseResolver


class DuplicateMappingKeyError(ValueError):
    """Raised when a mapping file repeats a key that YAML would otherwise replace."""


class _UniqueMappingYamlLoader(yaml.SafeLoader):
    pass


def load_unique_yaml(path: str | Path) -> Any:
    source_path = Path(path).expanduser().resolve()
    return yaml.load(source_path.read_text(encoding="utf-8"), Loader=_UniqueMappingYamlLoader)


def _construct_unique_mapping(loader: yaml.SafeLoader, node: yaml.Node, deep: bool = False) -> dict[Any, Any]:
    mapping: dict[Any, Any] = {}
    for key_node, value_node in node.value:
        key = loader.construct_object(key_node, deep=deep)
        try:
            duplicate = key in mapping
        except TypeError as exc:
            raise DuplicateMappingKeyError(f"unsupported mapping key: {key!r}.") from exc
        if duplicate:
            raise DuplicateMappingKeyError(f"duplicate mapping key: {key!r}.")
        mapping[key] = loader.construct_object(value_node, deep=deep)
    return mapping


_UniqueMappingYamlLoader.add_constructor(BaseResolver.DEFAULT_MAPPING_TAG, _construct_unique_mapping)


__all__ = ["DuplicateMappingKeyError", "load_unique_yaml"]
