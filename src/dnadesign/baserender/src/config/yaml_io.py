"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/baserender/src/config/yaml_io.py

Read safe YAML with explicit-key uniqueness, retaining intentional merge overrides.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from typing import Any

import yaml

from ..core import SchemaError


class _UniqueKeyLoader(yaml.SafeLoader):
    def construct_mapping(self, node, deep=False):
        seen = set()
        for key_node, _ in node.value:
            if not isinstance(key_node, yaml.ScalarNode):
                raise SchemaError(f"YAML mapping keys must be scalar at line {key_node.start_mark.line + 1}")
            key = "<<" if key_node.tag == "tag:yaml.org,2002:merge" else self.construct_object(key_node, deep=deep)
            if key in seen:
                raise SchemaError(f"Duplicate YAML key {key!r} at line {key_node.start_mark.line + 1}")
            seen.add(key)
        return super().construct_mapping(node, deep=deep)


def parse_unique_yaml(text: str) -> Any:
    """Reject duplicate explicit keys without exposing configuration values in errors."""
    try:
        return yaml.load(text, Loader=_UniqueKeyLoader)
    except yaml.YAMLError as exc:
        mark = getattr(exc, "problem_mark", None)
        location = "" if mark is None else f" at line {mark.line + 1}"
        raise SchemaError(f"Invalid YAML syntax{location}") from exc
