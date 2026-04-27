"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/usr/src/registry/__init__.py

Namespace registry loading and validation for USR overlays.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

import yaml

from .models import (
    DERIVED_NAMESPACE,
    REGISTRY_FILENAME,
    SEQ_ANNOT_NAMESPACE,
    USR_LABEL_NAMESPACE,
    USR_STATE_COLUMNS,
    USR_STATE_NAMESPACE,
    RegistryColumn,
    RegistryEntry,
)
from .storage import (
    _REGISTRY_CACHE,
    _REGISTRY_CACHE_MAX,
    _registry_cache_entry,
    _registry_canonical_bytes,
    _registry_canonical_hash,
    derived_entry,
    ensure_registry_entries,
    ensure_sequence_contract_namespaces,
    load_registry,
    load_registry_file,
    namespace_contract_hash,
    namespace_contract_hash_for_entries,
    register_namespace,
    registry_bytes,
    registry_bytes_for_entries,
    registry_hash,
    registry_hash_for_entries,
    registry_path,
    save_registry,
    seq_annot_entry,
    usr_label_entry,
    usr_state_entry,
)
from .typespec import _parse_fixed_size_list, _split_top_level, arrow_type_from_str, arrow_type_str, parse_type_str
from .validation import (
    _ensure_usr_state_entry,
    _parse_entry,
    _validate_columns,
    parse_columns_spec,
    registry_entry,
    validate_overlay_schema,
)

__all__ = [
    "REGISTRY_FILENAME",
    "USR_LABEL_NAMESPACE",
    "SEQ_ANNOT_NAMESPACE",
    "DERIVED_NAMESPACE",
    "USR_STATE_COLUMNS",
    "USR_STATE_NAMESPACE",
    "RegistryColumn",
    "RegistryEntry",
    "_REGISTRY_CACHE",
    "_REGISTRY_CACHE_MAX",
    "_ensure_usr_state_entry",
    "_parse_entry",
    "_parse_fixed_size_list",
    "_registry_cache_entry",
    "_registry_canonical_bytes",
    "_registry_canonical_hash",
    "_split_top_level",
    "_validate_columns",
    "arrow_type_from_str",
    "arrow_type_str",
    "derived_entry",
    "ensure_registry_entries",
    "ensure_sequence_contract_namespaces",
    "load_registry",
    "load_registry_file",
    "namespace_contract_hash",
    "namespace_contract_hash_for_entries",
    "parse_columns_spec",
    "parse_type_str",
    "register_namespace",
    "registry_bytes",
    "registry_bytes_for_entries",
    "registry_entry",
    "registry_hash",
    "registry_hash_for_entries",
    "registry_path",
    "save_registry",
    "seq_annot_entry",
    "usr_label_entry",
    "usr_state_entry",
    "validate_overlay_schema",
    "yaml",
]
