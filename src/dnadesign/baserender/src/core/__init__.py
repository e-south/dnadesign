"""
--------------------------------------------------------------------------------
<dnadesign project>
src/dnadesign/baserender/src/core/__init__.py

Core contracts, errors, types, and record model exports.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from .contracts import ensure, reject_unknown_keys, require_mapping, require_one_of
from .errors import (
    AlphabetError,
    BaseRenderError,
    BoundsError,
    ContractError,
    ExportError,
    LayoutError,
    PluginError,
    RenderingError,
    SchemaError,
    SkipRecord,
)
from .record import Display, Effect, Feature, Record, TrajectoryInset
from .registry import (
    clear_feature_effect_contracts,
    get_effect_contract,
    get_feature_contract,
    register_builtin_contracts,
    register_effect_contract,
    register_feature_contract,
    validate_record_kinds,
)
from .types import Alphabet, Span, Strand

__all__ = [
    "Alphabet",
    "Strand",
    "Span",
    "Feature",
    "Effect",
    "Display",
    "TrajectoryInset",
    "Record",
    "BaseRenderError",
    "SchemaError",
    "ContractError",
    "AlphabetError",
    "BoundsError",
    "LayoutError",
    "RenderingError",
    "ExportError",
    "PluginError",
    "SkipRecord",
    "ensure",
    "reject_unknown_keys",
    "require_mapping",
    "require_one_of",
    "register_feature_contract",
    "register_effect_contract",
    "get_feature_contract",
    "get_effect_contract",
    "validate_record_kinds",
    "register_builtin_contracts",
    "clear_feature_effect_contracts",
]
