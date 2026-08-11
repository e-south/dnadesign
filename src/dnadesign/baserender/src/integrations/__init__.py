"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/baserender/src/integrations/__init__.py

Internal facade for built-in BaseRender integration capabilities.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from .adapters import (
    adapter_contract,
    adapter_descriptor,
    adapter_descriptors,
    adapter_grid_record_limit,
    adapter_kinds,
    build_adapter,
    declared_adapter_path_values,
    finalize_adapter,
    normalize_adapter_config,
    required_source_columns,
    validate_adapter_output_policy,
    validate_record_renderer_compatibility,
    validate_records_output_policy,
)
from .contracts import AdapterDescriptor, IntegrationProvider, TransformDescriptor
from .registry import integration_providers, registered_render_contracts
from .transforms import (
    declared_transform_path_values,
    normalize_transform_config,
    transform_descriptor,
    transform_descriptors,
    transform_names,
)

__all__ = [
    "AdapterDescriptor",
    "IntegrationProvider",
    "TransformDescriptor",
    "adapter_contract",
    "adapter_descriptor",
    "adapter_descriptors",
    "adapter_grid_record_limit",
    "adapter_kinds",
    "build_adapter",
    "declared_adapter_path_values",
    "declared_transform_path_values",
    "finalize_adapter",
    "integration_providers",
    "registered_render_contracts",
    "normalize_adapter_config",
    "normalize_transform_config",
    "required_source_columns",
    "transform_descriptor",
    "transform_descriptors",
    "transform_names",
    "validate_adapter_output_policy",
    "validate_record_renderer_compatibility",
    "validate_records_output_policy",
]
