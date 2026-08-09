"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/msd/compiler/__init__.py

Public operations for resolving and compiling MSD designs.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from .errors import RetronMsdCompilerError
from .identifiers import (
    MsdDesignPartInput,
    MsdIdError,
    ParsedMsdConstructLabel,
    canonical_msd_construct_label,
    compute_scar_nick_profile,
    parse_msd_construct_label,
    parse_msd_design_parts,
)
from .registry import RetronMsdRegistry, RetronMsdRegistryError, load_retron_msd_registry
from .resolution import (
    MsdCompilerSpecError,
    ResolvedMsdCompilerSpec,
    RetronMsdCompilerSpecV1,
    load_msd_compiler_spec,
    resolve_msd_compiler_spec_payload,
)
from .sequence import MsdSequenceInputError, validate_dna_sequence
from .units import MsdCompiledSegment, MsdCompiledUnitV1, compile_msd_design_unit

__all__ = [
    "MsdCompiledSegment",
    "MsdCompiledUnitV1",
    "MsdCompilerSpecError",
    "MsdDesignPartInput",
    "MsdIdError",
    "MsdSequenceInputError",
    "ParsedMsdConstructLabel",
    "ResolvedMsdCompilerSpec",
    "RetronMsdCompilerError",
    "RetronMsdCompilerSpecV1",
    "RetronMsdRegistry",
    "RetronMsdRegistryError",
    "canonical_msd_construct_label",
    "compute_scar_nick_profile",
    "compile_msd_design_unit",
    "load_msd_compiler_spec",
    "load_retron_msd_registry",
    "parse_msd_construct_label",
    "parse_msd_design_parts",
    "resolve_msd_compiler_spec_payload",
    "validate_dna_sequence",
]
