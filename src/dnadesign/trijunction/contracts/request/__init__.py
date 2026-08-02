"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/trijunction/contracts/request/__init__.py

Immutable public request contract for TriJunction planning.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from ...errors import TriJunctionConfigError
from .codec import canonical_request_bytes, parse_request, request_to_mapping
from .files import load_request
from .limits import MAX_REQUEST_BYTES, MAX_REQUEST_IDENTIFIER_BYTES, MAX_REQUEST_PLAIN_TEXT_BYTES
from .model import (
    MAX_BARCODE_GENERATION_ATTEMPTS,
    MAX_BARCODE_SUBSET_ITERATIONS,
    MAX_MATCHING_ITERATIONS,
    MAX_TOEHOLD_SEARCH_ITERATIONS,
    REQUEST_SCHEMA,
    ComplementEndPreparation,
    OrderPolicy,
    PlanningProfile,
    Primer,
    RecoveryPrimerMode,
    RecoveryPrimerPair,
    Target,
    TriJunctionRequest,
)

__all__ = [
    "MAX_BARCODE_GENERATION_ATTEMPTS",
    "MAX_BARCODE_SUBSET_ITERATIONS",
    "MAX_MATCHING_ITERATIONS",
    "MAX_REQUEST_BYTES",
    "MAX_REQUEST_IDENTIFIER_BYTES",
    "MAX_REQUEST_PLAIN_TEXT_BYTES",
    "MAX_TOEHOLD_SEARCH_ITERATIONS",
    "REQUEST_SCHEMA",
    "ComplementEndPreparation",
    "OrderPolicy",
    "PlanningProfile",
    "Primer",
    "RecoveryPrimerMode",
    "RecoveryPrimerPair",
    "Target",
    "TriJunctionConfigError",
    "TriJunctionRequest",
    "canonical_request_bytes",
    "load_request",
    "parse_request",
    "request_to_mapping",
]
