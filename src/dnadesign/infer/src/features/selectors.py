"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/infer/src/features/selectors.py

Canonical selector resolution for Evo2 promoter feature extraction.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import re

from ..errors import CapabilityError, ConfigError
from .contracts import SelectorResolution

_SUPPORTED_MODELS = {"evo2_7b", "evo2_20b"}
_CANONICAL_SELECTOR_PATTERN = re.compile(r"^block(\d+)_mlp_out$")


def canonical_selector_for_block(block: int) -> str:
    if int(block) < 0:
        raise ConfigError("intermediate_block must be >= 0")
    return f"block{int(block)}_mlp_out"


def provider_layer_for_block(block: int) -> str:
    if int(block) < 0:
        raise ConfigError("intermediate_block must be >= 0")
    return f"blocks.{int(block)}.mlp.l3"


def provider_layer_from_selector(selector: str) -> str:
    text = str(selector or "").strip()
    if not text:
        raise CapabilityError("intermediate selector must be non-empty.")
    match = _CANONICAL_SELECTOR_PATTERN.fullmatch(text)
    if match is None:
        raise CapabilityError(f"Unsupported canonical Evo2 selector '{selector}'. Expected pattern 'block<N>_mlp_out'.")
    return provider_layer_for_block(int(match.group(1)))


def resolve_intermediate_selector(*, model_id: str, intermediate_block: int) -> SelectorResolution:
    if model_id not in _SUPPORTED_MODELS:
        rendered = ", ".join(sorted(_SUPPORTED_MODELS))
        raise ConfigError(f"Evo2 promoter feature bundle supports model.id values: {rendered}. Received '{model_id}'.")
    selector = canonical_selector_for_block(intermediate_block)
    return SelectorResolution(
        intermediate_block=int(intermediate_block),
        intermediate_selector=selector,
        provider_layer=provider_layer_for_block(intermediate_block),
    )
