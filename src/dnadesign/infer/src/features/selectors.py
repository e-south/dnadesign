"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/infer/src/features/selectors.py

Canonical selector resolution for Evo2 feature extraction.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import re

from ..errors import CapabilityError, ConfigError
from .contracts import SelectorResolution

_SUPPORTED_MODELS = {"evo2_7b", "evo2_20b"}
_MODEL_DEFAULT_INTERMEDIATE_BLOCK = {
    "evo2_7b": 26,
    "evo2_20b": 23,
}
_MODEL_MAX_INTERMEDIATE_BLOCK = {
    "evo2_7b": 31,
    "evo2_20b": 23,
}
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


def default_intermediate_block_for_model(model_id: str) -> int:
    try:
        return int(_MODEL_DEFAULT_INTERMEDIATE_BLOCK[model_id])
    except KeyError as e:
        rendered = ", ".join(sorted(_SUPPORTED_MODELS))
        raise ConfigError(f"Evo2 feature bundle supports model.id values: {rendered}. Received '{model_id}'.") from e


def resolve_intermediate_selector(*, model_id: str, intermediate_block: int) -> SelectorResolution:
    if model_id not in _SUPPORTED_MODELS:
        rendered = ", ".join(sorted(_SUPPORTED_MODELS))
        raise ConfigError(f"Evo2 feature bundle supports model.id values: {rendered}. Received '{model_id}'.")
    requested_block = int(intermediate_block)
    max_block = int(_MODEL_MAX_INTERMEDIATE_BLOCK[model_id])
    if requested_block > max_block:
        default_block = default_intermediate_block_for_model(model_id)
        if requested_block == 26 and default_block <= max_block:
            requested_block = default_block
        else:
            raise ConfigError(
                f"intermediate_block={requested_block} is unavailable for model_id='{model_id}'. "
                f"Supported block range is 0..{max_block}."
            )
    selector = canonical_selector_for_block(requested_block)
    return SelectorResolution(
        intermediate_block=requested_block,
        intermediate_selector=selector,
        provider_layer=provider_layer_for_block(requested_block),
    )
