"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/baserender/src/core/descriptors.py

Neutral capability descriptors shared by configuration and integrations.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from .envelopes import InputEnvelope


@dataclass(frozen=True)
class RenderContractDescriptor:
    kind: str
    schema_version: int
    display_name: str
    purpose: str
    accepted_renderers: tuple[str, ...]
    compatibility_aliases: tuple[str, ...] = ()
    docs_slug: str | None = None
    sensitivity: Literal["public", "private"] = "public"
    input_envelope: InputEnvelope | None = None


__all__ = ["RenderContractDescriptor"]
