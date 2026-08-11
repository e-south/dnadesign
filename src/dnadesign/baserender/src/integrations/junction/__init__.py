"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/baserender/src/integrations/junction/__init__.py

Adapt Junction review records for BaseRender.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from ...core import InputEnvelope, RenderContractDescriptor
from ..contracts import AdapterDescriptor, IntegrationProvider

THREE_WAY_JUNCTION_REVIEW_INPUT_ENVELOPE = InputEnvelope(
    max_bytes=64 * 1024 * 1024,
    max_records=2_000,
    max_bases=10_000_000,
    base_field_path=("target", "sequence_5to3"),
    accepted_input_kinds=("json",),
)


def _build_adapter(cfg, alphabet: str):
    from .review_v1 import ThreeWayJunctionReviewV1Adapter

    return ThreeWayJunctionReviewV1Adapter(columns=cfg.columns, policies=cfg.policies, alphabet=alphabet)


PROVIDER = IntegrationProvider(
    name="junction",
    adapters=(
        AdapterDescriptor(
            kind="three_way_junction_review_v1",
            owner_tool="junction",
            contract_kind="three_way_junction_review_v1",
            supported_renderers=("junction_annealed_fragments", "junction_three_way_assembly"),
            supported_alphabets=("DNA",),
            factory=_build_adapter,
            docs_slug="three-way-junction-review-v1",
            allowed_config_columns=(),
            required_config_columns=(),
            required_source_columns=(),
            sensitivity="private",
            input_envelope=THREE_WAY_JUNCTION_REVIEW_INPUT_ENVELOPE,
            output_kinds=("images",),
            image_output_modes=("directory",),
            max_grid_records=1,
            validation_scope="document",
        ),
    ),
    render_contracts=(
        RenderContractDescriptor(
            kind="three_way_junction_review_render_v1",
            schema_version=1,
            display_name="Three-way-junction review render contract",
            purpose="Selected annealed-fragment and three-way-assembly QA views from explicit design evidence.",
            accepted_renderers=("junction_annealed_fragments", "junction_three_way_assembly"),
            docs_slug="three-way-junction-review-render-v1",
            sensitivity="private",
            input_envelope=THREE_WAY_JUNCTION_REVIEW_INPUT_ENVELOPE,
        ),
    ),
)

__all__ = ["PROVIDER", "THREE_WAY_JUNCTION_REVIEW_INPUT_ENVELOPE"]
