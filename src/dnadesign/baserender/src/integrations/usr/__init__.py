"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/baserender/src/integrations/usr/__init__.py

Adapt USR GenBank annotations for BaseRender.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from ...core import RenderContractDescriptor, SchemaError, ensure, require_one_of
from ..contracts import AdapterDescriptor, IntegrationProvider, SequencePanelDefaults

_POLICY_KEYS = (
    "on_invalid_row",
    "require_non_empty",
    "min_per_record",
    "include_untyped_features",
    "overlay_text_template",
)


def _normalize_policies(policies: Mapping[str, Any], ctx: str) -> dict[str, Any]:
    parsed = dict(policies)
    if "on_invalid_row" in parsed:
        require_one_of(str(parsed["on_invalid_row"]).lower(), {"skip", "error"}, f"{ctx}.on_invalid_row")
    if "min_per_record" in parsed:
        value = parsed["min_per_record"]
        if isinstance(value, bool):
            raise SchemaError(f"{ctx}.min_per_record must be int")
        try:
            parsed["min_per_record"] = int(value)
        except Exception as exc:
            raise SchemaError(f"{ctx}.min_per_record must be int") from exc
        ensure(parsed["min_per_record"] >= 0, f"{ctx}.min_per_record must be >= 0", SchemaError)
    for key in ("require_non_empty", "include_untyped_features"):
        if key in parsed and not isinstance(parsed[key], bool):
            raise SchemaError(f"{ctx}.{key} must be bool")
    if "overlay_text_template" in parsed:
        template = parsed["overlay_text_template"]
        if not isinstance(template, str) or not template.strip():
            raise SchemaError(f"{ctx}.overlay_text_template must be a non-empty string")
    return parsed


def _build_adapter(cfg, alphabet: str):
    from .genbank_annotations_v1 import UsrGenbankAnnotationsV1Adapter

    return UsrGenbankAnnotationsV1Adapter(columns=cfg.columns, policies=cfg.policies, alphabet=alphabet)


PROVIDER = IntegrationProvider(
    name="usr",
    adapters=(
        AdapterDescriptor(
            kind="usr_genbank_annotations_v1",
            owner_tool="usr",
            contract_kind="usr_genbank_annotations_v1",
            supported_renderers=("sequence_rows",),
            supported_alphabets=("DNA", "IUPAC_DNA"),
            factory=_build_adapter,
            docs_slug="usr-genbank-annotations-v1",
            allowed_config_columns=(
                "sequence",
                "annotations",
                "id",
                "overlay_text",
                "video_subtitle",
                "source_file",
                "product_kind",
            ),
            required_config_columns=("sequence", "annotations"),
            required_source_columns=("sequence", "annotations"),
            optional_source_columns=(
                "id",
                "overlay_text",
                "video_subtitle",
                "source_file",
                "product_kind",
            ),
            allowed_policy_keys=_POLICY_KEYS,
            normalize_policies=_normalize_policies,
        ),
    ),
    sequence_panels=(
        SequencePanelDefaults(
            adapter_kind="usr_genbank_annotations_v1",
            supported_profiles=("promoter_compact_slide.v1",),
            columns=(
                ("sequence", "sequence"),
                ("annotations", "seq_annot__features"),
                ("id", "id"),
                ("overlay_text", "usr_label__primary"),
                ("source_file", "seq_annot__source_file"),
                ("product_kind", "derived__product_kind"),
            ),
            policies=(("overlay_text_template", "{overlay_text}"), ("on_invalid_row", "error")),
        ),
    ),
    render_contracts=(
        RenderContractDescriptor(
            kind="usr_genbank_annotation_render_v1",
            schema_version=1,
            display_name="USR GenBank annotation render contract",
            purpose="Linear sequence rows for USR records with declared GenBank feature overlays.",
            accepted_renderers=("sequence_rows",),
            docs_slug="usr-genbank-annotation-render-v1",
        ),
    ),
)

__all__ = ["PROVIDER"]
