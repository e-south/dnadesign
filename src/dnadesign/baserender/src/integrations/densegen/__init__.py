"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/baserender/src/integrations/densegen/__init__.py

Adapt DenseGen records for BaseRender.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from ...core import SchemaError, ensure, require_one_of
from ..contracts import AdapterDescriptor, IntegrationProvider, SequencePanelDefaults

_POLICY_KEYS = (
    "ambiguous",
    "offset_mode",
    "zero_as_unspecified",
    "on_missing_kmer",
    "require_non_empty",
    "min_per_record",
    "require_non_null_cols",
    "on_invalid_row",
    "overlay_text_template",
)


def _normalize_policies(policies: Mapping[str, Any], ctx: str) -> dict[str, Any]:
    parsed = dict(policies)
    if "ambiguous" in parsed:
        require_one_of(str(parsed["ambiguous"]).lower(), {"error", "first", "last", "drop"}, f"{ctx}.ambiguous")
    if "offset_mode" in parsed:
        require_one_of(
            str(parsed["offset_mode"]).lower(),
            {"auto", "zero_based", "one_based"},
            f"{ctx}.offset_mode",
        )
    if "on_missing_kmer" in parsed:
        require_one_of(str(parsed["on_missing_kmer"]).lower(), {"error", "skip_entry"}, f"{ctx}.on_missing_kmer")
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
    if "require_non_null_cols" in parsed:
        cols = parsed["require_non_null_cols"]
        if not isinstance(cols, (list, tuple)):
            raise SchemaError(f"{ctx}.require_non_null_cols must be a list")
        parsed["require_non_null_cols"] = [str(column) for column in cols]
    if "overlay_text_template" in parsed:
        template = parsed["overlay_text_template"]
        if not isinstance(template, str) or not template.strip():
            raise SchemaError(f"{ctx}.overlay_text_template must be a non-empty string")
    for key in ("zero_as_unspecified", "require_non_empty"):
        if key in parsed and not isinstance(parsed[key], bool):
            raise SchemaError(f"{ctx}.{key} must be bool")
    return parsed


def _build_adapter(cfg, alphabet: str):
    from .adapter import DensegenTfbsAdapter

    return DensegenTfbsAdapter(columns=cfg.columns, policies=cfg.policies, alphabet=alphabet)


PROVIDER = IntegrationProvider(
    name="densegen",
    adapters=(
        AdapterDescriptor(
            kind="densegen_tfbs",
            owner_tool="densegen",
            contract_kind="densegen_tfbs",
            supported_renderers=("sequence_rows",),
            supported_alphabets=("DNA",),
            factory=_build_adapter,
            docs_slug="densegen-tfbs",
            allowed_config_columns=(
                "sequence",
                "annotations",
                "promoter_detail",
                "id",
                "overlay_text",
                "video_subtitle",
            ),
            required_config_columns=("sequence", "annotations"),
            required_source_columns=("sequence", "annotations"),
            optional_source_columns=("promoter_detail", "id", "overlay_text", "video_subtitle"),
            allowed_policy_keys=_POLICY_KEYS,
            normalize_policies=_normalize_policies,
        ),
    ),
    sequence_panels=(
        SequencePanelDefaults(
            adapter_kind="densegen_tfbs",
            supported_profiles=("promoter_compact_slide.v1",),
            columns=(
                ("sequence", "sequence"),
                ("annotations", "densegen__used_tfbs_detail"),
                ("id", "id"),
            ),
            policies=(("on_invalid_row", "error"), ("require_non_empty", False)),
        ),
    ),
)

__all__ = ["PROVIDER"]
