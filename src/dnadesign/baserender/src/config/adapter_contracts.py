"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/baserender/src/config/adapter_contracts.py

Canonical adapter descriptor registry used by job parsing and public inspection.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from typing import Any, Callable, Literal

from dnadesign.contracts.visual import (
    ScarNickVisualV1,
    SequenceEvidenceMapV1,
    SnapbackVisualV1,
    ThreeWayJunctionReviewV1,
    YiuHairpinTopologyV1,
    YiuLinearStateV1,
    YiuPayloadVisualV1,
    YiuTopologyCartoonV1,
)

from ..core import ContractError, Record, SchemaError, ensure, reject_unknown_keys, require_one_of
from .job_contracts import THREE_WAY_JUNCTION_REVIEW_INPUT_ENVELOPE, InputEnvelope

PolicyNormalizer = Callable[[Mapping[str, Any], str], dict[str, Any]]
AdapterFactory = Callable[[Any, str], Any]
AdapterPathResolver = Callable[[str, Any], str]


def _normalize_policies_passthrough(policies: Mapping[str, Any], _ctx: str) -> dict[str, Any]:
    return dict(policies)


def _normalize_densegen_policies(policies: Mapping[str, Any], ctx: str) -> dict[str, Any]:
    parsed = dict(policies)
    if "ambiguous" in parsed:
        require_one_of(
            str(parsed["ambiguous"]).lower(),
            {"error", "first", "last", "drop"},
            f"{ctx}.ambiguous",
        )
    if "offset_mode" in parsed:
        require_one_of(
            str(parsed["offset_mode"]).lower(),
            {"auto", "zero_based", "one_based"},
            f"{ctx}.offset_mode",
        )
    if "on_missing_kmer" in parsed:
        require_one_of(
            str(parsed["on_missing_kmer"]).lower(),
            {"error", "skip_entry"},
            f"{ctx}.on_missing_kmer",
        )
    if "on_invalid_row" in parsed:
        require_one_of(
            str(parsed["on_invalid_row"]).lower(),
            {"skip", "error"},
            f"{ctx}.on_invalid_row",
        )
    if "min_per_record" in parsed:
        value = parsed["min_per_record"]
        if isinstance(value, bool):
            raise SchemaError(f"{ctx}.min_per_record must be int")
        try:
            min_per_record = int(value)
        except Exception as exc:
            raise SchemaError(f"{ctx}.min_per_record must be int") from exc
        ensure(min_per_record >= 0, f"{ctx}.min_per_record must be >= 0", SchemaError)
        parsed["min_per_record"] = min_per_record
    if "require_non_null_cols" in parsed:
        cols = parsed["require_non_null_cols"]
        if not isinstance(cols, (list, tuple)):
            raise SchemaError(f"{ctx}.require_non_null_cols must be a list")
        parsed["require_non_null_cols"] = [str(c) for c in cols]
    if "overlay_text_template" in parsed:
        template = parsed["overlay_text_template"]
        if not isinstance(template, str) or not template.strip():
            raise SchemaError(f"{ctx}.overlay_text_template must be a non-empty string")
        parsed["overlay_text_template"] = template
    for key in ("zero_as_unspecified", "require_non_empty"):
        if key in parsed:
            val = parsed[key]
            if not isinstance(val, bool):
                raise SchemaError(f"{ctx}.{key} must be bool")
            parsed[key] = val
    return parsed


def _normalize_usr_genbank_policies(policies: Mapping[str, Any], ctx: str) -> dict[str, Any]:
    parsed = dict(policies)
    if "on_invalid_row" in parsed:
        require_one_of(
            str(parsed["on_invalid_row"]).lower(),
            {"skip", "error"},
            f"{ctx}.on_invalid_row",
        )
    if "min_per_record" in parsed:
        value = parsed["min_per_record"]
        if isinstance(value, bool):
            raise SchemaError(f"{ctx}.min_per_record must be int")
        try:
            min_per_record = int(value)
        except Exception as exc:
            raise SchemaError(f"{ctx}.min_per_record must be int") from exc
        ensure(min_per_record >= 0, f"{ctx}.min_per_record must be >= 0", SchemaError)
        parsed["min_per_record"] = min_per_record
    for key in ("require_non_empty", "include_untyped_features"):
        if key in parsed:
            val = parsed[key]
            if not isinstance(val, bool):
                raise SchemaError(f"{ctx}.{key} must be bool")
            parsed[key] = val
    if "overlay_text_template" in parsed:
        template = parsed["overlay_text_template"]
        if not isinstance(template, str) or not template.strip():
            raise SchemaError(f"{ctx}.overlay_text_template must be a non-empty string")
        parsed["overlay_text_template"] = template
    return parsed


def _normalize_cruncher_policies(policies: Mapping[str, Any], ctx: str) -> dict[str, Any]:
    parsed = dict(policies)
    if "on_missing_hit" in parsed:
        require_one_of(
            str(parsed["on_missing_hit"]).lower(),
            {"error", "skip"},
            f"{ctx}.on_missing_hit",
        )
    if "on_missing_pwm" in parsed:
        require_one_of(
            str(parsed["on_missing_pwm"]).lower(),
            {"error", "skip_effect"},
            f"{ctx}.on_missing_pwm",
        )
    return parsed


def _build_densegen(cfg: Any, alphabet: str) -> Any:
    from ..adapters.densegen_tfbs import DensegenTfbsAdapter

    return DensegenTfbsAdapter(columns=cfg.columns, policies=cfg.policies, alphabet=alphabet)


def _build_generic(cfg: Any, alphabet: str) -> Any:
    from ..adapters.generic_features import GenericFeaturesAdapter

    return GenericFeaturesAdapter(columns=cfg.columns, policies=cfg.policies, alphabet=alphabet)


def _build_usr_genbank(cfg: Any, alphabet: str) -> Any:
    from ..adapters.usr_genbank_annotations_v1 import UsrGenbankAnnotationsV1Adapter

    return UsrGenbankAnnotationsV1Adapter(columns=cfg.columns, policies=cfg.policies, alphabet=alphabet)


def _build_cruncher(cfg: Any, alphabet: str) -> Any:
    from ..adapters.cruncher_best_window import CruncherBestWindowAdapter

    return CruncherBestWindowAdapter.from_config(columns=cfg.columns, policies=cfg.policies, alphabet=alphabet)


def _build_sequence_windows(cfg: Any, alphabet: str) -> Any:
    from ..adapters.sequence_windows_v1 import SequenceWindowsV1Adapter

    return SequenceWindowsV1Adapter(columns=cfg.columns, policies=cfg.policies, alphabet=alphabet)


def _build_sequence_evidence_map(cfg: Any, alphabet: str) -> Any:
    from ..adapters.sequence_evidence_map_v1 import SequenceEvidenceMapV1Adapter

    return SequenceEvidenceMapV1Adapter(columns=cfg.columns, policies=cfg.policies, alphabet=alphabet)


def _build_scar_nick_visual(cfg: Any, alphabet: str) -> Any:
    from ..adapters.scar_nick_visual_v1 import ScarNickVisualV1Adapter

    return ScarNickVisualV1Adapter(columns=cfg.columns, policies=cfg.policies, alphabet=alphabet)


def _build_snapback_visual(cfg: Any, alphabet: str) -> Any:
    from ..adapters.snapback_visual_v1 import SnapbackVisualV1Adapter

    return SnapbackVisualV1Adapter(columns=cfg.columns, policies=cfg.policies, alphabet=alphabet)


def _build_three_way_junction_review(cfg: Any, alphabet: str) -> Any:
    from ..adapters.three_way_junction_review_v1 import ThreeWayJunctionReviewV1Adapter

    return ThreeWayJunctionReviewV1Adapter(columns=cfg.columns, policies=cfg.policies, alphabet=alphabet)


def _build_duplex_sequence(cfg: Any, alphabet: str) -> Any:
    from ..adapters.duplex_sequence_v1 import DuplexSequenceV1Adapter

    return DuplexSequenceV1Adapter(columns=cfg.columns, policies=cfg.policies, alphabet=alphabet)


def _build_hairpin_topology(cfg: Any, alphabet: str) -> Any:
    from ..adapters.hairpin_topology_v1 import HairpinTopologyV1Adapter

    return HairpinTopologyV1Adapter(columns=cfg.columns, policies=cfg.policies, alphabet=alphabet)


def _build_yiu_linear_state(cfg: Any, alphabet: str) -> Any:
    from ..adapters.yiu_linear_state_v1 import YiuLinearStateV1Adapter

    return YiuLinearStateV1Adapter(columns=cfg.columns, policies=cfg.policies, alphabet=alphabet)


def _build_yiu_payload_visual(cfg: Any, alphabet: str) -> Any:
    from ..adapters.yiu_payload_visual_v1 import YiuPayloadVisualV1Adapter

    return YiuPayloadVisualV1Adapter(columns=cfg.columns, policies=cfg.policies, alphabet=alphabet)


def _build_yiu_hairpin_topology(cfg: Any, alphabet: str) -> Any:
    from ..adapters.yiu_hairpin_topology_v1 import YiuHairpinTopologyV1Adapter

    return YiuHairpinTopologyV1Adapter(columns=cfg.columns, policies=cfg.policies, alphabet=alphabet)


def _build_yiu_topology_cartoon(cfg: Any, alphabet: str) -> Any:
    from ..adapters.yiu_topology_cartoon_v1 import YiuTopologyCartoonV1Adapter

    return YiuTopologyCartoonV1Adapter(columns=cfg.columns, policies=cfg.policies, alphabet=alphabet)


@dataclass(frozen=True)
class AdapterDescriptor:
    kind: str
    owner_tool: str | None
    contract_kind: str
    schema_model: type | None
    supported_renderers: tuple[str, ...]
    supported_alphabets: tuple[str, ...]
    factory: AdapterFactory
    docs_slug: str
    allowed_config_columns: tuple[str, ...]
    required_config_columns: tuple[str, ...]
    required_source_columns: tuple[str, ...]
    optional_source_columns: tuple[str, ...] = ()
    allowed_policy_keys: tuple[str, ...] = ()
    resolved_path_columns: tuple[str, ...] = ()
    normalize_policies: PolicyNormalizer = _normalize_policies_passthrough
    sensitivity: Literal["public", "private"] = "public"
    input_envelope: InputEnvelope | None = None
    output_kinds: tuple[Literal["images", "video"], ...] = ("images", "video")
    image_output_modes: tuple[Literal["directory", "single_file"], ...] = (
        "directory",
        "single_file",
    )
    max_grid_records: int | None = None
    validation_scope: Literal["row", "document"] = "row"


AdapterContract = AdapterDescriptor


_DENSEGEN_POLICY_KEYS = (
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

_USR_GENBANK_POLICY_KEYS = (
    "on_invalid_row",
    "require_non_empty",
    "min_per_record",
    "include_untyped_features",
    "overlay_text_template",
)

_CRUNCHER_POLICY_KEYS = ("on_missing_hit", "on_missing_pwm")


ADAPTER_DESCRIPTORS: dict[str, AdapterDescriptor] = {
    "densegen_tfbs": AdapterDescriptor(
        kind="densegen_tfbs",
        owner_tool="densegen",
        contract_kind="densegen_tfbs",
        schema_model=None,
        supported_renderers=("sequence_rows",),
        supported_alphabets=("DNA",),
        factory=_build_densegen,
        docs_slug="densegen-tfbs",
        allowed_config_columns=("sequence", "annotations", "promoter_detail", "id", "overlay_text", "video_subtitle"),
        required_config_columns=("sequence", "annotations"),
        required_source_columns=("sequence", "annotations"),
        optional_source_columns=("promoter_detail", "id", "overlay_text", "video_subtitle"),
        allowed_policy_keys=_DENSEGEN_POLICY_KEYS,
        normalize_policies=_normalize_densegen_policies,
    ),
    "generic_features": AdapterDescriptor(
        kind="generic_features",
        owner_tool=None,
        contract_kind="generic_features",
        schema_model=None,
        supported_renderers=("sequence_rows",),
        supported_alphabets=("DNA", "IUPAC_DNA"),
        factory=_build_generic,
        docs_slug="generic-features",
        allowed_config_columns=("sequence", "features", "effects", "display", "id"),
        required_config_columns=("sequence", "features"),
        required_source_columns=("sequence", "features"),
        optional_source_columns=("effects", "display", "id"),
    ),
    "usr_genbank_annotations_v1": AdapterDescriptor(
        kind="usr_genbank_annotations_v1",
        owner_tool="usr",
        contract_kind="usr_genbank_annotations_v1",
        schema_model=None,
        supported_renderers=("sequence_rows",),
        supported_alphabets=("DNA", "IUPAC_DNA"),
        factory=_build_usr_genbank,
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
        optional_source_columns=("id", "overlay_text", "video_subtitle", "source_file", "product_kind"),
        allowed_policy_keys=_USR_GENBANK_POLICY_KEYS,
        normalize_policies=_normalize_usr_genbank_policies,
    ),
    "cruncher_best_window": AdapterDescriptor(
        kind="cruncher_best_window",
        owner_tool="cruncher",
        contract_kind="cruncher_best_window",
        schema_model=None,
        supported_renderers=("sequence_rows",),
        supported_alphabets=("DNA",),
        factory=_build_cruncher,
        docs_slug="cruncher-best-window",
        allowed_config_columns=(
            "sequence",
            "id",
            "hits_path",
            "hits_elite_id",
            "hits_tf",
            "hits_start",
            "hits_strand",
            "hits_window_seq",
            "hits_core_seq",
            "config_path",
        ),
        required_config_columns=("sequence", "id", "hits_path", "config_path"),
        required_source_columns=("sequence", "id"),
        allowed_policy_keys=_CRUNCHER_POLICY_KEYS,
        resolved_path_columns=("hits_path", "config_path"),
        normalize_policies=_normalize_cruncher_policies,
    ),
    "sequence_windows_v1": AdapterDescriptor(
        kind="sequence_windows_v1",
        owner_tool=None,
        contract_kind="sequence_windows_v1",
        schema_model=None,
        supported_renderers=("sequence_rows",),
        supported_alphabets=("DNA", "IUPAC_DNA"),
        factory=_build_sequence_windows,
        docs_slug="sequence-windows-v1",
        allowed_config_columns=("id", "sequence", "regulator_windows", "motifs", "display"),
        required_config_columns=("sequence", "regulator_windows"),
        required_source_columns=("sequence", "regulator_windows"),
        optional_source_columns=("id", "motifs", "display"),
    ),
    "sequence_evidence_map_v1": AdapterDescriptor(
        kind="sequence_evidence_map_v1",
        owner_tool=None,
        contract_kind="sequence_evidence_map_v1",
        schema_model=SequenceEvidenceMapV1,
        supported_renderers=("nucleotide_evidence_map",),
        supported_alphabets=("DNA", "IUPAC_DNA"),
        factory=_build_sequence_evidence_map,
        docs_slug="sequence-evidence-map-v1",
        allowed_config_columns=(),
        required_config_columns=(),
        required_source_columns=(),
    ),
    "scar_nick_visual_v1": AdapterDescriptor(
        kind="scar_nick_visual_v1",
        owner_tool="scar_nick",
        contract_kind="scar_nick_visual_v1",
        schema_model=ScarNickVisualV1,
        supported_renderers=("nucleotide_evidence_map",),
        supported_alphabets=("DNA", "IUPAC_DNA"),
        factory=_build_scar_nick_visual,
        docs_slug="scar-nick-visual-v1",
        allowed_config_columns=(),
        required_config_columns=(),
        required_source_columns=(),
    ),
    "snapback_visual_v1": AdapterDescriptor(
        kind="snapback_visual_v1",
        owner_tool="snapback",
        contract_kind="snapback_visual_v1",
        schema_model=SnapbackVisualV1,
        supported_renderers=("snapback_map",),
        supported_alphabets=("DNA", "IUPAC_DNA"),
        factory=_build_snapback_visual,
        docs_slug="snapback-visual-v1",
        allowed_config_columns=(),
        required_config_columns=(),
        required_source_columns=(),
    ),
    "three_way_junction_review_v1": AdapterDescriptor(
        kind="three_way_junction_review_v1",
        owner_tool="trijunction",
        contract_kind="three_way_junction_review_v1",
        schema_model=ThreeWayJunctionReviewV1,
        supported_renderers=("three_way_junction_review",),
        supported_alphabets=("DNA",),
        factory=_build_three_way_junction_review,
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
    "duplex_sequence_v1": AdapterDescriptor(
        kind="duplex_sequence_v1",
        owner_tool="cassette",
        contract_kind="duplex_sequence_v1",
        schema_model=None,
        supported_renderers=("sequence_rows",),
        supported_alphabets=("DNA", "IUPAC_DNA"),
        factory=_build_duplex_sequence,
        docs_slug="duplex-sequence-v1",
        allowed_config_columns=(),
        required_config_columns=(),
        required_source_columns=(),
    ),
    "hairpin_topology_v1": AdapterDescriptor(
        kind="hairpin_topology_v1",
        owner_tool="cassette",
        contract_kind="hairpin_topology_v1",
        schema_model=None,
        supported_renderers=("hairpin_cartoon",),
        supported_alphabets=("DNA",),
        factory=_build_hairpin_topology,
        docs_slug="hairpin-topology-v1",
        allowed_config_columns=(),
        required_config_columns=(),
        required_source_columns=(),
    ),
    "yiu_linear_state_v1": AdapterDescriptor(
        kind="yiu_linear_state_v1",
        owner_tool="yiu",
        contract_kind="yiu_linear_state_v1",
        schema_model=YiuLinearStateV1,
        supported_renderers=("sequence_rows",),
        supported_alphabets=("DNA", "IUPAC_DNA"),
        factory=_build_yiu_linear_state,
        docs_slug="yiu-linear-state-v1",
        allowed_config_columns=(),
        required_config_columns=(),
        required_source_columns=(),
    ),
    "yiu_payload_visual_v1": AdapterDescriptor(
        kind="yiu_payload_visual_v1",
        owner_tool="yiu",
        contract_kind="yiu_payload_visual_v1",
        schema_model=YiuPayloadVisualV1,
        supported_renderers=("nucleotide_evidence_map",),
        supported_alphabets=("DNA", "IUPAC_DNA"),
        factory=_build_yiu_payload_visual,
        docs_slug="yiu-payload-visual-v1",
        allowed_config_columns=(),
        required_config_columns=(),
        required_source_columns=(),
    ),
    "yiu_hairpin_topology_v1": AdapterDescriptor(
        kind="yiu_hairpin_topology_v1",
        owner_tool="yiu",
        contract_kind="yiu_hairpin_topology_v1",
        schema_model=YiuHairpinTopologyV1,
        supported_renderers=("hairpin_cartoon",),
        supported_alphabets=("DNA",),
        factory=_build_yiu_hairpin_topology,
        docs_slug="yiu-hairpin-topology-v1",
        allowed_config_columns=(),
        required_config_columns=(),
        required_source_columns=(),
    ),
    "yiu_topology_cartoon_v1": AdapterDescriptor(
        kind="yiu_topology_cartoon_v1",
        owner_tool="yiu",
        contract_kind="yiu_topology_cartoon_v1",
        schema_model=YiuTopologyCartoonV1,
        supported_renderers=("topology_cartoon",),
        supported_alphabets=("DNA", "IUPAC_DNA"),
        factory=_build_yiu_topology_cartoon,
        docs_slug="yiu-topology-cartoon-v1",
        allowed_config_columns=(),
        required_config_columns=(),
        required_source_columns=(),
    ),
}


def adapter_kinds() -> set[str]:
    return set(ADAPTER_DESCRIPTORS.keys())


def adapter_descriptors() -> tuple[AdapterDescriptor, ...]:
    return tuple(ADAPTER_DESCRIPTORS[kind] for kind in sorted(ADAPTER_DESCRIPTORS))


def adapter_descriptor(kind: str) -> AdapterDescriptor:
    descriptor = ADAPTER_DESCRIPTORS.get(kind)
    if descriptor is None:
        raise SchemaError(f"Unsupported adapter kind: {kind}")
    return descriptor


def adapter_contract(kind: str) -> AdapterContract:
    return adapter_descriptor(kind)


def validate_adapter_output_policy(
    adapter_kind: str,
    *,
    output_kind: Literal["images", "video"],
    image_output_mode: Literal["directory", "single_file"] | None = None,
) -> None:
    """Enforce one adapter's publication policy without depending on job types."""

    descriptor = adapter_contract(adapter_kind)
    if output_kind not in descriptor.output_kinds:
        allowed = ", ".join(descriptor.output_kinds)
        raise SchemaError(f"adapter {adapter_kind!r} only supports output kinds: {allowed}")
    if output_kind != "images":
        return
    if image_output_mode is None:
        raise SchemaError("image_output_mode is required for images output")
    if image_output_mode not in descriptor.image_output_modes:
        if descriptor.image_output_modes == ("directory",):
            raise SchemaError(
                f"adapter {adapter_kind!r} requires a directory for images output; single-file images are not supported"
            )
        allowed = ", ".join(descriptor.image_output_modes)
        raise SchemaError(
            f"adapter {adapter_kind!r} does not support images output mode {image_output_mode!r}; "
            f"supported modes: {allowed}"
        )


def _record_adapter_kind(record: Record, *, record_index: int) -> str | None:
    raw_kind = record.meta.get("adapter")
    if raw_kind is None:
        return None
    if not isinstance(raw_kind, str) or not raw_kind.strip():
        raise SchemaError(f"records[{record_index}].meta.adapter must be a non-empty string")
    kind = raw_kind.strip()
    adapter_contract(kind)
    return kind


def adapter_grid_record_limit(records: Iterable[Record]) -> int | None:
    """Return the strictest contact-sheet limit declared by record origins."""

    limit: int | None = None
    for record_index, record in enumerate(records):
        adapter_kind = _record_adapter_kind(record, record_index=record_index)
        if adapter_kind is None:
            continue
        adapter_limit = adapter_contract(adapter_kind).max_grid_records
        if adapter_limit is not None:
            limit = adapter_limit if limit is None else min(limit, adapter_limit)
    return limit


def validate_records_output_policy(
    records: Iterable[Record],
    *,
    output_kind: Literal["images", "video"],
    image_output_mode: Literal["directory", "single_file"] | None = None,
) -> None:
    """Preserve adapter publication constraints on direct writer surfaces."""

    for record_index, record in enumerate(records):
        adapter_kind = _record_adapter_kind(record, record_index=record_index)
        if adapter_kind is not None:
            validate_adapter_output_policy(
                adapter_kind,
                output_kind=output_kind,
                image_output_mode=image_output_mode,
            )


def normalize_adapter_config(
    *,
    kind: Any,
    columns: Mapping[str, Any],
    policies: Mapping[str, Any],
    alphabet: str | None = None,
    resolve_path: AdapterPathResolver | None = None,
) -> tuple[str, dict[str, Any], dict[str, Any]]:
    try:
        parsed_kind = str(kind).strip()
        require_one_of(parsed_kind, adapter_kinds(), "input.adapter.kind")
        contract = adapter_contract(parsed_kind)

        reject_unknown_keys(columns, set(contract.allowed_config_columns), "input.adapter.columns")
        missing = sorted(set(contract.required_config_columns) - set(columns.keys()))
        if missing:
            raise SchemaError(f"input.adapter.columns missing required keys for {parsed_kind}: {missing}")

        parsed_columns = dict(columns)
        if resolve_path is not None:
            for key in contract.resolved_path_columns:
                if key in parsed_columns and parsed_columns[key] is not None:
                    parsed_columns[key] = resolve_path(key, parsed_columns[key])

        reject_unknown_keys(policies, set(contract.allowed_policy_keys), "input.adapter.policies")
        parsed_policies = contract.normalize_policies(policies, "input.adapter.policies")

        if alphabet is not None and alphabet not in contract.supported_alphabets:
            allowed = ", ".join(sorted(contract.supported_alphabets))
            raise SchemaError(
                "input.adapter.kind "
                f"{parsed_kind!r} is not compatible with input.alphabet {alphabet!r}; "
                f"supported input.alphabet values: {allowed}"
            )

        return parsed_kind, parsed_columns, parsed_policies
    except ContractError as exc:
        raise SchemaError(str(exc)) from exc
