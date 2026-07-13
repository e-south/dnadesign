"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/studies/units/stress_ethanol_cipro_growth/promoter_candidate_bindings/source_adapters.py

Typed adapters from study records to exact promoter alias rows.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from string import Formatter
from typing import Any

import pandas as pd

from .contracts import BindingSourceArtifact, PromoterCandidateBindingsError
from .source_io import file_sha256, read_parquet, source_artifact
from .source_registry import AliasSource, relative_config_path
from .synthesis_alias_sources import load_synthesis_alias_sources
from .values import require_columns, required_text


@dataclass(frozen=True)
class AliasSourceResult:
    alias_rows: tuple[dict[str, str], ...]
    source_artifacts: tuple[BindingSourceArtifact, ...]
    genbank_annotations: pd.DataFrame


def load_alias_source(repo_root: Path, source: AliasSource) -> AliasSourceResult:
    adapters = {
        "sequence_view_source_label.v1": _sequence_view_source_labels,
        "reference_label_aliases.v1": _reference_label_aliases,
        "synthesis_handoff.v1": _synthesis_handoff_aliases,
    }
    try:
        adapter = adapters[source.adapter]
    except KeyError as exc:
        raise PromoterCandidateBindingsError(
            f"Alias source {source.source_id!r} uses unsupported adapter {source.adapter!r}."
        ) from exc
    return adapter(repo_root, source)


def _sequence_view_source_labels(repo_root: Path, source: AliasSource) -> AliasSourceResult:
    config = _config(
        source,
        fields={"records_path", "views_path", "alias_namespace", "authority_dataset_id"},
    )
    records_path = repo_root / relative_config_path(config["records_path"], context="records_path")
    views_path = repo_root / relative_config_path(config["views_path"], context="views_path")
    records = read_parquet(records_path)
    views = read_parquet(views_path)
    require_columns(records, ("id", "sequence"), label=source.source_id)
    require_columns(views, ("sequence_id", "source_label"), label=source.source_id)
    by_id = records.assign(id=records["id"].astype(str)).set_index("id")
    authority_sha256 = file_sha256(records_path)
    rows: list[dict[str, str]] = []
    for record in views.to_dict(orient="records"):
        candidate_id = required_text(record["sequence_id"], field="sequence view candidate ID")
        if candidate_id not in by_id.index:
            raise PromoterCandidateBindingsError(
                f"Alias source {source.source_id!r} references missing candidate {candidate_id!r}."
            )
        alias = required_text(record["source_label"], field="sequence view source label")
        rows.append(
            _alias_row(
                namespace=required_text(config["alias_namespace"], field="alias namespace"),
                alias=alias,
                display_label=alias,
                candidate_id=candidate_id,
                sequence=required_text(by_id.loc[candidate_id, "sequence"], field="source sequence"),
                authority_dataset_id=required_text(config["authority_dataset_id"], field="authority dataset ID"),
                authority_id=candidate_id,
                authority_sha256=authority_sha256,
            )
        )
    return AliasSourceResult(
        alias_rows=tuple(rows),
        source_artifacts=(
            source_artifact(repo_root, f"{source.source_id}:records", records_path),
            source_artifact(repo_root, f"{source.source_id}:views", views_path),
        ),
        genbank_annotations=pd.DataFrame(),
    )


def _reference_label_aliases(repo_root: Path, source: AliasSource) -> AliasSourceResult:
    config = _config(
        source,
        fields={"records_path", "annotations_path", "authority_dataset_id", "aliases"},
    )
    records_path = repo_root / relative_config_path(config["records_path"], context="records_path")
    annotations_path = repo_root / relative_config_path(config["annotations_path"], context="annotations_path")
    records = read_parquet(records_path)
    annotations = read_parquet(annotations_path)
    require_columns(records, ("id", "sequence", "usr_label__primary"), label=source.source_id)
    require_columns(annotations, ("id",), label=f"{source.source_id} annotations")
    authority_sha256 = file_sha256(records_path)
    rows: list[dict[str, str]] = []
    candidate_ids: list[str] = []
    for index, raw in enumerate(_list(config["aliases"], context=f"{source.source_id}.aliases")):
        item = _mapping(raw, context=f"{source.source_id}.aliases[{index}]")
        if set(item) != {"source_label", "display_label", "names"}:
            raise PromoterCandidateBindingsError("Reference alias fields must be source_label, display_label, names.")
        source_label = required_text(item["source_label"], field="reference source label")
        hits = records.loc[records["usr_label__primary"].astype(str).eq(source_label)]
        if len(hits) != 1:
            raise PromoterCandidateBindingsError(
                f"Reference label {source_label!r} must resolve exactly once; found {len(hits)}."
            )
        record = hits.iloc[0]
        candidate_id = str(record["id"])
        candidate_ids.append(candidate_id)
        for name_index, raw_name in enumerate(_list(item["names"], context=f"{source_label}.names")):
            name = _mapping(raw_name, context=f"{source_label}.names[{name_index}]")
            if set(name) != {"namespace", "alias"}:
                raise PromoterCandidateBindingsError("Reference alias names require namespace and alias.")
            rows.append(
                _alias_row(
                    namespace=name["namespace"],
                    alias=name["alias"],
                    display_label=item["display_label"],
                    candidate_id=candidate_id,
                    sequence=record["sequence"],
                    authority_dataset_id=config["authority_dataset_id"],
                    authority_id=candidate_id,
                    authority_sha256=authority_sha256,
                )
            )
    selected_annotations = annotations.loc[annotations["id"].astype(str).isin(candidate_ids)].reset_index(drop=True)
    if set(selected_annotations["id"].astype(str)) != set(candidate_ids):
        raise PromoterCandidateBindingsError("Reference aliases are missing required GenBank annotation rows.")
    return AliasSourceResult(
        alias_rows=tuple(rows),
        source_artifacts=(
            source_artifact(repo_root, f"{source.source_id}:records", records_path),
            source_artifact(repo_root, f"{source.source_id}:annotations", annotations_path),
        ),
        genbank_annotations=selected_annotations,
    )


def _synthesis_handoff_aliases(repo_root: Path, source: AliasSource) -> AliasSourceResult:
    config = _config(
        source,
        fields={"record_path", "handoff_id", "authority_dataset_id", "aliases"},
    )
    record_path = relative_config_path(config["record_path"], context="record_path")
    handoff_id = required_text(config["handoff_id"], field="synthesis handoff ID")
    synthesis = load_synthesis_alias_sources(repo_root, record_path=record_path, handoff_id=handoff_id)
    templates = _alias_templates(config["aliases"])
    rows: list[dict[str, str]] = []
    for record in synthesis.to_dict(orient="records"):
        synthesis_name = required_text(record["synthesis_name"], field="synthesis name")
        for namespace, template in templates:
            rows.append(
                _alias_row(
                    namespace=namespace,
                    alias=template.format(synthesis_name=synthesis_name),
                    display_label=synthesis_name,
                    candidate_id=record["id"],
                    sequence=record["core_sequence"],
                    authority_dataset_id=config["authority_dataset_id"],
                    authority_id=record["id"],
                    authority_sha256=record["source_manifest_sha256"],
                )
            )
    artifacts = [source_artifact(repo_root, f"{source.source_id}:record", repo_root / record_path)]
    for record in synthesis[["campaign_slug", "source_manifest_path"]].drop_duplicates().to_dict(orient="records"):
        artifacts.append(
            source_artifact(
                repo_root,
                f"{source.source_id}:manifest:{record['campaign_slug']}",
                repo_root / str(record["source_manifest_path"]),
            )
        )
    return AliasSourceResult(tuple(rows), tuple(artifacts), pd.DataFrame())


def _alias_templates(value: object) -> tuple[tuple[str, str], ...]:
    templates: list[tuple[str, str]] = []
    for index, raw in enumerate(_list(value, context="synthesis aliases")):
        item = _mapping(raw, context=f"synthesis aliases[{index}]")
        if set(item) != {"namespace", "template"}:
            raise PromoterCandidateBindingsError("Synthesis alias entries require namespace and template.")
        template = required_text(item["template"], field="synthesis alias template")
        fields = [field for _, field, spec, conversion in Formatter().parse(template) if field]
        has_format_options = any(spec or conversion for _, _, spec, conversion in Formatter().parse(template))
        if fields != ["synthesis_name"] or has_format_options:
            raise PromoterCandidateBindingsError(
                "Synthesis alias template must contain exactly one unformatted {synthesis_name} field."
            )
        templates.append((required_text(item["namespace"], field="alias namespace"), template))
    return tuple(templates)


def _alias_row(**values: object) -> dict[str, str]:
    return {
        "alias_namespace": required_text(values["namespace"], field="alias namespace"),
        "alias": required_text(values["alias"], field="alias"),
        "display_label": required_text(values["display_label"], field="display label"),
        "candidate_id": required_text(values["candidate_id"], field="candidate ID"),
        "authority_sequence": required_text(values["sequence"], field="authority sequence"),
        "sequence_authority_dataset_id": required_text(
            values["authority_dataset_id"], field="sequence authority dataset ID"
        ),
        "sequence_authority_id": required_text(values["authority_id"], field="sequence authority ID"),
        "sequence_authority_sha256": required_text(values["authority_sha256"], field="sequence authority SHA-256"),
    }


def _config(source: AliasSource, *, fields: set[str]) -> dict[str, Any]:
    if set(source.config) != fields:
        raise PromoterCandidateBindingsError(
            f"Alias source {source.source_id!r} config fields must be exactly {sorted(fields)}."
        )
    return source.config


def _mapping(value: object, *, context: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise PromoterCandidateBindingsError(f"{context} must be a mapping.")
    return {str(key): item for key, item in value.items()}


def _list(value: object, *, context: str) -> list[Any]:
    if not isinstance(value, list) or not value:
        raise PromoterCandidateBindingsError(f"{context} must be a non-empty list.")
    return value


__all__ = ["AliasSourceResult", "load_alias_source"]
