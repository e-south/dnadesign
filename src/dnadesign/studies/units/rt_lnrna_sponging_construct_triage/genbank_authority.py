"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/studies/units/rt_lnrna_sponging_construct_triage/genbank_authority.py

GenBank source-authority validation for the RT-lnRNA sponging construct triage.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from dnadesign.usr import BiopythonGenBankParser

_STUDY_DIR = Path("docs/studies/rt_lnrna_sponging_construct_triage")
_DEFAULT_REGISTRY_PATH = _STUDY_DIR / "workbench/provenance/genbank-source-authority.yaml"
_RT_FEATURE_LABEL = "ECD_00831"


@dataclass(frozen=True)
class GenBankAuthoritySource:
    source_id: str
    path: str
    role: str
    required_unique_labels: tuple[str, ...] = ()
    required_label_counts: dict[str, int] = field(default_factory=dict)


@dataclass(frozen=True)
class GenBankSubsequenceCheck:
    subject_source_id: str
    container_source_id: str


@dataclass(frozen=True)
class GenBankAuthorityRegistry:
    sources: tuple[GenBankAuthoritySource, ...]
    rt_cds_identity_source_ids: tuple[str, ...] = ()
    subsequence_checks: tuple[GenBankSubsequenceCheck, ...] = ()


@dataclass(frozen=True)
class GenBankFeatureAudit:
    label: str
    feature_type: str
    location_raw: str
    span_1: tuple[int, int] | None
    strand: int | None
    sequence: str
    feature_id: str


@dataclass(frozen=True)
class GenBankSourceAudit:
    source_id: str
    role: str
    path: str
    source_sha256: str
    record_id: str | None
    record_name: str | None
    length: int
    topology: str | None
    molecule_type: str | None
    sequence: str
    key_features: tuple[GenBankFeatureAudit, ...]

    def feature(self, label: str) -> GenBankFeatureAudit:
        matches = [feature for feature in self.key_features if feature.label == label]
        if len(matches) != 1:
            raise KeyError(f"{self.source_id}: expected one audited feature label {label!r}, found {len(matches)}")
        return matches[0]


@dataclass(frozen=True)
class GenBankAuthorityAudit:
    sources: tuple[GenBankSourceAudit, ...]
    errors: tuple[str, ...]
    rt_cds_identity_source_ids: tuple[str, ...] = ()

    @property
    def ok(self) -> bool:
        return not self.errors

    def source(self, source_id: str) -> GenBankSourceAudit:
        matches = [source for source in self.sources if source.source_id == source_id]
        if len(matches) != 1:
            raise KeyError(f"expected one audited source id {source_id!r}, found {len(matches)}")
        return matches[0]


def load_source_authority_registry(path: Path) -> GenBankAuthorityRegistry:
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{path}: expected mapping payload")

    sources = tuple(_source_from_payload(source) for source in _as_list(payload.get("sources"), key="sources"))
    rt_cds_identity = payload.get("rt_cds_identity") or {}
    if not isinstance(rt_cds_identity, dict):
        raise ValueError(f"{path}: rt_cds_identity must be a mapping when provided")

    subsequence_checks = tuple(
        GenBankSubsequenceCheck(
            subject_source_id=str(check["subject_source_id"]),
            container_source_id=str(check["container_source_id"]),
        )
        for check in _as_list(payload.get("subsequence_checks", []), key="subsequence_checks")
    )
    return GenBankAuthorityRegistry(
        sources=sources,
        rt_cds_identity_source_ids=tuple(str(source_id) for source_id in rt_cds_identity.get("source_ids", ())),
        subsequence_checks=subsequence_checks,
    )


def run_default_authority_audit(*, repo_root: Path | None = None) -> GenBankAuthorityAudit:
    root = _resolve_repo_root(repo_root)
    registry = load_source_authority_registry(root / _DEFAULT_REGISTRY_PATH)
    return validate_genbank_authority_registry(repo_root=root, registry=registry)


def validate_genbank_authority_registry(
    *,
    repo_root: Path,
    registry: GenBankAuthorityRegistry,
) -> GenBankAuthorityAudit:
    parser = BiopythonGenBankParser()
    errors: list[str] = []
    audits: list[GenBankSourceAudit] = []
    parsed_records: dict[str, Any] = {}

    for source in registry.sources:
        path = (repo_root / source.path).resolve()
        if not path.exists():
            errors.append(f"{source.source_id}: GenBank path does not exist: {source.path}")
            continue
        records = parser.parse_file(path)
        if len(records) != 1:
            errors.append(f"{source.source_id}: expected exactly one GenBank record, found {len(records)}")
            continue

        record = records[0]
        parsed_records[source.source_id] = record
        audited_features: list[GenBankFeatureAudit] = []
        for label in source.required_unique_labels:
            matches = _features_by_label(record, label)
            if not matches:
                errors.append(f"{source.source_id}: missing required feature label {label!r}")
                continue
            if len(matches) > 1:
                errors.append(f"{source.source_id}: ambiguous required feature label {label!r}: {len(matches)} matches")
                continue
            audited_features.append(_feature_audit(record, matches[0]))

        for label, expected_count in source.required_label_counts.items():
            matches = _features_by_label(record, label)
            if len(matches) != expected_count:
                errors.append(
                    f"{source.source_id}: expected {expected_count} feature label {label!r} matches, "
                    f"found {len(matches)}"
                )
                continue
            audited_features.extend(_feature_audit(record, feature) for feature in matches)

        audits.append(
            GenBankSourceAudit(
                source_id=source.source_id,
                role=source.role,
                path=source.path,
                source_sha256=record.source_sha256,
                record_id=record.record_id,
                record_name=record.record_name,
                length=len(record.sequence),
                topology=record.topology,
                molecule_type=record.molecule_type,
                sequence=record.sequence,
                key_features=tuple(audited_features),
            )
        )

    _validate_rt_cds_identity(
        source_ids=registry.rt_cds_identity_source_ids,
        parsed_records=parsed_records,
        errors=errors,
    )
    _validate_subsequence_checks(
        checks=registry.subsequence_checks,
        parsed_records=parsed_records,
        errors=errors,
    )
    return GenBankAuthorityAudit(
        sources=tuple(audits),
        errors=tuple(errors),
        rt_cds_identity_source_ids=registry.rt_cds_identity_source_ids if not errors else (),
    )


def _source_from_payload(payload: object) -> GenBankAuthoritySource:
    if not isinstance(payload, dict):
        raise ValueError("source entries must be mappings")
    required_label_counts = payload.get("required_label_counts") or {}
    if not isinstance(required_label_counts, dict):
        raise ValueError(f"{payload.get('source_id')}: required_label_counts must be a mapping")
    return GenBankAuthoritySource(
        source_id=str(payload["source_id"]),
        path=str(payload["path"]),
        role=str(payload["role"]),
        required_unique_labels=tuple(str(label) for label in payload.get("required_unique_labels", ())),
        required_label_counts={str(label): int(count) for label, count in required_label_counts.items()},
    )


def _as_list(value: object, *, key: str) -> list[object]:
    if not isinstance(value, list):
        raise ValueError(f"{key} must be a list")
    return value


def _resolve_repo_root(repo_root: Path | None) -> Path:
    if repo_root is not None:
        return Path(repo_root).resolve()
    current = Path(__file__).resolve()
    for parent in current.parents:
        if (parent / "pyproject.toml").exists():
            return parent
    raise RuntimeError("repo root not found")


def _features_by_label(record: Any, label: str) -> list[Any]:
    return [feature for feature in record.features if feature.label == label]


def _feature_audit(record: Any, feature: Any) -> GenBankFeatureAudit:
    span_1 = None
    sequence = ""
    if feature.start_0 is not None and feature.end_0 is not None:
        span_1 = (feature.start_0 + 1, feature.end_0)
        sequence = record.sequence[feature.start_0 : feature.end_0]
    return GenBankFeatureAudit(
        label=str(feature.label),
        feature_type=feature.feature_type,
        location_raw=feature.location_raw,
        span_1=span_1,
        strand=feature.strand,
        sequence=sequence,
        feature_id=feature.feature_id,
    )


def _validate_rt_cds_identity(
    *,
    source_ids: tuple[str, ...],
    parsed_records: dict[str, Any],
    errors: list[str],
) -> None:
    if not source_ids:
        return

    sequences: dict[str, str] = {}
    translations: dict[str, str] = {}
    for source_id in source_ids:
        record = parsed_records.get(source_id)
        if record is None:
            errors.append(f"{source_id}: missing parsed record for RT CDS identity check")
            continue
        rt_features = _features_by_label(record, _RT_FEATURE_LABEL)
        if len(rt_features) != 1:
            errors.append(f"{source_id}: expected one {_RT_FEATURE_LABEL!r} feature, found {len(rt_features)}")
            continue
        rt_feature = rt_features[0]
        if rt_feature.start_0 is None or rt_feature.end_0 is None:
            errors.append(f"{source_id}: {_RT_FEATURE_LABEL!r} feature lacks exact span")
            continue
        matching_cds = [
            feature
            for feature in record.features
            if feature.feature_type == "CDS"
            and feature.start_0 == rt_feature.start_0
            and feature.end_0 == rt_feature.end_0
            and feature.strand == rt_feature.strand
        ]
        if len(matching_cds) != 1:
            errors.append(
                f"{source_id}: expected one CDS matching {_RT_FEATURE_LABEL!r} span, found {len(matching_cds)}"
            )
            continue
        translation = _qualifier_value(matching_cds[0], "translation")
        if not translation:
            errors.append(f"{source_id}: matching RT CDS lacks translation qualifier")
            continue
        sequences[source_id] = record.sequence[rt_feature.start_0 : rt_feature.end_0]
        translations[source_id] = translation

    if len(set(sequences.values())) > 1:
        errors.append("RT CDS sequence identity check failed across GenBank sources")
    if len(set(translations.values())) > 1:
        errors.append("RT CDS translation identity check failed across GenBank sources")


def _validate_subsequence_checks(
    *,
    checks: tuple[GenBankSubsequenceCheck, ...],
    parsed_records: dict[str, Any],
    errors: list[str],
) -> None:
    for check in checks:
        subject = parsed_records.get(check.subject_source_id)
        container = parsed_records.get(check.container_source_id)
        if subject is None or container is None:
            errors.append(
                f"{check.subject_source_id}->{check.container_source_id}: missing parsed record for subsequence check"
            )
            continue
        if subject.sequence not in container.sequence:
            errors.append(f"{check.subject_source_id}: sequence is not contained in {check.container_source_id}")


def _qualifier_value(feature: Any, key: str) -> str | None:
    for qualifier in feature.qualifiers:
        if qualifier.key == key:
            return qualifier.value
    return None
