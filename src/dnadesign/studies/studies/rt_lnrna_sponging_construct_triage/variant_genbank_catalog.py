"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/studies/studies/rt_lnrna_sponging_construct_triage/variant_genbank_catalog.py

GenBank-backed variant cataloging for RT-lnRNA sponging construct triage.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import yaml

from dnadesign.usr import BiopythonGenBankParser

_STUDY_DIR = Path("docs/studies/rt_lnrna_sponging_construct_triage")
_DEFAULT_METADATA_PATH = _STUDY_DIR / "workbench/provenance/retron-variant-genbank-metadata.yaml"
_DEFAULT_CATALOG_PATH = _STUDY_DIR / "workbench/provenance/retron-variant-genbank-catalog.yaml"
_BASE_TEMPLATE_LNRNA_SPAN_0 = (186, 359)
_BASE_TEMPLATE_RT_SPAN_0 = (524, 1487)
_TARGET_CONTEXT_START_0 = 56
_TARGET_CONTEXT_LENGTH_NT = 1600
_BASE_CONTEXT_END_0 = _TARGET_CONTEXT_START_0 + _TARGET_CONTEXT_LENGTH_NT
_ANCILLARY_GENBANK_FILES = frozenset(
    {
        "1600bp-region.gb",
        "pes-retron-26-a1-a2.gb",
        "retron-179-a1-a2.gb",
        "retron-eco1-rt.gb",
    }
)


class VariantGenBankCatalogError(ValueError):
    """Raised when variant GenBank source authority cannot be cataloged."""


@dataclass(frozen=True, slots=True)
class ExtractedSequenceAuthority:
    sequence_id: str
    label: str
    span_0: tuple[int, int]
    span_1: tuple[int, int]
    strand: int | None
    length_nt: int
    sequence_sha256: str
    authority_kind: str
    mutation_labels: tuple[str, ...] = ()
    fusion_part_labels: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, object]:
        payload = asdict(self)
        payload["span_0"] = list(self.span_0)
        payload["span_1"] = list(self.span_1)
        payload["mutation_labels"] = list(self.mutation_labels)
        payload["fusion_part_labels"] = list(self.fusion_part_labels)
        return payload


@dataclass(frozen=True, slots=True)
class VariantGenBankCatalogRecord:
    variant_id: str
    retron_number: int
    plasmid_name: str
    source_file: str
    source_kind: str
    source_path: str
    source_sha256: str
    record_id: str
    record_name: str | None
    sequence_length_bp: int
    topology: str | None
    antibiotic: str
    benchling_url: str
    reader_design_id: str
    variant_class: str
    comment: str
    lnrna: ExtractedSequenceAuthority
    rt_cds: ExtractedSequenceAuthority
    construct_candidate_id: str
    construct_spans_0: dict[str, tuple[int, int]]
    construct_projection_status: str
    qc_flags: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, object]:
        payload = asdict(self)
        payload["lnrna"] = self.lnrna.to_dict()
        payload["rt_cds"] = self.rt_cds.to_dict()
        payload["construct_spans_0"] = {key: list(value) for key, value in self.construct_spans_0.items()}
        payload["qc_flags"] = list(self.qc_flags)
        return payload


@dataclass(frozen=True, slots=True)
class VariantGenBankCatalog:
    catalog_id: str
    study_id: str
    source_metadata_path: str
    genbank_dir: str
    records: tuple[VariantGenBankCatalogRecord, ...]
    errors: tuple[str, ...] = ()
    missing_metadata_source_files: tuple[str, ...] = ()
    missing_genbank_source_files: tuple[str, ...] = ()

    @property
    def ok(self) -> bool:
        return not self.errors

    @property
    def variant_count(self) -> int:
        return len(self.records)

    @property
    def records_by_variant_id(self) -> dict[str, VariantGenBankCatalogRecord]:
        return {record.variant_id: record for record in self.records}

    def record(self, variant_id: str) -> VariantGenBankCatalogRecord:
        try:
            return self.records_by_variant_id[variant_id]
        except KeyError as exc:
            raise KeyError(f"variant {variant_id!r} is absent from the GenBank catalog") from exc

    def to_dict(self) -> dict[str, object]:
        return {
            "catalog_id": self.catalog_id,
            "study_id": self.study_id,
            "source_metadata_path": self.source_metadata_path,
            "genbank_dir": self.genbank_dir,
            "variant_count": self.variant_count,
            "ok": self.ok,
            "missing_metadata_source_files": list(self.missing_metadata_source_files),
            "missing_genbank_source_files": list(self.missing_genbank_source_files),
            "errors": list(self.errors),
            "records": {record.variant_id: record.to_dict() for record in self.records},
        }


def build_variant_genbank_catalog(
    *,
    repo_root: Path | None = None,
    metadata_path: Path | None = None,
) -> VariantGenBankCatalog:
    """Build the RT-lnRNA variant catalog from study-owned GenBank files."""
    root = _resolve_repo_root(repo_root)
    resolved_metadata = root / (metadata_path or _DEFAULT_METADATA_PATH)
    metadata = _load_metadata(resolved_metadata)
    genbank_dir = root / _string(metadata.get("genbank_dir"), label="genbank_dir")
    records_payload = _list(metadata.get("records"), label="records")
    source_files = tuple(_string(item.get("source_file"), label="records[].source_file") for item in records_payload)
    missing_source_files = tuple(
        sorted(source_file for source_file in source_files if not (genbank_dir / source_file).exists())
    )
    genbank_files = tuple(sorted(path.name for path in genbank_dir.glob("*.gb")))
    known_files = set(source_files) | set(_ANCILLARY_GENBANK_FILES)
    missing_metadata_files = tuple(sorted(name for name in genbank_files if name not in known_files))

    errors: list[str] = []
    if missing_source_files:
        errors.append(f"metadata references missing GenBank file(s): {', '.join(missing_source_files)}")
    if missing_metadata_files:
        errors.append(f"GenBank file(s) lack variant metadata: {', '.join(missing_metadata_files)}")
    duplicate_ids = _duplicates(
        _string(item.get("variant_id"), label="records[].variant_id") for item in records_payload
    )
    if duplicate_ids:
        errors.append(f"duplicate variant_id value(s): {', '.join(duplicate_ids)}")

    parser = BiopythonGenBankParser()
    wt_rt_reference = _load_wt_rt_reference(
        parser=parser,
        genbank_dir=genbank_dir,
        records_payload=records_payload,
        errors=errors,
    )
    records: list[VariantGenBankCatalogRecord] = []
    for item in records_payload:
        if _string(item.get("source_file"), label="records[].source_file") in missing_source_files:
            continue
        try:
            records.append(
                _build_record(
                    item=item,
                    parser=parser,
                    repo_root=root,
                    genbank_dir=genbank_dir,
                    wt_rt_reference=wt_rt_reference,
                )
            )
        except VariantGenBankCatalogError as exc:
            errors.append(str(exc))

    return VariantGenBankCatalog(
        catalog_id="rt_lnrna_sponging_construct_triage_retron_variant_genbank_catalog_v1",
        study_id="rt_lnrna_sponging_construct_triage",
        source_metadata_path=str(resolved_metadata.relative_to(root)),
        genbank_dir=str(genbank_dir.relative_to(root)),
        records=tuple(sorted(records, key=lambda record: record.retron_number)),
        errors=tuple(errors),
        missing_metadata_source_files=missing_metadata_files,
        missing_genbank_source_files=missing_source_files,
    )


def write_variant_genbank_catalog(
    *,
    repo_root: Path | None = None,
    metadata_path: Path | None = None,
    output_path: Path | None = None,
) -> VariantGenBankCatalog:
    root = _resolve_repo_root(repo_root)
    catalog = build_variant_genbank_catalog(repo_root=root, metadata_path=metadata_path)
    if not catalog.ok:
        raise VariantGenBankCatalogError("; ".join(catalog.errors))
    destination = root / (output_path or _DEFAULT_CATALOG_PATH)
    destination.write_text(yaml.safe_dump(catalog.to_dict(), sort_keys=False), encoding="utf-8")
    return catalog


def _build_record(
    *,
    item: Mapping[str, object],
    parser: BiopythonGenBankParser,
    repo_root: Path,
    genbank_dir: Path,
    wt_rt_reference: str,
) -> VariantGenBankCatalogRecord:
    variant_id = _string(item.get("variant_id"), label="variant_id")
    retron_number = int(item.get("retron_number") or 0)
    source_file = _string(item.get("source_file"), label=f"{variant_id}.source_file")
    source_kind = _source_kind(item)
    source_path = genbank_dir / source_file
    parsed = parser.parse_file(source_path)
    if len(parsed) != 1:
        raise VariantGenBankCatalogError(
            f"{variant_id}: expected one GenBank record in {source_file}, found {len(parsed)}"
        )
    record = parsed[0]
    if source_kind == "whole_plasmid" and record.topology != "circular":
        raise VariantGenBankCatalogError(
            f"{variant_id}: whole-plasmid source must be circular, got {record.topology!r}"
        )
    if source_kind == "lnrna_only" and record.topology != "linear":
        raise VariantGenBankCatalogError(f"{variant_id}: lnRNA-only source must be linear, got {record.topology!r}")

    lnrna = _extract_lnrna_authority(variant_id=variant_id, item=item, record=record, source_file=source_file)
    rt_cds = _extract_rt_authority(
        variant_id=variant_id,
        item=item,
        parser=parser,
        genbank_dir=genbank_dir,
        record=record,
        source_file=source_file,
        wt_rt_reference=wt_rt_reference,
    )
    construct_spans, qc_flags = _construct_spans(lnrna_length=lnrna.length_nt, rt_length=rt_cds.length_nt)
    construct_status = (
        "representable"
        if not any(flag.endswith("_flank_exhausted") for flag in qc_flags)
        else "blocked_full_slot_set_outside_1600bp_context"
    )
    return VariantGenBankCatalogRecord(
        variant_id=variant_id,
        retron_number=retron_number,
        plasmid_name=_string(item.get("plasmid_name"), label=f"{variant_id}.plasmid_name"),
        source_file=source_file,
        source_kind=source_kind,
        source_path=str(source_path.relative_to(repo_root)),
        source_sha256=record.source_sha256,
        record_id=record.record_id or "",
        record_name=record.record_name,
        sequence_length_bp=len(record.sequence),
        topology=record.topology,
        antibiotic=_string(item.get("antibiotic"), label=f"{variant_id}.antibiotic"),
        benchling_url=_string(item.get("benchling_url"), label=f"{variant_id}.benchling_url"),
        reader_design_id=str(item.get("reader_design_id") or f"pES-retron-{retron_number}; pBbS2c-rfp"),
        variant_class=_string(item.get("variant_class"), label=f"{variant_id}.variant_class"),
        comment=_string(item.get("comment"), label=f"{variant_id}.comment"),
        lnrna=lnrna,
        rt_cds=rt_cds,
        construct_candidate_id=_construct_candidate_id(
            variant_id=variant_id,
            rt_authority_kind=rt_cds.authority_kind,
            item=item,
        ),
        construct_spans_0=construct_spans,
        construct_projection_status=construct_status,
        qc_flags=tuple(qc_flags),
    )


def _extract_lnrna_authority(
    *,
    variant_id: str,
    item: Mapping[str, object],
    record: Any,
    source_file: str,
) -> ExtractedSequenceAuthority:
    extraction = str(item.get("lnrna_extraction") or "a1_20_to_a2_20").strip()
    if extraction == "record":
        _one_feature(record, "a1", variant_id=variant_id)
        _one_feature(record, "a2", variant_id=variant_id)
        start = 0
        end = len(record.sequence)
        label = "record"
    elif extraction == "legacy_msr_to_a2":
        start_feature = _one_feature(record, "msr", variant_id=variant_id)
        end_feature = _one_feature(record, "a2", variant_id=variant_id)
        label = "msr..a2"
        start, end = _span_from_features(start_feature, end_feature, variant_id=variant_id, label=label)
    elif extraction == "a1_20_to_a2_20":
        start_feature = _one_feature(record, "a1(20)", variant_id=variant_id)
        end_feature = _one_feature(record, "a2(20)", variant_id=variant_id)
        label = "a1(20)..a2(20)"
        start, end = _span_from_features(start_feature, end_feature, variant_id=variant_id, label=label)
    else:
        raise VariantGenBankCatalogError(f"{variant_id}: unsupported lnrna_extraction {extraction!r}")
    sequence = record.sequence[start:end]
    return ExtractedSequenceAuthority(
        sequence_id=f"genbank:{source_file}#{label}",
        label=label,
        span_0=(start, end),
        span_1=(start + 1, end),
        strand=1,
        length_nt=end - start,
        sequence_sha256=_sha256_text(sequence),
        authority_kind=extraction,
    )


def _extract_rt_authority(
    *,
    variant_id: str,
    item: Mapping[str, object],
    parser: BiopythonGenBankParser,
    genbank_dir: Path,
    record: Any,
    source_file: str,
    wt_rt_reference: str,
) -> ExtractedSequenceAuthority:
    rt_mode = _string(item.get("rt_mode"), label=f"{variant_id}.rt_mode")
    if rt_mode in {"wt_eco1_rt", "rt_point_mutation"}:
        rt_reference_source_file = str(item.get("rt_reference_source_file") or "").strip()
        if rt_mode == "wt_eco1_rt" and rt_reference_source_file:
            rt_record = _single_record(
                parser=parser,
                source_path=genbank_dir / rt_reference_source_file,
                variant_id=variant_id,
                source_file=rt_reference_source_file,
            )
            rt_feature = _one_feature(rt_record, "ECD_00831", variant_id=variant_id)
            start, end = _feature_span(rt_feature, variant_id=variant_id, label="ECD_00831")
            sequence = rt_record.sequence[start:end]
            if sequence != wt_rt_reference:
                raise VariantGenBankCatalogError(
                    f"{variant_id}: RT reference {rt_reference_source_file} differs from retron26 WT RT"
                )
            return ExtractedSequenceAuthority(
                sequence_id=f"genbank:{rt_reference_source_file}#ECD_00831",
                label="ECD_00831",
                span_0=(start, end),
                span_1=(start + 1, end),
                strand=rt_feature.strand,
                length_nt=len(sequence),
                sequence_sha256=_sha256_text(sequence),
                authority_kind=rt_mode,
            )
        rt_feature = _one_feature(record, "ECD_00831", variant_id=variant_id)
        start, end = _feature_span(rt_feature, variant_id=variant_id, label="ECD_00831")
        sequence = record.sequence[start:end]
        if len(sequence) != len(wt_rt_reference):
            raise VariantGenBankCatalogError(f"{variant_id}: RT CDS length {len(sequence)} does not match WT Eco1 RT")
        mutation_labels = tuple(
            _string(label, label=f"{variant_id}.expected_mutation_labels[]")
            for label in item.get("expected_mutation_labels", ())
        )
        if rt_mode == "wt_eco1_rt" and sequence != wt_rt_reference:
            raise VariantGenBankCatalogError(f"{variant_id}: rt_mode=wt_eco1_rt but ECD_00831 differs from retron26")
        if rt_mode == "rt_point_mutation":
            if sequence == wt_rt_reference:
                raise VariantGenBankCatalogError(f"{variant_id}: rt_point_mutation sequence unexpectedly matches WT RT")
            _require_mutation_labels(record, labels=mutation_labels, variant_id=variant_id)
        return ExtractedSequenceAuthority(
            sequence_id=f"genbank:{source_file}#ECD_00831",
            label="ECD_00831",
            span_0=(start, end),
            span_1=(start + 1, end),
            strand=rt_feature.strand,
            length_nt=len(sequence),
            sequence_sha256=_sha256_text(sequence),
            authority_kind=rt_mode,
            mutation_labels=mutation_labels,
        )
    if rt_mode == "rt_translational_fusion":
        labels = tuple(
            _string(label, label=f"{variant_id}.expected_fusion_part_labels[]")
            for label in item.get("expected_fusion_part_labels", ())
        )
        if not labels:
            raise VariantGenBankCatalogError(
                f"{variant_id}: rt_translational_fusion requires expected_fusion_part_labels"
            )
        features = [_one_cds_feature(record, label, variant_id=variant_id) for label in labels]
        start = min(_feature_span(feature, variant_id=variant_id, label=str(feature.label))[0] for feature in features)
        end = max(_feature_span(feature, variant_id=variant_id, label=str(feature.label))[1] for feature in features)
        sequence = record.sequence[start:end]
        if len(sequence) <= len(wt_rt_reference):
            raise VariantGenBankCatalogError(f"{variant_id}: fusion RT slot is not longer than WT RT")
        return ExtractedSequenceAuthority(
            sequence_id=f"genbank:{source_file}#{'+'.join(labels)}",
            label="+".join(labels),
            span_0=(start, end),
            span_1=(start + 1, end),
            strand=1,
            length_nt=len(sequence),
            sequence_sha256=_sha256_text(sequence),
            authority_kind=rt_mode,
            fusion_part_labels=labels,
        )
    raise VariantGenBankCatalogError(f"{variant_id}: unsupported rt_mode {rt_mode!r}")


def _construct_spans(*, lnrna_length: int, rt_length: int) -> tuple[dict[str, tuple[int, int]], list[str]]:
    base_lnrna_start, base_lnrna_end = _BASE_TEMPLATE_LNRNA_SPAN_0
    base_rt_start, base_rt_end = _BASE_TEMPLATE_RT_SPAN_0
    base_lnrna_length = base_lnrna_end - base_lnrna_start
    base_rt_length = base_rt_end - base_rt_start
    base_prefix_length = base_lnrna_start - _TARGET_CONTEXT_START_0
    base_interstitial_length = base_rt_start - base_lnrna_end
    base_suffix_length = _BASE_CONTEXT_END_0 - base_rt_end
    length_delta = (lnrna_length - base_lnrna_length) + (rt_length - base_rt_length)
    left_adjust = length_delta // 2
    right_adjust = length_delta - left_adjust
    prefix_length = base_prefix_length - left_adjust
    suffix_length = base_suffix_length - right_adjust
    lnrna_start = prefix_length
    lnrna_end = lnrna_start + lnrna_length
    rt_start = lnrna_end + base_interstitial_length
    rt_end = rt_start + rt_length
    spans = {
        "lnrna": (lnrna_start, lnrna_end),
        "rt_cds": (rt_start, rt_end),
    }
    qc_flags: list[str] = []
    if length_delta > 0:
        qc_flags.append("context_flanks_truncated_to_1600bp")
    elif length_delta < 0:
        qc_flags.append("context_flanks_extended_to_1600bp")
    if abs(length_delta) % 2:
        qc_flags.append("context_flank_adjustment_1bp_asymmetry")
    if prefix_length < 0:
        qc_flags.append("prefix_flank_exhausted")
    if suffix_length < 0:
        qc_flags.append("suffix_flank_exhausted")
    if rt_end + suffix_length != _TARGET_CONTEXT_LENGTH_NT:
        qc_flags.append("context_span_geometry_mismatch")
    return spans, qc_flags


def _load_wt_rt_reference(
    *,
    parser: BiopythonGenBankParser,
    genbank_dir: Path,
    records_payload: Sequence[Mapping[str, object]],
    errors: list[str],
) -> str:
    retron26 = next((item for item in records_payload if item.get("variant_id") == "retron26"), None)
    if retron26 is None:
        errors.append("metadata must include retron26 as WT RT reference")
        return ""
    source_file = _string(retron26.get("source_file"), label="retron26.source_file")
    source_path = genbank_dir / source_file
    if not source_path.exists():
        errors.append(f"retron26 WT RT reference file is absent: {source_file}")
        return ""
    records = parser.parse_file(source_path)
    if len(records) != 1:
        errors.append(f"retron26 WT RT reference expected one record, found {len(records)}")
        return ""
    rt_feature = _one_feature(records[0], "ECD_00831", variant_id="retron26")
    start, end = _feature_span(rt_feature, variant_id="retron26", label="ECD_00831")
    return records[0].sequence[start:end]


def _load_metadata(path: Path) -> Mapping[str, object]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise VariantGenBankCatalogError(f"{path}: metadata must be a mapping")
    return payload


def _one_feature(record: Any, label: str, *, variant_id: str) -> Any:
    matches = [feature for feature in record.features if feature.label == label]
    if len(matches) != 1:
        raise VariantGenBankCatalogError(
            f"{variant_id}: expected exactly one feature label {label!r}, found {len(matches)}"
        )
    return matches[0]


def _one_cds_feature(record: Any, label: str, *, variant_id: str) -> Any:
    matches = [feature for feature in record.features if feature.label == label and feature.feature_type == "CDS"]
    if len(matches) != 1:
        raise VariantGenBankCatalogError(
            f"{variant_id}: expected exactly one CDS feature label {label!r}, found {len(matches)}"
        )
    return matches[0]


def _feature_span(feature: Any, *, variant_id: str, label: str) -> tuple[int, int]:
    if feature.start_0 is None or feature.end_0 is None:
        raise VariantGenBankCatalogError(f"{variant_id}: feature {label!r} does not have an exact span")
    if feature.end_0 <= feature.start_0:
        raise VariantGenBankCatalogError(f"{variant_id}: feature {label!r} has invalid span")
    return int(feature.start_0), int(feature.end_0)


def _span_from_features(start_feature: Any, end_feature: Any, *, variant_id: str, label: str) -> tuple[int, int]:
    start, _ = _feature_span(start_feature, variant_id=variant_id, label=label)
    _, end = _feature_span(end_feature, variant_id=variant_id, label=label)
    if end <= start:
        raise VariantGenBankCatalogError(f"{variant_id}: {label} resolves to an invalid span")
    return start, end


def _require_mutation_labels(record: Any, *, labels: tuple[str, ...], variant_id: str) -> None:
    if not labels:
        raise VariantGenBankCatalogError(f"{variant_id}: rt_point_mutation requires expected_mutation_labels")
    present = {str(feature.label) for feature in record.features}
    missing = [label for label in labels if label not in present]
    if missing:
        raise VariantGenBankCatalogError(f"{variant_id}: missing expected RT mutation label(s): {missing}")


def _single_record(
    *,
    parser: BiopythonGenBankParser,
    source_path: Path,
    variant_id: str,
    source_file: str,
) -> Any:
    records = parser.parse_file(source_path)
    if len(records) != 1:
        raise VariantGenBankCatalogError(
            f"{variant_id}: expected one GenBank record in {source_file}, found {len(records)}"
        )
    return records[0]


def _construct_candidate_id(*, variant_id: str, rt_authority_kind: str, item: Mapping[str, object]) -> str:
    explicit = str(item.get("construct_candidate_id") or "").strip()
    if explicit:
        return explicit
    if rt_authority_kind == "wt_eco1_rt":
        return f"rt_lnrna_pair__eco1_wt_rt__{variant_id}_lnrna__tetO"
    if rt_authority_kind == "rt_point_mutation":
        return f"rt_lnrna_pair__{variant_id}_rt_variant__{variant_id}_lnrna__tetO"
    return f"rt_lnrna_pair__{variant_id}_rt_fusion__{variant_id}_lnrna__tetO"


def _source_kind(item: Mapping[str, object]) -> str:
    kind = str(item.get("source_kind") or "whole_plasmid").strip()
    if kind not in {"whole_plasmid", "lnrna_only"}:
        variant_id = str(item.get("variant_id") or "<unknown>")
        raise VariantGenBankCatalogError(f"{variant_id}: unsupported source_kind {kind!r}")
    return kind


def _duplicates(values: Sequence[str]) -> tuple[str, ...]:
    seen: set[str] = set()
    duplicates: set[str] = set()
    for value in values:
        if value in seen:
            duplicates.add(value)
        seen.add(value)
    return tuple(sorted(duplicates))


def _sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("ascii")).hexdigest()


def _list(value: object, *, label: str) -> list[Mapping[str, object]]:
    if not isinstance(value, list):
        raise VariantGenBankCatalogError(f"{label} must be a list")
    out: list[Mapping[str, object]] = []
    for index, item in enumerate(value, start=1):
        if not isinstance(item, Mapping):
            raise VariantGenBankCatalogError(f"{label}[{index}] must be a mapping")
        out.append(item)
    return out


def _string(value: object, *, label: str) -> str:
    text = str(value or "").strip()
    if not text:
        raise VariantGenBankCatalogError(f"{label} must be non-empty")
    return text


def _resolve_repo_root(repo_root: Path | None) -> Path:
    if repo_root is not None:
        return Path(repo_root).resolve()
    current = Path(__file__).resolve()
    for parent in current.parents:
        if (parent / "pyproject.toml").exists():
            return parent
    raise RuntimeError("repo root not found")


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Validate and write RT-lnRNA variant GenBank catalog.")
    parser.add_argument("--repo-root", type=Path, default=None)
    parser.add_argument("--metadata", type=Path, default=None)
    parser.add_argument("--write-catalog", action="store_true")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)
    if args.write_catalog:
        catalog = write_variant_genbank_catalog(repo_root=args.repo_root, metadata_path=args.metadata)
    else:
        catalog = build_variant_genbank_catalog(repo_root=args.repo_root, metadata_path=args.metadata)
    if args.json:
        print(json.dumps(catalog.to_dict(), indent=2, sort_keys=True))
    else:
        print(f"variant GenBank catalog: ok={catalog.ok} variants={catalog.variant_count} errors={len(catalog.errors)}")
        for error in catalog.errors:
            print(f"- {error}")
    return 0 if catalog.ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
