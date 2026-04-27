"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/usr/src/genbank/importer.py

Manifest-driven GenBank import into USR datasets with annotation overlays and
sequence-view sidecars.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import shutil
from dataclasses import dataclass
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import yaml

from ..contracts import SchemaError, compute_id
from ..dataset import Dataset
from ..overlays import overlay_dir_path, overlay_parts, overlay_path
from ..registry import arrow_type_from_str, ensure_sequence_contract_namespaces, load_registry, registry_entry
from ..sequence_views import SequenceViewRecord, write_sequence_views
from .models import (
    FeatureExtractionSpec,
    GenBankImportManifest,
    GenBankImportRecordSpec,
    ParsedGenBankFeature,
    ParsedGenBankRecord,
)
from .parser import BiopythonGenBankParser

_GENBANK_ARTIFACT_DIR = "_artifacts/genbank"


@dataclass(frozen=True)
class GenBankImportResult:
    dataset: str
    native_records: int
    extracted_records: int
    sequence_views_written: int


def load_genbank_import_manifest(path: Path) -> GenBankImportManifest:
    manifest_path = Path(path)
    payload = yaml.safe_load(manifest_path.read_text(encoding="utf-8")) or {}
    return GenBankImportManifest.model_validate(payload)


def import_genbank_manifest(
    *,
    root: Path,
    manifest_path: Path,
    actor: dict[str, object] | None = None,
) -> GenBankImportResult:
    manifest = load_genbank_import_manifest(manifest_path)
    ensure_sequence_contract_namespaces(root)
    dataset = Dataset(root, manifest.output_dataset)

    parser = BiopythonGenBankParser()
    created_at = _created_at()
    native_payloads = [
        _parse_manifest_record(
            manifest_path=manifest_path,
            record_spec=record_spec,
            parser=parser,
            role_hint_rules=manifest.role_hint_rules,
        )
        for record_spec in manifest.records
    ]

    native_rows: list[dict[str, object]] = []
    native_label_rows: list[dict[str, object]] = []
    native_seq_annot_rows: list[dict[str, object]] = []
    native_view_rows: list[SequenceViewRecord] = []
    native_by_label: dict[str, dict[str, object]] = {}

    existing_source_hashes = _existing_overlay_values(dataset, namespace="seq_annot", column="seq_annot__source_sha256")
    for payload in native_payloads:
        record_spec = payload["record_spec"]
        parsed_record = payload["parsed_record"]
        sequence_id = compute_id("dna", parsed_record.sequence)
        if manifest.on_conflict == "idempotent":
            observed_hash = existing_source_hashes.get(sequence_id)
            if observed_hash is not None and observed_hash != parsed_record.source_sha256:
                raise SchemaError(
                    f"GenBank import for '{record_spec.label}' conflicts with existing seq_annot source hash "
                    f"for sequence '{sequence_id}'."
                )

        artifact_uri = (
            _source_artifact_uri(
                dataset=dataset,
                source_path=payload["source_path"],
                source_sha256=parsed_record.source_sha256,
            )
            if manifest.copy_source_artifacts
            else None
        )
        native_label_rows.append(
            {
                "id": sequence_id,
                "usr_label__primary": record_spec.label,
                "usr_label__aliases": list(record_spec.aliases or []),
            }
        )
        native_seq_annot_rows.append(
            {
                "id": sequence_id,
                "seq_annot__format": "genbank",
                "seq_annot__source_file": payload["source_file_label"],
                "seq_annot__source_sha256": parsed_record.source_sha256,
                "seq_annot__source_artifact_uri": artifact_uri,
                "seq_annot__parser": parser.parser_name,
                "seq_annot__parser_version": None,
                "seq_annot__record_id": parsed_record.record_id,
                "seq_annot__record_name": parsed_record.record_name,
                "seq_annot__description": parsed_record.description,
                "seq_annot__topology": parsed_record.topology,
                "seq_annot__molecule_type": parsed_record.molecule_type,
                "seq_annot__sequence_region_start_0": parsed_record.sequence_region_start_0,
                "seq_annot__sequence_region_end_0": parsed_record.sequence_region_end_0,
                "seq_annot__features": [_feature_to_overlay_value(feature) for feature in parsed_record.features],
            }
        )
        native_view_rows.append(
            SequenceViewRecord(
                sequence_id=sequence_id,
                view_name=record_spec.label,
                aliases=record_spec.aliases,
                product_kind="native_record",
                orientation="unknown",
                analysis_only=False,
                source_dataset_id=dataset.name,
                source_label=record_spec.label,
                created_at=created_at,
                created_by="usr.genbank_import",
            )
        )
        native_rows.append(
            {
                "sequence_id": sequence_id,
                "label": record_spec.label,
            }
        )
        native_by_label[record_spec.label.casefold()] = {
            "record_spec": record_spec,
            "parsed_record": parsed_record,
            "sequence_id": sequence_id,
        }

    extracted_rows: list[dict[str, object]] = []
    extracted_label_rows: list[dict[str, object]] = []
    derived_rows: list[dict[str, object]] = []
    extracted_view_rows: list[SequenceViewRecord] = []
    for extraction in manifest.extract_features:
        extracted = _extract_feature_row(
            extraction=extraction,
            native_by_label=native_by_label,
            dataset=dataset,
            created_at=created_at,
        )
        extracted_rows.append(
            {
                "sequence": extracted["sequence"],
                "bio_type": "dna",
                "alphabet": "dna_4",
                "source": f"genbank-feature:{extraction.source_label}",
            }
        )
        extracted_sequence_id = compute_id("dna", extracted["sequence"])
        extracted_label_rows.append(
            {
                "id": extracted_sequence_id,
                "usr_label__primary": extraction.view_name,
                "usr_label__aliases": list(extraction.aliases or []),
            }
        )
        derived_rows.append(
            {
                "id": extracted_sequence_id,
                "derived__parent_id": extracted["parent_sequence_id"],
                "derived__parent_dataset": dataset.name,
                "derived__operation": "extract_feature",
                "derived__product_kind": extraction.product_kind,
                "derived__target_length": len(extracted["sequence"]),
                "derived__source_interval_start_0": extracted["feature"].start_0,
                "derived__source_interval_end_0": extracted["feature"].end_0,
                "derived__source_intervals_0": [interval.model_dump() for interval in extracted["feature"].intervals_0],
                "derived__orientation": extracted["orientation"],
                "derived__template_id": None,
                "derived__template_dataset": None,
                "derived__focal_rule": None,
                "derived__focal_features": [extracted["feature"].feature_id],
                "derived__focal_confidence": extracted["feature"].confidence,
                "derived__analysis_only": False,
                "derived__added_left_bp": None,
                "derived__added_right_bp": None,
                "derived__added_sequence_source": None,
                "derived__features_retained": None,
                "derived__features_clipped": None,
                "derived__features_lost": None,
                "derived__created_by": "usr.genbank_import",
                "derived__spec_id": extracted["derivation_spec_id"],
            }
        )
        extracted_view_rows.append(
            SequenceViewRecord(
                sequence_id=extracted_sequence_id,
                view_name=extraction.view_name,
                aliases=extraction.aliases,
                product_kind=extraction.product_kind,
                orientation=extracted["orientation"],
                analysis_only=False,
                source_dataset_id=dataset.name,
                source_label=extraction.view_name,
                parent_sequence_id=extracted["parent_sequence_id"],
                parent_dataset_id=dataset.name,
                derivation_id=extracted["derivation_id"],
                derivation_spec_id=extracted["derivation_spec_id"],
                source_interval_start_0=extracted["feature"].start_0,
                source_interval_end_0=extracted["feature"].end_0,
                created_at=created_at,
                created_by="usr.genbank_import",
            )
        )

    if manifest.copy_source_artifacts:
        for payload in native_payloads:
            parsed_record = payload["parsed_record"]
            _copy_source_artifact(dataset, payload["source_path"], parsed_record.source_sha256)

    with dataset.write_session() as session:
        session.init_if_missing(source=f"genbank-import:{manifest_path}")

    native_sequences = [payload["parsed_record"].sequence for payload in native_payloads]
    on_conflict = "error" if manifest.on_conflict == "error" else "ignore"
    dataset.add_sequences(
        native_sequences,
        bio_type="dna",
        alphabet="dna_4",
        source=f"genbank:{manifest_path}",
        on_conflict=on_conflict,
        actor=actor,
    )

    _merge_and_replace_overlay(
        dataset,
        namespace="usr_label",
        rows=native_label_rows,
        on_conflict=manifest.on_conflict,
        actor=actor,
    )
    _merge_and_replace_overlay(
        dataset,
        namespace="seq_annot",
        rows=native_seq_annot_rows,
        on_conflict=manifest.on_conflict,
        actor=actor,
    )
    native_views_written = write_sequence_views(
        dataset,
        native_view_rows,
        conflict_policy="error" if manifest.on_conflict == "error" else "idempotent",
        actor=actor,
    )

    if extracted_rows:
        dataset.add_sequences(
            extracted_rows,
            bio_type="dna",
            alphabet="dna_4",
            source=f"genbank-feature:{manifest_path}",
            on_conflict=on_conflict,
            actor=actor,
        )
        _merge_and_replace_overlay(
            dataset,
            namespace="usr_label",
            rows=extracted_label_rows,
            on_conflict=manifest.on_conflict,
            actor=actor,
        )
        _merge_and_replace_overlay(
            dataset,
            namespace="derived",
            rows=derived_rows,
            on_conflict=manifest.on_conflict,
            actor=actor,
        )
        write_sequence_views(
            dataset,
            extracted_view_rows,
            conflict_policy="error" if manifest.on_conflict == "error" else "idempotent",
            actor=actor,
        )

    dataset.log_event(
        "genbank_import",
        args={
            "manifest": str(Path(manifest_path)),
            "native_records": len(native_payloads),
            "extracted_records": len(extracted_rows),
            "on_conflict": manifest.on_conflict,
        },
        artifacts={
            "sources": [payload["source_file_label"] for payload in native_payloads],
            "copied_artifacts": [
                _source_artifact_uri(
                    dataset=dataset,
                    source_path=payload["source_path"],
                    source_sha256=payload["parsed_record"].source_sha256,
                )
                for payload in native_payloads
                if manifest.copy_source_artifacts
            ],
        },
        actor=actor,
    )
    return GenBankImportResult(
        dataset=dataset.name,
        native_records=len(native_payloads),
        extracted_records=len(extracted_rows),
        sequence_views_written=native_views_written + len(extracted_view_rows),
    )


def _created_at() -> str:
    from ..storage.parquet import now_utc

    return now_utc()


def _parse_manifest_record(
    *,
    manifest_path: Path,
    record_spec: GenBankImportRecordSpec,
    parser: BiopythonGenBankParser,
    role_hint_rules,
) -> dict[str, object]:
    source_path = (Path(manifest_path).parent / record_spec.source_file).resolve()
    parsed_records = parser.parse_file(source_path, role_hint_rules=role_hint_rules)
    if len(parsed_records) != 1:
        raise SchemaError(
            f"GenBank source '{source_path}' produced {len(parsed_records)} records; "
            "use one-record sources for usr.genbank_import."
        )
    parsed_record = parsed_records[0]
    return {
        "record_spec": record_spec,
        "parsed_record": parsed_record,
        "source_path": source_path,
        "source_file_label": str(source_path),
    }


def _source_artifact_uri(*, dataset: Dataset, source_path: Path, source_sha256: str) -> str:
    return (Path(_GENBANK_ARTIFACT_DIR) / f"{source_sha256[:16]}-{source_path.name}").as_posix()


def _copy_source_artifact(dataset: Dataset, source_path: Path, source_sha256: str) -> str:
    artifact_dir = dataset.dir / _GENBANK_ARTIFACT_DIR
    artifact_dir.mkdir(parents=True, exist_ok=True)
    target = artifact_dir / f"{source_sha256[:16]}-{source_path.name}"
    if not target.exists():
        shutil.copy2(source_path, target)
    return _source_artifact_uri(dataset=dataset, source_path=source_path, source_sha256=source_sha256)


def _feature_to_overlay_value(feature: ParsedGenBankFeature) -> dict[str, object]:
    return {
        "feature_id": feature.feature_id,
        "feature_order": feature.feature_order,
        "feature_type": feature.feature_type,
        "label": feature.label,
        "role_hint": feature.role_hint,
        "location_raw": feature.location_raw,
        "location_kind": feature.location_kind,
        "start_0": feature.start_0,
        "end_0": feature.end_0,
        "strand": feature.strand,
        "intervals_0": [interval.model_dump() for interval in feature.intervals_0],
        "is_fuzzy": feature.is_fuzzy,
        "is_compound": feature.is_compound,
        "qualifiers": [qualifier.model_dump() for qualifier in feature.qualifiers],
        "confidence": feature.confidence,
        "source": feature.source,
    }


def _extract_feature_row(
    *,
    extraction: FeatureExtractionSpec,
    native_by_label: dict[str, dict[str, object]],
    dataset: Dataset,
    created_at: str,
) -> dict[str, object]:
    native = native_by_label.get(extraction.source_label.casefold())
    if native is None:
        raise SchemaError(f"Unknown source_label '{extraction.source_label}' for feature extraction.")
    parsed_record: ParsedGenBankRecord = native["parsed_record"]
    matches = _select_features(parsed_record.features, extraction=extraction)
    if len(matches) != 1:
        raise SchemaError(
            f"Feature extraction '{extraction.view_name}' matched {len(matches)} features "
            f"in '{extraction.source_label}'."
        )
    feature = matches[0]
    if feature.start_0 is None or feature.end_0 is None:
        raise SchemaError(f"Feature '{feature.feature_id}' lacks precise bounds for extraction.")
    if feature.is_fuzzy:
        raise SchemaError(f"Feature '{feature.feature_id}' uses fuzzy bounds and cannot be sliced exactly.")
    if len(feature.intervals_0) != 1:
        raise SchemaError(f"Feature '{feature.feature_id}' is compound and cannot be sliced as one exact interval.")
    if feature.end_0 > len(parsed_record.sequence):
        raise SchemaError(f"Feature '{feature.feature_id}' exceeds the parent sequence length.")
    derivation_spec_id = f"extract:{extraction.source_label}:{feature.feature_id}:{extraction.product_kind}"
    parent_sequence_id = str(native["sequence_id"])
    sequence = parsed_record.sequence[feature.start_0 : feature.end_0]
    orientation = "reverse_complement" if feature.strand == -1 else "forward"
    if orientation == "reverse_complement":
        sequence = _reverse_complement(sequence)
    return {
        "sequence": sequence,
        "feature": feature,
        "orientation": orientation,
        "parent_sequence_id": parent_sequence_id,
        "derivation_id": f"drv_{compute_id('dna', sequence)[-12:]}",
        "derivation_spec_id": derivation_spec_id,
        "created_at": created_at,
        "dataset": dataset.name,
    }


def _select_features(
    features: list[ParsedGenBankFeature],
    *,
    extraction: FeatureExtractionSpec,
) -> list[ParsedGenBankFeature]:
    if extraction.selector.kind == "label":
        needle = str(extraction.selector.label).casefold()
        return [feature for feature in features if (feature.label or "").casefold() == needle]
    if extraction.selector.kind == "feature_id":
        needle = str(extraction.selector.feature_id)
        return [feature for feature in features if feature.feature_id == needle]
    raise SchemaError(f"Unsupported feature selector kind '{extraction.selector.kind}'.")


def _reverse_complement(sequence: str) -> str:
    translation = str.maketrans("ACGTNacgtn", "TGCANtgcan")
    return sequence.translate(translation)[::-1].upper()


def _namespace_schema(dataset: Dataset, *, namespace: str) -> pa.Schema:
    entry = registry_entry(load_registry(dataset.root, required=True), namespace)
    fields = [pa.field("id", pa.string())]
    for column in entry.columns:
        fields.append(pa.field(column.name, arrow_type_from_str(column.type)))
    return pa.schema(fields)


def _load_overlay_rows(dataset: Dataset, *, namespace: str, schema: pa.Schema) -> list[dict[str, object]]:
    file_path = overlay_path(dataset.dir, namespace)
    dir_path = overlay_dir_path(dataset.dir, namespace)
    if not file_path.exists() and not dir_path.exists():
        return []
    source_path = dir_path if dir_path.exists() else file_path
    parts = overlay_parts(source_path)
    if not parts:
        return []
    tables = [pq.read_table(part).select(schema.names).cast(schema) for part in parts]
    table = pa.concat_tables(tables)
    return [dict(row) for row in table.to_pylist()]


def _merge_overlay_rows(
    *,
    existing_rows: list[dict[str, object]],
    incoming_rows: list[dict[str, object]],
    on_conflict: str,
) -> list[dict[str, object]]:
    by_id = {str(row["id"]): row for row in existing_rows}
    for row in incoming_rows:
        row_id = str(row["id"])
        existing = by_id.get(row_id)
        if existing is None:
            by_id[row_id] = row
            continue
        if on_conflict == "error":
            raise SchemaError(f"Overlay row for id '{row_id}' already exists.")
        if existing != row:
            raise SchemaError(f"Overlay row for id '{row_id}' differs under idempotent import.")
    return [by_id[key] for key in sorted(by_id)]


def _table_from_rows(schema: pa.Schema, rows: list[dict[str, object]]) -> pa.Table:
    arrays = [pa.array([row.get(field.name) for row in rows], type=field.type) for field in schema]
    if not rows:
        arrays = [pa.array([], type=field.type) for field in schema]
    return pa.Table.from_arrays(arrays, schema=schema)


def _merge_and_replace_overlay(
    dataset: Dataset,
    *,
    namespace: str,
    rows: list[dict[str, object]],
    on_conflict: str,
    actor: dict[str, object] | None,
) -> None:
    if not rows:
        return
    schema = _namespace_schema(dataset, namespace=namespace)
    merged_rows = _merge_overlay_rows(
        existing_rows=_load_overlay_rows(dataset, namespace=namespace, schema=schema),
        incoming_rows=rows,
        on_conflict=on_conflict,
    )
    table = _table_from_rows(schema, merged_rows)
    dataset.remove_overlay(namespace, mode="delete")
    dataset.write_overlay_part(namespace, table, actor=actor)
    with dataset.maintenance("genbank_import_overlay", actor=actor):
        dataset.compact_overlay(namespace)


def _existing_overlay_values(dataset: Dataset, *, namespace: str, column: str) -> dict[str, str]:
    schema = _namespace_schema(dataset, namespace=namespace)
    rows = _load_overlay_rows(dataset, namespace=namespace, schema=schema)
    return {
        str(row["id"]): str(row[column]) for row in rows if row.get("id") is not None and row.get(column) is not None
    }


__all__ = ["GenBankImportResult", "import_genbank_manifest", "load_genbank_import_manifest"]
