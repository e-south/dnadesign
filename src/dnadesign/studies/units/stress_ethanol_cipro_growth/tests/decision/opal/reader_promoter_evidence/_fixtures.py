"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/studies/units/stress_ethanol_cipro_growth/tests/decision/opal/reader_promoter_evidence/_fixtures.py

Fixtures for canonical Reader diagnostic display projections.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import pandas as pd

from dnadesign.studies.core.reader_records import (
    ReaderArtifactFile,
    ReaderRecordSet,
    ReaderResolvedRecord,
)
from dnadesign.studies.units.stress_ethanol_cipro_growth.decision.opal.reader_promoter_evidence.contracts import (
    VerifiedReaderPromoterEvidenceSource,
)
from dnadesign.studies.units.stress_ethanol_cipro_growth.promoter_candidate_bindings import (
    BindingSourceArtifact,
    materialize_promoter_candidate_bindings,
    preview_promoter_candidate_bindings,
)
from dnadesign.studies.units.stress_ethanol_cipro_growth.response_window_observations.reader_records import (
    ReaderResponseDisplay,
    ReaderResponseRecords,
)


def verified_source(tmp_path: Path) -> VerifiedReaderPromoterEvidenceSource:
    """Build one already verified source object for publication tests."""

    experiment_root = tmp_path / "reader" / "experiments" / "2026" / "aggregate"
    outputs_root = experiment_root / "outputs"
    config_path = experiment_root / "config.yaml"
    catalog_path = outputs_root / "manifests" / "records.json"
    projection_path = tmp_path / "projection.yaml"
    diagnostic_path = outputs_root / "plots" / "four_state_event_window_diagnostic.png"
    for path, content in (
        (config_path, b"reader/v8\n"),
        (catalog_path, b"{}\n"),
        (projection_path, b"projection\n"),
        (diagnostic_path, b"\x89PNG\r\n\x1a\nreader diagnostic"),
    ):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content)
    designs = _dataframe_record(
        record_id="four_state_event_window/designs",
        contract_id="plate_reader.four_state_event_window.designs.v4",
        reader_path="artifacts/four_state_event_window/designs.parquet",
        digest_character="1",
    )
    traces = _dataframe_record(
        record_id="four_state_event_window/traces",
        contract_id="plate_reader.four_state_event_window.traces.v3",
        reader_path="artifacts/four_state_event_window/traces.parquet",
        digest_character="2",
    )
    file_digest = sha256(diagnostic_path)
    diagnostic = ReaderResolvedRecord._verified(
        record_id="plot:four_state_event_window_diagnostic",
        kind="file_bundle",
        schema_version=6,
        revision=3,
        revision_digest="sha256:" + "d" * 64,
        contract_id=None,
        producer={
            "kind": "plot",
            "id": "four_state_event_window_diagnostic",
            "plugin": "plot/four_state_event_window_diagnostic",
        },
        producer_config_digest="sha256:" + "c" * 64,
        inputs=(
            {
                "label": "designs",
                "kind": "record",
                "record": designs.record_id,
                "discovery_policy": "record",
                "record_revision_digest": designs.revision_digest,
            },
            {
                "label": "traces",
                "kind": "record",
                "record": traces.record_id,
                "discovery_policy": "record",
                "record_revision_digest": traces.revision_digest,
            },
        ),
        path=None,
        reader_path=None,
        size_bytes=None,
        content_digest=None,
        content=None,
        files=(
            ReaderArtifactFile(
                reader_path="plots/four_state_event_window_diagnostic.png",
                path=diagnostic_path,
                size_bytes=diagnostic_path.stat().st_size,
                content_digest=file_digest,
                content=diagnostic_path.read_bytes(),
            ),
        ),
    )
    record_set = ReaderRecordSet(
        reader_root=tmp_path / "reader",
        experiment_root=experiment_root,
        config_path=config_path,
        outputs_root=outputs_root,
        catalog_path=catalog_path,
        catalog_sha256=hashlib.sha256(catalog_path.read_bytes()).hexdigest(),
        catalog_schema_version=4,
        provenance_epoch_id="123e4567-e89b-42d3-a456-426614174000",
        experiment_id="20260717_stress_response_window_aggregate",
        protocol_id="plate_reader/four_state_event_window",
        experiment_evidence={},
        records={"designs": designs, "traces": traces},
    )
    records = ReaderResponseRecords(
        source=record_set,
        projection_path=projection_path,
        projection_sha256=hashlib.sha256(projection_path.read_bytes()).hexdigest(),
        projection={"primary_reduction_id": "event_logmean_4_8h_post"},
        designs=pd.DataFrame(),
        descriptive_resampling_draws=pd.DataFrame(),
        wells=pd.DataFrame(),
        traces=pd.DataFrame(),
        events=pd.DataFrame(),
    )
    display = ReaderResponseDisplay(
        source_experiment_id="20260619_sfxi_sensor-panel-m9-glu-1-10",
        design_id="pDual-10-ES1p",
        record=diagnostic,
        selected_file=diagnostic.files[0],
    )
    selected_binding = {
        "reader_design_id": "pDual-10-ES1p",
        "candidate_id": "candidate-1",
        "sequence_sha256": "sha256:" + "3" * 64,
        "sequence_authority_dataset_id": "reader-test-authority",
        "sequence_authority_id": "authority:pDual-10-ES1p",
        "sequence_authority_sha256": "sha256:" + "4" * 64,
        "source_class": "densegen",
        "design_family": "ethanol_ciprofloxacin",
        "binding_status": "resolved",
        "binding_method": "exact_alias",
        "densegen_plan": "ethanol_ciprofloxacin",
        "densegen_run_id": "run-1",
        "densegen_sampling_library_hash": "library-1",
    }
    binding_source = {
        "schema_id": "dnadesign.study.promoter_candidate_bindings.v1",
        "schema_version": "1",
        "study_id": "stress_ethanol_cipro_growth",
        "manifest_sha256": "sha256:" + "5" * 64,
        "records_sha256": "sha256:" + "6" * 64,
        "candidate_table_id": "usr_prom_eth_cip_opal_candidates",
        "candidate_selection_sha256": "sha256:" + "7" * 64,
    }
    return VerifiedReaderPromoterEvidenceSource(
        records=records,
        display=display,
        selected_binding=selected_binding,
        binding_source=binding_source,
    )


def write_candidate_bindings(
    bundle: Path,
    specs: list[tuple[str, str, str]] | None = None,
) -> Path:
    specs = specs or [("candidate-1", "pDual-10-ES1p", "densegen_tfbs")]
    sequence = "ACGTACGT" + "CTGACA" + "AAAA" + "TATAAT"
    aliases: list[dict[str, str]] = []
    candidates: list[dict[str, object]] = []
    annotations: list[dict[str, object]] = []
    for candidate_id, design_id, adapter_kind in specs:
        authority = f"authority:{design_id}"
        aliases.append(
            {
                "alias_namespace": "reader.design_id",
                "alias": design_id,
                "display_label": design_id,
                "candidate_id": candidate_id,
                "authority_sequence": sequence,
                "sequence_authority_dataset_id": "reader-test-authority",
                "sequence_authority_id": authority,
                "sequence_authority_sha256": hashlib.sha256(authority.encode()).hexdigest(),
            }
        )
        densegen = adapter_kind == "densegen_tfbs"
        candidates.append(
            {
                "id": candidate_id,
                "sequence": sequence,
                "usr_label__primary": None if densegen else design_id,
                "opal_candidate__source_class": "densegen" if densegen else "construct_derived",
                "opal_candidate__design_family": "ethanol_ciprofloxacin" if densegen else "control",
                "densegen__plan": "ethanol_ciprofloxacin" if densegen else None,
                "densegen__run_id": "reader_sfxi_pdual10_archive_port" if densegen else None,
                "densegen__sampling_library_hash": "archive_library_hash" if densegen else None,
                "densegen__used_tfbs_detail": _densegen_annotations() if densegen else None,
                "densegen__required_regulators": ["baeR"] if densegen else None,
            }
        )
        if not densegen:
            annotations.append(
                {
                    "id": candidate_id,
                    "seq_annot__features": [
                        {
                            "feature_id": f"{candidate_id}-promoter",
                            "feature_type": "promoter",
                            "label": design_id,
                            "start_0": 0,
                            "end_0": 6,
                            "strand": 1,
                        }
                    ],
                    "seq_annot__source_artifact_uri": f"artifacts/genbank/{candidate_id}.gb",
                }
            )
    preview = preview_promoter_candidate_bindings(
        alias_rows=pd.DataFrame(aliases),
        candidate_records=pd.DataFrame(candidates),
        genbank_annotations=pd.DataFrame(annotations),
        candidate_table_id="usr_prom_eth_cip_opal_candidates",
        candidate_selection_sha256="7" * 64,
        source_artifacts=(BindingSourceArtifact("test-authority", "inputs/aliases.parquet", "8" * 64),),
    )
    materialize_promoter_candidate_bindings(preview, out_dir=bundle, allowed_output_root=bundle.parent)
    return bundle


def _dataframe_record(
    *,
    record_id: str,
    contract_id: str,
    reader_path: str,
    digest_character: str,
) -> ReaderResolvedRecord:
    return ReaderResolvedRecord._verified(
        record_id=record_id,
        kind="dataframe_artifact",
        schema_version=6,
        revision=2,
        revision_digest="sha256:" + digest_character * 64,
        contract_id=contract_id,
        producer={},
        producer_config_digest=None,
        inputs=(),
        path=Path("/verified") / reader_path,
        reader_path=reader_path,
        size_bytes=10,
        content_digest="sha256:" + digest_character * 64,
        content=b"0123456789",
        files=(),
    )


def _densegen_annotations() -> list[dict[str, object]]:
    return [
        {
            "part_kind": "tfbs",
            "sequence": "ACGT",
            "regulator": "baeR",
            "offset": 0,
            "offset_raw": 0,
            "length": 4,
            "end": 4,
            "orientation": "fwd",
        },
        {
            "part_kind": "fixed_element",
            "role": "upstream",
            "constraint_name": "sigma70_core",
            "sequence": "CTGACA",
            "offset": 8,
            "offset_raw": 8,
            "length": 6,
            "end": 14,
            "spacer_length": 4,
            "placement_index": 0,
        },
        {
            "part_kind": "fixed_element",
            "role": "downstream",
            "constraint_name": "sigma70_core",
            "sequence": "TATAAT",
            "offset": 18,
            "offset_raw": 18,
            "length": 6,
            "end": 24,
            "spacer_length": 4,
            "placement_index": 0,
        },
    ]


def sha256(path: Path) -> str:
    return "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()


__all__ = ["sha256", "verified_source", "write_candidate_bindings"]
