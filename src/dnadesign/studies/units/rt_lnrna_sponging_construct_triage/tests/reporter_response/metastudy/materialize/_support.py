"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/studies/units/rt_lnrna_sponging_construct_triage/tests/reporter_response/metastudy/materialize/_support.py

Shared source-closed materialization test construction.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import hashlib
import json
import statistics
import subprocess
from pathlib import Path

import pandas as pd

from dnadesign.studies.units.rt_lnrna_sponging_construct_triage.reader_evidence import (
    BiologicalReplicateIdentityScope,
    ReaderDataframeRecordRef,
    ReaderEvidenceBinding,
    ReaderEvidenceBindingSet,
)
from dnadesign.studies.units.rt_lnrna_sponging_construct_triage.reporter_response import (
    ReporterResponseObservationPolicy,
    TemporalSelectedRow,
    UncertaintyPolicy,
)
from dnadesign.studies.units.rt_lnrna_sponging_construct_triage.reporter_response.metastudy import (
    DEFAULT_PROTOCOL,
)
from dnadesign.studies.units.rt_lnrna_sponging_construct_triage.reporter_response.metastudy.condition_ontology import (
    DEFAULT_CONDITION_ONTOLOGY,
    ReporterResponseConditionOntology,
)

_REVISION_DIGEST = "sha256:" + "a" * 64
_SUBJECT_ID = DEFAULT_PROTOCOL.anchor_subject_order[1]
_READER_REDUCTION_SOURCE_DIGEST = "sha256:af3e7603928d3fd6f2b4a2fbb3e33d0309986b99473da834e1a8b5a9e6c36ada"
_READER_CONTRACT_SOURCE_DIGEST = "sha256:8c5cc9bf8dfa68eb2102c002c39eb7f4e7119a95712bff1cdc8c873d60d797b0"


def _reader_reduce_trace_rows(
    rows: tuple[TemporalSelectedRow, ...],
    *,
    temporal_policy,
) -> float:
    phd_root = next(parent for parent in Path(__file__).resolve().parents if (parent / "reader").is_dir())
    reader_root = phd_root / "reader"
    sources = {
        reader_root / "src/reader_workbench/domains/time_series/reduction.py": _READER_REDUCTION_SOURCE_DIGEST,
        reader_root / "src/reader_workbench/domains/time_series/contracts.py": _READER_CONTRACT_SOURCE_DIGEST,
    }
    for path, expected in sources.items():
        observed = "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()
        assert observed == expected, f"Reader conformance source changed: {path}"
    traces = []
    for observation_identity in sorted({row.observation_identity for row in rows}):
        trace = sorted(
            (row for row in rows if row.observation_identity == observation_identity),
            key=lambda row: row.time_h,
        )
        traces.append(
            {
                "trace_id": observation_identity,
                "times": [row.time_h for row in trace],
                "values": [row.value for row in trace],
                "policy_clipped": [row.value_policy_clipped for row in trace],
                "instrument_overflow": [row.value_instrument_overflow for row in trace],
                "bound_kinds": [row.value_bound_kind for row in trace],
            }
        )
    payload = {"spec": temporal_policy.to_reader_mapping(), "traces": traces}
    script = """
import json, sys
import numpy as np
from reader_workbench.domains.time_series import TemporalReductionSpec, reduce_temporal_trace
payload = json.load(sys.stdin)
spec = TemporalReductionSpec.from_mapping(payload["spec"])
assert spec.to_mapping() == payload["spec"]
outputs = []
for trace in payload["traces"]:
    result = reduce_temporal_trace(
        np.asarray(trace["times"], dtype=float),
        np.asarray(trace["values"], dtype=float),
        spec=spec,
        trace_id=trace["trace_id"],
        policy_clipped=np.asarray(trace["policy_clipped"], dtype=bool),
        instrument_overflow=np.asarray(trace["instrument_overflow"], dtype=bool),
        bound_kinds=np.asarray(trace["bound_kinds"], dtype=str),
    )
    outputs.append(result.value)
json.dump(outputs, sys.stdout)
"""
    completed = subprocess.run(
        [str(reader_root / ".venv/bin/python"), "-c", script],
        input=json.dumps(payload, sort_keys=True),
        text=True,
        capture_output=True,
        check=False,
    )
    if completed.returncode != 0:
        raise ValueError(completed.stderr.strip())
    return float(statistics.median(json.loads(completed.stdout)))


def _ontology(*, optional_doses: bool = False) -> ReporterResponseConditionOntology:
    del optional_doses
    return DEFAULT_CONDITION_ONTOLOGY


def _policy() -> ReporterResponseObservationPolicy:
    return ReporterResponseObservationPolicy(
        policy_id="rt_lnrna_reporter_response_observation_policy.v3",
        pairing_kind="pooled_controls_by_design",
        within_acquisition_reduction_statistic="median",
        biological_replicate_uncertainty_policy=UncertaintyPolicy(
            minimum_biological_replicates=2,
            biological_replicate_reduction_statistic="median",
        ),
    )


def _rows(*, quality_columns: bool = True, optional_doses: bool = False) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    timepoints = tuple(index / 6.0 for index in range(109))
    for observation_group in ("group-1", "group-2"):
        for position_index, position in enumerate(
            (
                f"{observation_group}-position-1",
                f"{observation_group}-position-2",
                f"{observation_group}-position-3",
            )
        ):
            treatments = [
                ("0 nm aTc; 0 uM IPTG", 0.0),
                ("200 nm aTc; 0 uM IPTG", 40.0),
                ("0 nm aTc; 500 uM IPTG", 20.0),
            ]
            if optional_doses:
                treatments.extend((("0 nm aTc; 5 uM IPTG", 5.0), ("0 nm aTc; 50 uM IPTG", 12.0)))
            for treatment, offset in treatments:
                for time_h in timepoints:
                    od = 1.0 + position_index / 10.0
                    for channel, value in (
                        ("RFP", (100.0 + offset) * od),
                        ("OD600", od),
                        ("RFP/OD600", 100.0 + offset),
                    ):
                        row: dict[str, object] = {
                            "type": "SAMPLE",
                            "position": position,
                            "time": time_h,
                            "channel": channel,
                            "value": value,
                            "treatment": treatment,
                            "design_id": "reader-anchor-alias",
                        }
                        if quality_columns:
                            row.update(
                                value_policy_clipped=False,
                                value_instrument_overflow=False,
                                value_bound_kind="exact",
                            )
                        rows.append(row)
    return rows


def _source_closed_inputs(
    tmp_path: Path,
    *,
    replicate_kind: str = "biological",
    replicate_identity_field: str | None = None,
    quality_columns: bool = True,
    optional_doses: bool = False,
) -> tuple[ReaderDataframeRecordRef, ReaderEvidenceBindingSet]:
    experiment_id = DEFAULT_PROTOCOL.planned_kinetic_experiment_ids[0]
    artifact = tmp_path / "outputs" / "artifacts" / "sample_measurements" / "df.parquet"
    artifact.parent.mkdir(parents=True)
    frame = pd.DataFrame(_rows(quality_columns=quality_columns, optional_doses=optional_doses))
    if replicate_identity_field is not None:
        frame[replicate_identity_field] = frame["position"].map(
            lambda value: "replicate-1" if str(value).startswith("group-1") else "replicate-2"
        )
    frame.to_parquet(artifact, index=False)
    digest = "sha256:" + hashlib.sha256(artifact.read_bytes()).hexdigest()
    manifest = tmp_path / "outputs" / "manifests" / "records.json"
    manifest.parent.mkdir(parents=True)
    manifest.write_text("{}\n", encoding="utf-8")
    record = ReaderDataframeRecordRef._from_source_closed_reader(
        reader_root=tmp_path,
        experiment_id=experiment_id,
        protocol_id="plate_reader/single_reporter_screen",
        replicate_kind=replicate_kind,
        replicate_identity_field=replicate_identity_field,
        record_id="sample_measurements/df",
        record_kind="dataframe_artifact",
        record_schema_version=6,
        revision=1,
        revision_digest=_REVISION_DIGEST,
        contract_id="plate_reader.annotated.v1",
        reader_path="artifacts/sample_measurements/df.parquet",
        path=artifact,
        manifest_path=manifest,
        content_digest=digest,
    )
    binding = ReaderEvidenceBinding(
        reader_experiment_id=experiment_id,
        reader_protocol_id=record.protocol_id,
        reader_replicate_kind=replicate_kind,
        reader_replicate_identity_field=replicate_identity_field,
        reader_record_id=record.record_id,
        reader_record_kind=record.record_kind,
        reader_record_schema_version=record.record_schema_version,
        reader_record_revision=record.revision,
        reader_record_revision_digest=record.revision_digest,
        reader_record_contract_id=record.contract_id,
        reader_record_content_digest=record.content_digest,
        reader_record_path=record.reader_path,
        raw_design_id="reader-anchor-alias",
        raw_assay_subject_id=None,
        subject_id=_SUBJECT_ID,
        observation_identity_field="position",
        observation_identity_values=tuple(
            f"group-{group}-position-{position}" for group in (1, 2) for position in (1, 2, 3)
        ),
        biological_replicate_identity_scopes=(
            tuple(
                BiologicalReplicateIdentityScope(
                    condition_value=condition,
                    biological_replicate_id=replicate_id,
                )
                for condition in sorted(set(frame["treatment"].astype(str)))
                for replicate_id in ("replicate-1", "replicate-2")
            )
            if replicate_identity_field is not None
            else ()
        ),
        binding_state="bound",
        binding_reason="exact_subject_alias_match",
    )
    bindings = ReaderEvidenceBindingSet._from_source_closed_record(
        schema_id="rt_lnrna_reader_evidence_bindings_v4",
        subject_binding_set_id="rt_lnrna_subject_bindings_v1",
        rows=(binding,),
    )
    return record, bindings


def _rehash(record: ReaderDataframeRecordRef, bindings: ReaderEvidenceBindingSet) -> None:
    digest = "sha256:" + hashlib.sha256(record.path.read_bytes()).hexdigest()
    object.__setattr__(record, "content_digest", digest)
    object.__setattr__(bindings.rows[0], "reader_record_content_digest", digest)


__all__ = [
    "_SUBJECT_ID",
    "_ontology",
    "_policy",
    "_reader_reduce_trace_rows",
    "_rehash",
    "_source_closed_inputs",
]
