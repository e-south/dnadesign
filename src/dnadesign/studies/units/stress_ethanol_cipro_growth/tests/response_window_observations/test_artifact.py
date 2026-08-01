"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/studies/units/stress_ethanol_cipro_growth/tests/response_window_observations/test_artifact.py

Tests for atomic, fail-closed response-window observation publication.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import replace
from inspect import signature
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest

from dnadesign.studies.units.stress_ethanol_cipro_growth.response_window_observations import (
    artifact,
)
from dnadesign.studies.units.stress_ethanol_cipro_growth.response_window_observations.aggregation import (
    VALUE_COLUMNS,
    ResponseWindowObservationPreview,
)
from dnadesign.studies.units.stress_ethanol_cipro_growth.response_window_observations.policy import (
    load_response_window_observation_policy,
)
from dnadesign.studies.units.stress_ethanol_cipro_growth.response_window_observations.reader_record_receipt import (
    READER_RESPONSE_RECORD_CONTRACTS,
)
from dnadesign.studies.units.stress_ethanol_cipro_growth.response_window_observations.repeat_diagnostics import (
    REPEAT_DIAGNOSTIC_COLUMNS,
)
from dnadesign.studies.units.stress_ethanol_cipro_growth.response_window_observations.sources import (
    ResolvedReaderCandidateEvidence,
    ResponseWindowObservationEvidence,
)

CONFIG = Path(
    "src/dnadesign/studies/units/stress_ethanol_cipro_growth/response_window_observations/"
    "config/observation_policy.yaml"
)


def test_materialization_refuses_any_scientific_blocker(tmp_path: Path) -> None:
    evidence = _evidence(tmp_path, blockers=("candidate-a: repeat review required",))

    with pytest.raises(artifact.ResponseWindowObservationArtifactError, match="publication is blocked"):
        artifact.materialize_response_window_observations(
            evidence,
            out_dir=tmp_path / "published",
            allowed_output_root=tmp_path,
        )

    assert not (tmp_path / "published").exists()


def test_materialization_recomputes_bounded_component_blockers(tmp_path: Path) -> None:
    evidence = _evidence(tmp_path)
    evidence.preview.contributions.loc[0, "r01_bound_kind"] = "lower"
    evidence.preview.contributions.loc[0, "r01_has_instrument_overflow"] = True

    with pytest.raises(artifact.ResponseWindowObservationArtifactError, match="censor-aware policy"):
        artifact.materialize_response_window_observations(
            evidence,
            out_dir=tmp_path / "published",
            allowed_output_root=tmp_path,
        )

    assert not (tmp_path / "published").exists()


def test_complete_bundle_round_trips_and_detects_digest_drift(tmp_path: Path) -> None:
    evidence = _evidence(tmp_path)
    output = tmp_path / "published"

    written = artifact.materialize_response_window_observations(
        evidence,
        out_dir=output,
        allowed_output_root=tmp_path,
    )
    verified = artifact.verify_response_window_observations(output, allowed_root=tmp_path)

    assert written.candidate_count == 1
    assert verified.candidate_count == 1
    assert verified.policy_id == evidence.policy.policy_id
    assert verified.y_space == "reader_response_window_vector_v1"
    assert {path.name for path in output.iterdir()} == {
        "manifest.json",
        "observations.parquet",
        "contributions.parquet",
        "bootstrap_draws.parquet",
        "uncertainty.parquet",
        "repeat_diagnostics.parquet",
        "reduction_sensitivity.parquet",
        "event_time_sensitivity.parquet",
    }

    observations = pd.read_parquet(output / "observations.parquet")
    observations.loc[0, "r00"] = 99.0
    observations.to_parquet(output / "observations.parquet", index=False)
    with pytest.raises(artifact.ResponseWindowObservationArtifactError, match="digest mismatch"):
        artifact.verify_response_window_observations(output, allowed_root=tmp_path)


def test_observation_publication_is_create_only(tmp_path: Path) -> None:
    output = tmp_path / "published"
    artifact.materialize_response_window_observations(
        _evidence(tmp_path),
        out_dir=output,
        allowed_output_root=tmp_path,
    )
    before = {path.name: path.read_bytes() for path in output.iterdir()}

    assert "overwrite" not in signature(artifact.materialize_response_window_observations).parameters
    with pytest.raises(artifact.ResponseWindowObservationArtifactError, match="already exists"):
        artifact.materialize_response_window_observations(
            _evidence(tmp_path),
            out_dir=output,
            allowed_output_root=tmp_path,
        )

    assert {path.name: path.read_bytes() for path in output.iterdir()} == before


def test_source_manifest_drift_blocks_publication_before_staging(tmp_path: Path) -> None:
    evidence = _evidence(tmp_path)
    evidence.reader_catalog_path.write_text("changed", encoding="utf-8")

    with pytest.raises(artifact.ResponseWindowObservationArtifactError, match="Reader catalog drift"):
        artifact.materialize_response_window_observations(
            evidence,
            out_dir=tmp_path / "published",
            allowed_output_root=tmp_path,
        )


@pytest.mark.parametrize("source", ["reader", "candidate_bindings"])
def test_source_record_drift_blocks_publication_before_staging(tmp_path: Path, source: str) -> None:
    evidence = _evidence(tmp_path)
    if source == "reader":
        record = evidence.reader_records.record_refs["designs"].path
        message = "Reader 'designs' source record"
    else:
        record = evidence.candidate_bindings_path
        message = "candidate-binding record"
    record.write_bytes(record.read_bytes() + b"changed")

    with pytest.raises(artifact.ResponseWindowObservationArtifactError, match=message):
        artifact.materialize_response_window_observations(
            evidence,
            out_dir=tmp_path / "published",
            allowed_output_root=tmp_path,
        )

    assert not (tmp_path / "published").exists()


def test_source_record_race_after_staging_blocks_atomic_publication(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    evidence = _evidence(tmp_path)
    reader_record = evidence.reader_records.record_refs["designs"].path
    verify_staged = artifact.verify_response_window_observations

    def verify_then_mutate(*args, **kwargs):
        result = verify_staged(*args, **kwargs)
        reader_record.write_bytes(reader_record.read_bytes() + b"raced")
        return result

    monkeypatch.setattr(artifact, "verify_response_window_observations", verify_then_mutate)

    with pytest.raises(artifact.ResponseWindowObservationArtifactError, match="Reader 'designs' source record"):
        artifact.materialize_response_window_observations(
            evidence,
            out_dir=tmp_path / "published",
            allowed_output_root=tmp_path,
        )

    assert not (tmp_path / "published").exists()


def test_verifier_rejects_duplicate_manifest_keys(tmp_path: Path) -> None:
    output = tmp_path / "published"
    artifact.materialize_response_window_observations(
        _evidence(tmp_path),
        out_dir=output,
        allowed_output_root=tmp_path,
    )
    manifest = output / "manifest.json"
    raw = manifest.read_text(encoding="utf-8")
    manifest.write_text(raw.replace("{\n", '{\n  "schema_id": "ambiguous",\n', 1), encoding="utf-8")

    with pytest.raises(artifact.ResponseWindowObservationArtifactError, match="duplicate JSON key"):
        artifact.verify_response_window_observations(output, allowed_root=tmp_path)


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        ("missing", "exactly the five"),
        ("extra", "exactly the five"),
        ("fake_record_id", "identity disagrees"),
        ("fake_contract_id", "identity disagrees"),
        ("empty_experiment", "identity is malformed"),
        ("empty_epoch", "identity is malformed"),
        ("extra_top_level", "identity is malformed"),
        ("extra_catalog", "identity is malformed"),
        ("absolute_path", "path is invalid"),
        ("traversal_path", "path is invalid"),
    ],
)
def test_verifier_requires_exact_reader_projection_receipt(
    tmp_path: Path,
    mutation: str,
    message: str,
) -> None:
    output = tmp_path / "published"
    artifact.materialize_response_window_observations(
        _evidence(tmp_path),
        out_dir=output,
        allowed_output_root=tmp_path,
    )
    manifest_path = output / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    records = manifest["source_manifests"]["reader_records"]["records"]
    if mutation == "missing":
        records.pop("wells")
    elif mutation == "extra":
        records["unexpected"] = dict(records["wells"])
    elif mutation == "fake_record_id":
        records["designs"]["record_id"] = "response_window/not-designs"
    elif mutation == "fake_contract_id":
        records["designs"]["contract_id"] = "plate_reader.response_window.not-designs.v1"
    elif mutation == "empty_experiment":
        manifest["source_manifests"]["reader_records"]["experiment_id"] = ""
    elif mutation == "empty_epoch":
        manifest["source_manifests"]["reader_records"]["catalog"]["provenance_epoch_id"] = ""
    elif mutation == "extra_top_level":
        manifest["source_manifests"]["reader_records"]["unexpected"] = "value"
    elif mutation == "extra_catalog":
        manifest["source_manifests"]["reader_records"]["catalog"]["unexpected"] = "value"
    elif mutation == "absolute_path":
        records["designs"]["path"] = "/tmp/designs.parquet"
    else:
        records["designs"]["path"] = "tables/../designs.parquet"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(artifact.ResponseWindowObservationArtifactError, match=message):
        artifact.verify_response_window_observations(output, allowed_root=tmp_path)


@pytest.mark.parametrize(
    ("column", "value", "message"),
    [
        ("r00", 9.0, "point estimate"),
        ("reader_experiment_count", 2, "experiment count"),
        ("label_source_method", "explicit_repeat_selection", "label-source method"),
        ("label_source_reader_experiment_id", "experiment-z", "label-source experiment"),
    ],
)
def test_verifier_recomputes_observation_semantics_from_contributions(
    tmp_path: Path,
    column: str,
    value: object,
    message: str,
) -> None:
    output = tmp_path / "published"
    artifact.materialize_response_window_observations(
        _evidence(tmp_path),
        out_dir=output,
        allowed_output_root=tmp_path,
    )
    observations = pd.read_parquet(output / "observations.parquet")
    observations.loc[0, column] = value
    _rewrite_record(output, "observations", observations)

    with pytest.raises(artifact.ResponseWindowObservationArtifactError, match=message):
        artifact.verify_response_window_observations(output, allowed_root=tmp_path)


def test_verifier_recomputes_selected_label_source(tmp_path: Path) -> None:
    output = tmp_path / "published"
    artifact.materialize_response_window_observations(
        _evidence(tmp_path),
        out_dir=output,
        allowed_output_root=tmp_path,
    )
    contributions = pd.read_parquet(output / "contributions.parquet")
    contributions.loc[0, "selected_as_label_source"] = False
    _rewrite_record(output, "contributions", contributions)

    with pytest.raises(artifact.ResponseWindowObservationArtifactError, match="included contributions"):
        artifact.verify_response_window_observations(output, allowed_root=tmp_path)


def test_verifier_binds_uncertainty_points_to_observations(tmp_path: Path) -> None:
    output = tmp_path / "published"
    artifact.materialize_response_window_observations(
        _evidence(tmp_path),
        out_dir=output,
        allowed_output_root=tmp_path,
    )
    uncertainty = pd.read_parquet(output / "uncertainty.parquet")
    uncertainty.loc[uncertainty["component"].eq("r00"), "point_estimate"] = 9.0
    _rewrite_record(output, "uncertainty", uncertainty)

    with pytest.raises(artifact.ResponseWindowObservationArtifactError, match="uncertainty point estimate"):
        artifact.verify_response_window_observations(output, allowed_root=tmp_path)


def _rewrite_record(output: Path, record_id: str, frame: pd.DataFrame) -> None:
    path = output / artifact.RECORD_FILES[record_id]
    frame.to_parquet(path, index=False)
    manifest_path = output / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["records"][record_id]["sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")


def _evidence(tmp_path: Path, *, blockers: tuple[str, ...] = ()) -> ResponseWindowObservationEvidence:
    tmp_path.mkdir(parents=True, exist_ok=True)
    reader_root = tmp_path / "reader"
    reader_table_root = reader_root / "tables"
    reader_table_root.mkdir(parents=True, exist_ok=True)
    reader_record_paths: dict[str, Path] = {}
    for name in ("wells", "designs", "descriptive_resampling_draws", "traces", "events"):
        path = reader_table_root / f"{name}.parquet"
        path.write_bytes(f"reader-record:{name}".encode())
        reader_record_paths[name] = path
    reader_catalog = reader_root / "records.json"
    reader_catalog.write_text("{}", encoding="utf-8")
    reader_projection = reader_root / "reader_response_projection.yaml"
    reader_projection.write_text("projection", encoding="utf-8")
    binding_root = tmp_path / "candidate-bindings"
    binding_root.mkdir(parents=True, exist_ok=True)
    binding_manifest = binding_root / "manifest.json"
    binding_rows = binding_root / "bindings.parquet"
    pd.DataFrame({"candidate_id": ["candidate-a"]}).to_parquet(binding_rows, index=False)
    binding_manifest.write_text(
        json.dumps(
            {
                "record": {
                    "path": "bindings.parquet",
                    "sha256": hashlib.sha256(binding_rows.read_bytes()).hexdigest(),
                }
            }
        ),
        encoding="utf-8",
    )
    reader_sha = hashlib.sha256(reader_catalog.read_bytes()).hexdigest()
    projection_sha = hashlib.sha256(reader_projection.read_bytes()).hexdigest()
    binding_sha = hashlib.sha256(binding_manifest.read_bytes()).hexdigest()
    loaded = load_response_window_observation_policy(CONFIG)
    policy = replace(
        loaded,
        approval_status="approved",
        approved_by="study-reviewer",
        approved_at="2026-07-15T12:00:00+00:00",
        candidate_bindings_sha256=binding_sha,
        repeat_decisions=pd.DataFrame(columns=loaded.repeat_decisions.columns),
        aggregation=replace(loaded.aggregation, bootstrap_samples=100),
    )
    observation = {
        "candidate_id": "candidate-a",
        "reader_design_ids": ["design-a"],
        "reader_experiment_count": 1,
        "label_source_reader_experiment_id": "experiment-a",
        "label_source_method": "singleton_identity",
        "display_label": "Candidate A",
        "sequence_sha256": hashlib.sha256(b"ACGT").hexdigest(),
        "source_class": "densegen",
        "design_family": "stress_promoter",
        "baserender_adapter_kind": "densegen_tfbs",
        **{column: 1.0 for column in VALUE_COLUMNS},
    }
    contributions = pd.DataFrame(
        [
            {
                "candidate_id": "candidate-a",
                "design_id": "design-a",
                "reader_experiment_id": "experiment-a",
                "reduction_id": policy.aggregation.primary_reduction_id,
                "repeat_decision": "singleton",
                "repeat_decision_reason": "single_experiment",
                "repeat_classification": None,
                "repeat_evidence_artifact": None,
                "repeat_evidence_sha256": None,
                "repeat_adjudicated_by": None,
                "repeat_adjudicated_at": None,
                "label_source_reader_experiment_id": "experiment-a",
                "selected_as_label_source": True,
                "included_in_label": True,
                "label_exclusion_reason": None,
                **{column: 1.0 for column in VALUE_COLUMNS},
                **{
                    f"{component}_{suffix}": False if suffix != "bound_kind" else "exact"
                    for component in VALUE_COLUMNS
                    for suffix in ("has_policy_clipping", "has_instrument_overflow", "bound_kind")
                },
            }
        ]
    )
    draws = pd.DataFrame.from_records(
        [
            {
                "candidate_id": "candidate-a",
                "draw_index": draw_index,
                **{column: 1.0 for column in VALUE_COLUMNS},
            }
            for draw_index in range(100)
        ]
    )
    uncertainty = pd.DataFrame.from_records(
        [
            {
                "candidate_id": "candidate-a",
                "component": component,
                "label_source_reader_experiment_id": "experiment-a",
                "point_estimate": 1.0,
                "bootstrap_sd": 0.1,
                "descriptive_interval_low": 0.8,
                "descriptive_interval_high": 1.2,
                "nominal_interval_mass": 0.9,
                "interval_scope": "descriptive_selected_source_joint_bootstrap",
                "population_coverage_claimed": False,
                "bootstrap_samples": 100,
            }
            for component in VALUE_COLUMNS
        ]
    )
    reduction = pd.DataFrame(
        [
            {
                "candidate_id": "candidate-a",
                "design_id": "design-a",
                "reader_experiment_id": "experiment-a",
                "reduction_id": policy.aggregation.primary_reduction_id,
                "reduction_role": "primary",
                **{column: 1.0 for column in VALUE_COLUMNS},
                **{f"{column}__delta_from_primary": 0.0 for column in VALUE_COLUMNS},
                "maximum_abs_delta_from_primary": 0.0,
            }
        ]
    )
    event = pd.DataFrame.from_records(
        [
            {
                "candidate_id": "candidate-a",
                "design_id": "design-a",
                "reader_experiment_id": "experiment-a",
                "component": component,
                "event_time_half_range": 0.1,
            }
            for component in VALUE_COLUMNS
        ]
    )
    preview = ResponseWindowObservationPreview(
        observations=pd.DataFrame([observation]),
        contributions=contributions,
        bootstrap_draws=draws,
        uncertainty=uncertainty,
        repeat_diagnostics=pd.DataFrame(columns=REPEAT_DIAGNOSTIC_COLUMNS),
        reduction_sensitivity=reduction,
        event_time_sensitivity=event,
        blockers=blockers,
    )
    record_refs = {
        name: SimpleNamespace(
            path=path,
            content_digest="sha256:" + hashlib.sha256(path.read_bytes()).hexdigest(),
            size_bytes=path.stat().st_size,
        )
        for name, path in reader_record_paths.items()
    }
    receipt_records = {
        name: {
            "record_id": READER_RESPONSE_RECORD_CONTRACTS[name][0],
            "kind": "dataframe_artifact",
            "schema_version": 6,
            "revision": 1,
            "revision_digest": "sha256:" + ("a" * 64),
            "contract_id": READER_RESPONSE_RECORD_CONTRACTS[name][1],
            "path": f"tables/{name}.parquet",
            "size_bytes": record_refs[name].size_bytes,
            "content_digest": record_refs[name].content_digest,
        }
        for name in record_refs
    }
    reader_records = SimpleNamespace(
        record_refs=record_refs,
        source_receipt=lambda: {
            "schema_version": "stress_ethanol_cipro_growth.reader_response_projection.v2",
            "experiment_id": "aggregate",
            "protocol_id": "plate_reader/response_window",
            "catalog": {
                "schema_version": 4,
                "provenance_epoch_id": "epoch-test",
                "sha256": reader_sha,
            },
            "projection_sha256": projection_sha,
            "records": receipt_records,
        },
    )
    return ResponseWindowObservationEvidence(
        policy=policy,
        resolved=ResolvedReaderCandidateEvidence(
            measurements=pd.DataFrame(),
            bootstrap_draws=pd.DataFrame(),
            excluded_reader_designs=pd.DataFrame(),
        ),
        preview=preview,
        reader_records=reader_records,  # type: ignore[arg-type]
        reader_catalog_path=reader_catalog,
        reader_catalog_sha256=reader_sha,
        reader_projection_path=reader_projection,
        reader_projection_sha256=projection_sha,
        candidate_bindings_manifest_path=binding_manifest,
        candidate_bindings_manifest_sha256=binding_sha,
        candidate_bindings_path=binding_rows,
    )
