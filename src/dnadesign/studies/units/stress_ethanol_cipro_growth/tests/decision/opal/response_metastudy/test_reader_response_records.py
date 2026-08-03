"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/studies/units/stress_ethanol_cipro_growth/tests/decision/opal/response_metastudy/test_reader_response_records.py

Study-projection tests for canonical Reader response-window records.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import hashlib
import operator
from collections.abc import Mapping, Sequence
from dataclasses import fields as dataclass_fields
from dataclasses import replace
from io import BytesIO
from pathlib import Path

import pandas as pd
import pytest
import yaml

from dnadesign.studies.core.reader_records import ReaderArtifactFile, ReaderRecordSet, ReaderResolvedRecord
from dnadesign.studies.units.stress_ethanol_cipro_growth.response_window_observations import (
    reader_config_attestation,
    reader_records,
)
from dnadesign.studies.units.stress_ethanol_cipro_growth.response_window_observations.reader_config_attestation import (
    ReaderResponseConfigAttestation,
    expected_reader_analysis,
)
from dnadesign.studies.units.stress_ethanol_cipro_growth.response_window_observations.reader_projection import (
    load_reader_response_projection,
)

PROJECTION = Path(
    "src/dnadesign/studies/units/stress_ethanol_cipro_growth/"
    "response_window_observations/config/reader_response_projection.yaml"
)
PROJECTION_PAYLOAD = yaml.safe_load(PROJECTION.read_text(encoding="utf-8"))
SOURCE_IDS = tuple(PROJECTION_PAYLOAD["source_experiment_ids"])
DIAGNOSTIC_SOURCE = "20260622_sfxi_sensor-panel-m9-glu-29-30-sulAp-spyp"
CONFIG_DIGEST = "sha256:" + "0" * 64


@pytest.fixture(autouse=True)
def _stub_config_attestation(monkeypatch: pytest.MonkeyPatch) -> None:
    def attest(source, projection, **_kwargs):
        return ReaderResponseConfigAttestation(
            config_sha256=hashlib.sha256(source.config_path.read_bytes()).hexdigest(),
            authoring_sha256="b" * 64,
            analysis=expected_reader_analysis(projection),
        )

    monkeypatch.setattr(reader_records, "attest_reader_response_config", attest)


def test_study_projection_parses_verified_reader_bytes(monkeypatch, tmp_path: Path) -> None:
    source = _source(tmp_path)
    monkeypatch.setattr(reader_records, "resolve_digest_verified_records", lambda *_args, **_kwargs: source)

    result = reader_records.load_reader_response_records(
        reader_root=source.reader_root,
        experiment_root=source.experiment_root,
        projection_path=PROJECTION,
    )

    assert result.primary_reduction_id == "event_logmean_4_8h_post"
    assert result.reference_design_id == "pDual-10"
    assert set(result.designs["design_id"]) == {"pDual-10", "pDual-10-spyp"}
    receipt = result.source_receipt()
    assert receipt["schema_version"] == "stress_ethanol_cipro_growth.reader_response_projection.v5"
    assert receipt["config"]["schema_version"] == reader_config_attestation.CONFIG_ATTESTATION_SCHEMA
    assert set(receipt["records"]) == set(reader_records.EXPECTED_RECORDS)
    assert len(result.source_receipt_sha256()) == 64


def test_reader_public_authoring_attestation_matches_projection(monkeypatch, tmp_path: Path) -> None:
    source = _source(tmp_path)
    projection = load_reader_response_projection(PROJECTION)
    payload = _inspect_payload(source, analysis=expected_reader_analysis(projection))
    monkeypatch.setattr(reader_config_attestation, "run_reader_json", lambda *_args, **_kwargs: payload)
    monkeypatch.setattr(reader_config_attestation, "verify_record_store", lambda *_args, **_kwargs: None)

    result = reader_config_attestation.attest_reader_response_config(source, projection, reader_command=("reader",))

    assert result.to_dict()["analysis"] == expected_reader_analysis(projection)
    assert result.config_sha256 == hashlib.sha256(source.config_path.read_bytes()).hexdigest()
    assert len(result.authoring_sha256) == 64


@pytest.mark.parametrize("drift", ["state_value", "channel", "random_seed", "pre_window_duration"])
def test_reader_public_authoring_attestation_rejects_scientific_config_drift(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    drift: str,
) -> None:
    source = _source(tmp_path)
    projection = load_reader_response_projection(PROJECTION)
    analysis = expected_reader_analysis(projection)
    if drift == "state_value":
        analysis["source"]["state_values"]["10"] = "different state value"
    elif drift == "channel":
        analysis["source"]["response_channel"] = "different channel"
    elif drift == "random_seed":
        analysis["aggregation"]["random_seed"] = 1730
    else:
        analysis["reductions"][-1]["pre_window_duration_h"] = 2.0
    payload = _inspect_payload(source, analysis=analysis)
    monkeypatch.setattr(reader_config_attestation, "run_reader_json", lambda *_args, **_kwargs: payload)
    monkeypatch.setattr(reader_config_attestation, "verify_record_store", lambda *_args, **_kwargs: None)

    with pytest.raises(ValueError, match="analysis settings disagree"):
        reader_config_attestation.attest_reader_response_config(
            source,
            projection,
            reader_command=("reader",),
        )


def test_study_projection_receipt_uses_the_same_snapshot_as_parsing(monkeypatch, tmp_path: Path) -> None:
    source = _source(tmp_path)
    projection_path = tmp_path / "reader_response_projection.yaml"
    original_bytes = PROJECTION.read_bytes()
    projection_path.write_bytes(original_bytes)

    def resolve_after_concurrent_edit(*_args, **_kwargs):
        edited = yaml.safe_load(original_bytes)
        edited["primary_reduction_id"] = "concurrently-edited-reduction"
        projection_path.write_text(yaml.safe_dump(edited, sort_keys=False), encoding="utf-8")
        return source

    monkeypatch.setattr(reader_records, "resolve_digest_verified_records", resolve_after_concurrent_edit)

    result = reader_records.load_reader_response_records(
        reader_root=source.reader_root,
        experiment_root=source.experiment_root,
        projection_path=projection_path,
    )

    assert result.primary_reduction_id == "event_logmean_4_8h_post"
    assert result.source_receipt()["projection_sha256"] == hashlib.sha256(original_bytes).hexdigest()
    assert projection_path.read_bytes() != original_bytes


def test_record_resolution_rejects_revision_drift_during_config_attestation(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    source = _source(tmp_path)
    designs = source.records["designs"]
    changed_values = {field.name: getattr(designs, field.name) for field in dataclass_fields(designs) if field.init}
    changed_values.update(
        revision=designs.revision + 1,
        revision_digest="sha256:" + "f" * 64,
    )
    changed_designs = ReaderResolvedRecord._verified(**changed_values)
    confirmed = replace(source, records={**source.records, "designs": changed_designs})
    snapshots = iter((source, confirmed))
    monkeypatch.setattr(
        reader_records,
        "resolve_digest_verified_records",
        lambda *_args, **_kwargs: next(snapshots),
    )

    with pytest.raises(reader_records.ReaderResponseRecordError, match="changed during config attestation"):
        reader_records.load_reader_response_records(
            reader_root=source.reader_root,
            experiment_root=source.experiment_root,
            projection_path=PROJECTION,
        )


def test_attested_study_projection_is_deeply_immutable(monkeypatch, tmp_path: Path) -> None:
    source = _source(tmp_path)
    monkeypatch.setattr(reader_records, "resolve_digest_verified_records", lambda *_args, **_kwargs: source)
    result = reader_records.load_reader_response_records(
        reader_root=source.reader_root,
        experiment_root=source.experiment_root,
        projection_path=PROJECTION,
    )
    receipt = result.source_receipt()
    display = result.projection.payload["display"]
    assert isinstance(display, Mapping)
    channels = display["channels"]
    assert isinstance(channels, Mapping)
    examples = display["examples"]
    assert isinstance(examples, Sequence)
    first_example = examples[0]
    assert isinstance(first_example, Mapping)

    mutations = (
        (result.projection.payload, "primary_reduction_id", "different-reduction"),
        (channels, "reference_design_id", "different-reference"),
        (examples, 0, {"design_id": "different-design"}),
        (first_example, "label", "different-label"),
    )
    for target, key, value in mutations:
        with pytest.raises(TypeError):
            operator.setitem(target, key, value)

    assert result.primary_reduction_id == "event_logmean_4_8h_post"
    assert result.reference_design_id == "pDual-10"
    assert result.response_examples == {
        "pDual-10-spyp": "SpyP measured ethanol-response example",
        "pDual-10-sulAp": "sulAp measured ciprofloxacin-response example",
    }
    assert result.source_receipt() == receipt


def test_unpinned_diagnostic_fails_closed_without_reader_resolution(monkeypatch, tmp_path: Path) -> None:
    source = _source(tmp_path)
    monkeypatch.setattr(reader_records, "resolve_digest_verified_records", lambda *_args, **_kwargs: source)
    projection_payload = yaml.safe_load(PROJECTION.read_text(encoding="utf-8"))
    projection_payload["display_artifact"] = None
    unpinned_projection = tmp_path / "reader_response_projection.yaml"
    unpinned_projection.write_text(yaml.safe_dump(projection_payload, sort_keys=False), encoding="utf-8")
    records = reader_records.load_reader_response_records(
        reader_root=source.reader_root,
        experiment_root=source.experiment_root,
        projection_path=unpinned_projection,
    )

    with pytest.raises(reader_records.ReaderResponseRecordError, match="no display_artifact pin"):
        reader_records.load_reader_response_display_record(records)


def test_pinned_diagnostic_resolves_exact_record_inputs_config_and_path(monkeypatch, tmp_path: Path) -> None:
    source = _source(tmp_path)
    projection = _pinned_projection(tmp_path)
    monkeypatch.setattr(reader_records, "resolve_digest_verified_records", lambda *_args, **_kwargs: source)
    records = reader_records.load_reader_response_records(
        reader_root=source.reader_root,
        experiment_root=source.experiment_root,
        projection_path=projection,
    )
    diagnostic_source = _source_with_diagnostic(source)
    monkeypatch.setattr(
        reader_records,
        "resolve_digest_verified_records",
        lambda *_args, **_kwargs: diagnostic_source,
    )

    display = reader_records.load_reader_response_display_record(records)

    assert display.source_experiment_id == DIAGNOSTIC_SOURCE
    assert display.design_id == "pDual-10-spyp"
    assert display.record.record_id == "plot:four_state_event_window_diagnostic"
    assert display.selected_file.reader_path == "plots/four_state_event_window_diagnostic.png"


@pytest.mark.parametrize(
    ("drift", "match"),
    [
        ("producer_config", "producer-config digest"),
        ("designs_revision", "designs input"),
        ("selected_path", "pinned path must resolve exactly once"),
        ("media_signature", "PNG signature"),
    ],
)
def test_pinned_diagnostic_rejects_record_input_config_path_or_media_drift(
    monkeypatch,
    tmp_path: Path,
    drift: str,
    match: str,
) -> None:
    source = _source(tmp_path)
    projection = _pinned_projection(tmp_path, missing_path=drift == "selected_path")
    monkeypatch.setattr(reader_records, "resolve_digest_verified_records", lambda *_args, **_kwargs: source)
    records = reader_records.load_reader_response_records(
        reader_root=source.reader_root,
        experiment_root=source.experiment_root,
        projection_path=projection,
    )
    diagnostic_source = _source_with_diagnostic(source, drift=drift)
    monkeypatch.setattr(
        reader_records,
        "resolve_digest_verified_records",
        lambda *_args, **_kwargs: diagnostic_source,
    )

    with pytest.raises(reader_records.ReaderResponseRecordError, match=match):
        reader_records.load_reader_response_display_record(records)


def test_projection_rejects_duplicate_yaml_keys(tmp_path: Path) -> None:
    path = tmp_path / "projection.yaml"
    path.write_text(PROJECTION.read_text(encoding="utf-8") + "\nstudy_id: duplicate\n", encoding="utf-8")

    with pytest.raises(reader_records.ReaderResponseRecordError, match="duplicate key"):
        reader_records.load_reader_response_records(
            reader_root=tmp_path / "reader",
            experiment_root=tmp_path / "experiment",
            projection_path=path,
        )


def test_projection_rejects_nonfinite_reduction_window(tmp_path: Path) -> None:
    payload = yaml.safe_load(PROJECTION.read_text(encoding="utf-8"))
    payload["reductions"][0]["window_end_event_h"] = float("nan")
    path = tmp_path / "projection.yaml"
    path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")

    with pytest.raises(reader_records.ReaderResponseRecordError, match="must be finite"):
        reader_records.load_reader_response_records(
            reader_root=tmp_path / "reader",
            experiment_root=tmp_path / "experiment",
            projection_path=path,
        )


@pytest.mark.parametrize(
    ("drift", "match"),
    [
        ("source", "source experiments disagree"),
        ("event", "event identity disagrees"),
        ("reduction", "reduction contract disagrees"),
        ("missing_reduction", "does not cover every projected source and reduction"),
        ("reference", "normalization reference disagrees"),
        ("draw_count", "descriptive-resampling draws"),
    ],
)
def test_projection_rejects_source_event_reduction_reference_or_draw_drift(
    monkeypatch,
    tmp_path: Path,
    drift: str,
    match: str,
) -> None:
    frames = _frames()
    if drift == "source":
        frames["traces"] = frames["traces"].loc[~frames["traces"]["experiment_id"].eq(SOURCE_IDS[0])]
    elif drift == "event":
        frames["events"].loc[:, "event_kind"] = "different_event"
    elif drift == "reduction":
        frames["designs"].loc[frames["designs"]["reduction_id"].eq("event_logmean_4_8h_post"), "reduction_method"] = (
            "arithmetic_mean"
        )
    elif drift == "missing_reduction":
        mask = frames["designs"]["experiment_id"].eq(SOURCE_IDS[0]) & frames["designs"]["reduction_id"].eq(
            "event_logmean_0_6h_post"
        )
        frames["designs"] = frames["designs"].loc[~mask]
    elif drift == "reference":
        frames["designs"].loc[:, "reference_design_id"] = "different-reference"
    else:
        identity = frames["descriptive_resampling_draws"].iloc[0][["experiment_id", "design_id", "reduction_id"]]
        mask = (
            frames["descriptive_resampling_draws"]["experiment_id"].eq(identity["experiment_id"])
            & frames["descriptive_resampling_draws"]["design_id"].eq(identity["design_id"])
            & frames["descriptive_resampling_draws"]["reduction_id"].eq(identity["reduction_id"])
            & frames["descriptive_resampling_draws"]["draw_index"].eq(499)
        )
        frames["descriptive_resampling_draws"] = frames["descriptive_resampling_draws"].loc[~mask]
    source = _source(tmp_path, frames=frames)
    monkeypatch.setattr(reader_records, "resolve_digest_verified_records", lambda *_args, **_kwargs: source)

    with pytest.raises(ValueError, match=match):
        reader_records.load_reader_response_records(
            reader_root=source.reader_root,
            experiment_root=source.experiment_root,
            projection_path=PROJECTION,
        )


def _source(tmp_path: Path, *, frames: dict[str, pd.DataFrame] | None = None) -> ReaderRecordSet:
    reader_root = tmp_path / "reader"
    experiment = reader_root / "experiments/2026/20260717_stress_response_window_aggregate"
    outputs = experiment / "outputs"
    catalog = outputs / "manifests/records.json"
    catalog.parent.mkdir(parents=True)
    catalog.write_text("{}", encoding="utf-8")
    config = experiment / "config.yaml"
    config.write_text("schema: reader/v8\n", encoding="utf-8")
    frames = _frames() if frames is None else frames
    records = {
        name: _record(
            record_id=record_id,
            contract_id=contract_id,
            frame=frames[name],
            path=outputs / f"tables/{name}.parquet",
        )
        for name, (record_id, contract_id) in reader_records.EXPECTED_RECORDS.items()
    }
    return ReaderRecordSet(
        reader_root=reader_root,
        experiment_root=experiment,
        config_path=config,
        outputs_root=outputs,
        catalog_path=catalog,
        catalog_sha256=hashlib.sha256(catalog.read_bytes()).hexdigest(),
        catalog_schema_version=4,
        provenance_epoch_id="epoch-test",
        experiment_id="20260717_stress_response_window_aggregate",
        protocol_id="plate_reader/four_state_event_window",
        experiment_evidence={},
        records=records,
    )


def _inspect_payload(source: ReaderRecordSet, *, analysis: dict[str, object]) -> dict[str, object]:
    return {
        "schema": "reader.cli/v1",
        "command": "inspect",
        "ok": True,
        "meta": {"projection": "section:authoring", "truncated": False, "continuation": None},
        "data": {
            "experiment": {
                "id": source.experiment_id,
                "protocol": source.protocol_id,
                "config": str(source.config_path),
                "root": str(source.experiment_root),
            },
            "authoring": {"inputs": {}, "analysis": analysis, "outputs": {}},
        },
    }


def _pinned_projection(tmp_path: Path, *, missing_path: bool = False) -> Path:
    payload = yaml.safe_load(PROJECTION.read_text(encoding="utf-8"))
    payload["display_artifact"] = {
        "record_id": "plot:four_state_event_window_diagnostic",
        "source_experiment_id": DIAGNOSTIC_SOURCE,
        "design_id": "pDual-10-spyp",
        "producer_config_digest": "sha256:" + "c" * 64,
        "path": "plots/missing.png" if missing_path else "plots/four_state_event_window_diagnostic.png",
    }
    path = tmp_path / "pinned_reader_response_projection.yaml"
    path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
    return path


def _source_with_diagnostic(source: ReaderRecordSet, *, drift: str | None = None) -> ReaderRecordSet:
    content = b"not a png" if drift == "media_signature" else b"\x89PNG\r\n\x1a\nreader diagnostic"
    path = source.outputs_root / "plots/four_state_event_window_diagnostic.png"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(content)
    designs_revision = source.records["designs"].revision_digest
    if drift == "designs_revision":
        designs_revision = "sha256:" + "f" * 64
    diagnostic = ReaderResolvedRecord._verified(
        record_id="plot:four_state_event_window_diagnostic",
        kind="file_bundle",
        schema_version=6,
        revision=1,
        revision_digest="sha256:" + "d" * 64,
        config_digest=CONFIG_DIGEST,
        contract_id=None,
        producer={
            "kind": "plot",
            "id": "four_state_event_window_diagnostic",
            "plugin": "plot/four_state_event_window_diagnostic",
        },
        producer_config_digest=("sha256:" + ("e" if drift == "producer_config" else "c") * 64),
        inputs=(
            {
                "label": "designs",
                "kind": "record",
                "record": source.records["designs"].record_id,
                "discovery_policy": "record",
                "record_revision_digest": designs_revision,
            },
            {
                "label": "traces",
                "kind": "record",
                "record": source.records["traces"].record_id,
                "discovery_policy": "record",
                "record_revision_digest": source.records["traces"].revision_digest,
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
                path=path,
                size_bytes=len(content),
                content_digest="sha256:" + hashlib.sha256(content).hexdigest(),
                content=content,
            ),
        ),
    )
    return ReaderRecordSet(
        reader_root=source.reader_root,
        experiment_root=source.experiment_root,
        config_path=source.config_path,
        outputs_root=source.outputs_root,
        catalog_path=source.catalog_path,
        catalog_sha256=source.catalog_sha256,
        catalog_schema_version=source.catalog_schema_version,
        provenance_epoch_id=source.provenance_epoch_id,
        experiment_id=source.experiment_id,
        protocol_id=source.protocol_id,
        experiment_evidence=source.experiment_evidence,
        records={**source.records, "diagnostic": diagnostic},
    )


def _record(
    *,
    record_id: str,
    contract_id: str,
    frame: pd.DataFrame,
    path: Path,
) -> ReaderResolvedRecord:
    stream = BytesIO()
    frame.to_parquet(stream, index=False)
    content = stream.getvalue()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(content)
    digest = "sha256:" + hashlib.sha256(content).hexdigest()
    return ReaderResolvedRecord._verified(
        record_id=record_id,
        kind="dataframe_artifact",
        schema_version=6,
        revision=1,
        revision_digest="sha256:" + ("a" * 64),
        config_digest=CONFIG_DIGEST,
        contract_id=contract_id,
        producer={
            "kind": "pipeline",
            "id": "four_state_event_window",
            "plugin": "protocol/plate_reader_four_state_event_window",
        },
        producer_config_digest="sha256:" + "9" * 64,
        inputs=(),
        path=path,
        reader_path=path.relative_to(path.parents[1]).as_posix(),
        size_bytes=len(content),
        content_digest=digest,
        content=content,
        files=(),
    )


def _frames() -> dict[str, pd.DataFrame]:
    values = {name: 1.0 for name in reader_records.VALUE_COLUMNS}
    bounds = {
        f"{prefix}{state}_{suffix}": "exact" if suffix == "bound_kind" else False
        for prefix in ("r", "b")
        for state in reader_records.STATE_ORDER
        for suffix in ("has_policy_clipping", "has_instrument_overflow", "bound_kind")
    }
    sensitivity = {
        f"{prefix}{state}_event_sensitivity_has_{cause}": False
        for prefix in ("r", "b")
        for state in reader_records.STATE_ORDER
        for cause in ("policy_clipping", "instrument_overflow")
    }
    aggregation = PROJECTION_PAYLOAD["aggregation"]
    event = PROJECTION_PAYLOAD["event"]
    design_rows = []
    for experiment_id in SOURCE_IDS:
        design_ids = ["pDual-10", *(["pDual-10-spyp"] if experiment_id == DIAGNOSTIC_SOURCE else [])]
        for reduction in PROJECTION_PAYLOAD["reductions"]:
            for design_id in design_ids:
                design_rows.append(
                    {
                        "experiment_id": experiment_id,
                        "design_id": design_id,
                        "reference_design_id": "pDual-10",
                        "reduction_id": reduction["id"],
                        "reduction_method": reduction["method"],
                        "response_basis": reduction["response_basis"],
                        "reduction_role": reduction["role"],
                        "event_id": event["event_id"],
                        "observation_stat": aggregation["observation_stat"],
                        "descriptive_resampling_draws": aggregation["descriptive_resampling_draws"],
                        "descriptive_interval_mass": aggregation["descriptive_interval_mass"],
                        "positive_floor": aggregation["quality"]["positive_floor"],
                        "allowed_max_interior_gap_h": aggregation["quality"]["allowed_max_interior_gap_h"],
                        "required_min_observations_per_state": aggregation["quality"][
                            "required_min_observations_per_state"
                        ],
                        "window_start_event_h": reduction["window_start_event_h"],
                        "window_end_event_h": reduction["window_end_event_h"],
                        "is_reference": design_id == "pDual-10",
                        **values,
                        **bounds,
                        **sensitivity,
                    }
                )
    designs = pd.DataFrame(design_rows)
    draws = pd.DataFrame.from_records(
        [
            {
                "experiment_id": row.experiment_id,
                "design_id": row.design_id,
                "reduction_id": row.reduction_id,
                "draw_index": draw_index,
                "is_reference": row.is_reference,
                **values,
            }
            for row in designs.itertuples(index=False)
            for draw_index in range(aggregation["descriptive_resampling_draws"])
        ]
    )
    wells = pd.DataFrame.from_records(
        [
            {
                "experiment_id": row.experiment_id,
                "design_id": row.design_id,
                "reduction_id": row.reduction_id,
                "reduction_method": row.reduction_method,
                "response_basis": row.response_basis,
                "reduction_role": row.reduction_role,
                "window_start_event_h": row.window_start_event_h,
                "window_end_event_h": row.window_end_event_h,
                "state": state,
                "position": f"A{index}",
                "response_well": f"A{index}",
                "magnitude_well": f"B{index}",
                "response_policy_clipped_point_count": 0,
                "response_instrument_overflow_point_count": 0,
                "response_bound_kind": "exact",
                "magnitude_policy_clipped_point_count": 0,
                "magnitude_instrument_overflow_point_count": 0,
                "magnitude_bound_kind": "exact",
                "is_reference": row.is_reference,
            }
            for row in designs.itertuples(index=False)
            for index, state in enumerate(reader_records.STATE_ORDER, start=1)
        ]
    )
    design_identities = designs.loc[:, ["experiment_id", "design_id", "is_reference"]].drop_duplicates()
    traces = pd.DataFrame.from_records(
        [
            {
                "experiment_id": row.experiment_id,
                "design_id": row.design_id,
                "position": f"A{index}",
                "state": state,
                "time_from_event_h": 4.0,
                "value": 1.0,
                "value_policy_clipped": False,
                "value_instrument_overflow": False,
                "value_bound_kind": "exact",
                "signal_kind": "response",
                "is_reference": row.is_reference,
            }
            for row in design_identities.itertuples(index=False)
            for index, state in enumerate(reader_records.STATE_ORDER, start=1)
        ]
    )
    events = pd.DataFrame(
        [
            {
                "experiment_id": experiment_id,
                "event_id": event["event_id"],
                "event_kind": event["event_kind"],
                "event_interval_start_assay_h": 0.0,
                "event_interval_end_assay_h": 0.0,
                "event_time_estimate_assay_h": 0.0,
                "event_time_estimate_method": event["estimate_method"],
                "declaration": event["declaration"],
            }
            for experiment_id in SOURCE_IDS
        ]
    )
    return {
        "designs": designs,
        "descriptive_resampling_draws": draws,
        "wells": wells,
        "traces": traces,
        "events": events,
    }
