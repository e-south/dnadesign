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
from io import BytesIO
from pathlib import Path

import pandas as pd
import pytest
import yaml

from dnadesign.studies.core.reader_records import ReaderArtifactFile, ReaderRecordSet, ReaderResolvedRecord
from dnadesign.studies.units.stress_ethanol_cipro_growth.response_window_observations import reader_records

PROJECTION = Path(
    "src/dnadesign/studies/units/stress_ethanol_cipro_growth/"
    "response_window_observations/config/reader_response_projection.yaml"
)


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
    assert result.designs["design_id"].tolist() == ["pDual-10", "pDual-10-spyp"]
    receipt = result.source_receipt()
    assert receipt["schema_version"] == "stress_ethanol_cipro_growth.reader_response_projection.v3"
    assert set(receipt["records"]) == set(reader_records.EXPECTED_RECORDS)


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


def test_attested_study_projection_is_deeply_immutable(monkeypatch, tmp_path: Path) -> None:
    source = _source(tmp_path)
    monkeypatch.setattr(reader_records, "resolve_digest_verified_records", lambda *_args, **_kwargs: source)
    result = reader_records.load_reader_response_records(
        reader_root=source.reader_root,
        experiment_root=source.experiment_root,
        projection_path=PROJECTION,
    )
    receipt = result.source_receipt()
    display = result.projection["display"]
    assert isinstance(display, Mapping)
    channels = display["channels"]
    assert isinstance(channels, Mapping)
    examples = display["examples"]
    assert isinstance(examples, Sequence)
    first_example = examples[0]
    assert isinstance(first_example, Mapping)

    mutations = (
        (result.projection, "primary_reduction_id", "different-reduction"),
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

    assert display.source_experiment_id == "source-a"
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


def _source(tmp_path: Path) -> ReaderRecordSet:
    reader_root = tmp_path / "reader"
    experiment = reader_root / "experiments/2026/20260717_stress_response_window_aggregate"
    outputs = experiment / "outputs"
    catalog = outputs / "manifests/records.json"
    catalog.parent.mkdir(parents=True)
    catalog.write_text("{}", encoding="utf-8")
    config = experiment / "config.yaml"
    config.write_text("schema: reader/v8\n", encoding="utf-8")
    frames = _frames()
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


def _pinned_projection(tmp_path: Path, *, missing_path: bool = False) -> Path:
    payload = yaml.safe_load(PROJECTION.read_text(encoding="utf-8"))
    payload["display_artifact"] = {
        "record_id": "plot:four_state_event_window_diagnostic",
        "source_experiment_id": "source-a",
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
        contract_id=contract_id,
        producer={},
        producer_config_digest=None,
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
    designs = pd.DataFrame(
        [
            {
                "experiment_id": "source-a",
                "design_id": design_id,
                "reduction_id": "event_logmean_4_8h_post",
                "reduction_role": "primary",
                "is_reference": is_reference,
                **values,
                **bounds,
                **sensitivity,
            }
            for design_id, is_reference in (("pDual-10", True), ("pDual-10-spyp", False))
        ]
    )
    draws = designs.loc[:, ["experiment_id", "design_id", "reduction_id", "is_reference"]].copy()
    draws["draw_index"] = 0
    for name in reader_records.VALUE_COLUMNS:
        draws[name] = 1.0
    wells = pd.DataFrame(
        [
            {
                "experiment_id": "source-a",
                "design_id": "pDual-10",
                "reduction_id": "event_logmean_4_8h_post",
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
                "is_reference": True,
            }
            for index, state in enumerate(reader_records.STATE_ORDER, start=1)
        ]
    )
    traces = pd.DataFrame(
        [
            {
                "experiment_id": "source-a",
                "design_id": "pDual-10",
                "position": f"A{index}",
                "state": state,
                "time_from_event_h": 4.0,
                "value": 1.0,
                "value_policy_clipped": False,
                "value_instrument_overflow": False,
                "value_bound_kind": "exact",
                "signal_kind": "response",
                "is_reference": True,
            }
            for index, state in enumerate(reader_records.STATE_ORDER, start=1)
        ]
    )
    events = pd.DataFrame(
        [
            {
                "experiment_id": "source-a",
                "event_id": "stress_addition",
                "event_interval_start_assay_h": 0.0,
                "event_interval_end_assay_h": 0.0,
                "event_time_estimate_assay_h": 0.0,
            }
        ]
    )
    return {
        "designs": designs,
        "descriptive_resampling_draws": draws,
        "wells": wells,
        "traces": traces,
        "events": events,
    }
