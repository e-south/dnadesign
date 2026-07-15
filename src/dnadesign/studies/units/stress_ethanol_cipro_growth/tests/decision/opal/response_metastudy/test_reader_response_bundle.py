"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/studies/units/stress_ethanol_cipro_growth/tests/decision/opal/response_metastudy/test_reader_response_bundle.py

Tests for the manifest-only Reader-to-study handoff.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd
import pytest

from dnadesign.studies.units.stress_ethanol_cipro_growth.decision.opal.response_metastudy.runtime import (
    selected_reader_rows,
)
from dnadesign.studies.units.stress_ethanol_cipro_growth.response_window_observations import (
    reader_bundle,
)


def test_reader_bundle_loads_only_after_contract_and_digest_verification(tmp_path: Path) -> None:
    root = _bundle_fixture(tmp_path)
    request_path = tmp_path / "request.yaml"
    request_path.write_text("schema_version: reader.response_window.request.v3\n", encoding="utf-8")
    _attach_request_digest(root, request_path)

    bundle = reader_bundle.load_reader_response_bundle(root, expected_request_path=request_path)

    assert bundle.primary_reduction_id == "primary"
    assert len(bundle.designs) == 2
    assert len(bundle.bootstrap_draws) == 2
    assert len(bundle.wells) == 2
    assert len(bundle.traces) == 2
    assert bundle.response_examples == {"d": "Response example"}


def test_reader_bundle_rejects_record_digest_drift(tmp_path: Path) -> None:
    root = _bundle_fixture(tmp_path)
    request_path = tmp_path / "request.yaml"
    request_path.write_text("schema_version: reader.response_window.request.v3\n", encoding="utf-8")
    _attach_request_digest(root, request_path)
    pd.DataFrame({"changed": [1]}).to_parquet(root / "tables" / "designs.parquet", index=False)

    with pytest.raises(ValueError, match="digest mismatch"):
        reader_bundle.load_reader_response_bundle(root, expected_request_path=request_path)


def test_reader_bundle_rejects_manifest_path_decoy_for_record_artifact(tmp_path: Path) -> None:
    root = _bundle_fixture(tmp_path)
    request_path = tmp_path / "request.yaml"
    request_path.write_text("schema_version: reader.response_window.request.v3\n", encoding="utf-8")
    _attach_request_digest(root, request_path)
    verified_path = root / "verified" / "designs.parquet"
    verified_path.parent.mkdir()
    verified_path.write_bytes((root / "tables" / "designs.parquet").read_bytes())
    decoy = pd.read_parquet(root / "tables" / "designs.parquet")
    decoy.loc[0, "r00"] = 99.0
    decoy.to_parquet(root / "tables" / "designs.parquet", index=False)
    manifest_path = root / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["artifacts"]["tables/designs.parquet"] = {
        "path": "verified/designs.parquet",
        "sha256": _sha256(verified_path),
        "bytes": verified_path.stat().st_size,
    }
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(ValueError, match="path disagrees with its manifest identity"):
        reader_bundle.load_reader_response_bundle(root, expected_request_path=request_path)


def test_reader_bundle_rejects_study_request_drift(tmp_path: Path) -> None:
    root = _bundle_fixture(tmp_path)
    request_path = tmp_path / "request.yaml"
    request_path.write_text("schema_version: reader.response_window.request.v3\n", encoding="utf-8")
    _attach_request_digest(root, request_path)
    request_path.write_text("schema_version: reader.response_window.request.v3\nstudy_id: changed\n", encoding="utf-8")

    with pytest.raises(ValueError, match="request digest disagrees"):
        reader_bundle.load_reader_response_bundle(root, expected_request_path=request_path)


def test_reader_bundle_rejects_incomplete_display_ontology(tmp_path: Path) -> None:
    root = _bundle_fixture(tmp_path)
    request_path = tmp_path / "request.yaml"
    request_path.write_text("schema_version: reader.response_window.request.v3\n", encoding="utf-8")
    _attach_request_digest(root, request_path)
    manifest_path = root / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    del manifest["display"]["state_labels"]["01"]
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(ValueError, match="label every response state"):
        reader_bundle.load_reader_response_bundle(root, expected_request_path=request_path)


def test_reader_bundle_rejects_duplicate_manifest_keys(tmp_path: Path) -> None:
    root = _bundle_fixture(tmp_path)
    request_path = tmp_path / "request.yaml"
    request_path.write_text("schema_version: reader.response_window.request.v3\n", encoding="utf-8")
    _attach_request_digest(root, request_path)
    manifest_path = root / "manifest.json"
    raw = manifest_path.read_text(encoding="utf-8")
    manifest_path.write_text(raw.replace("{", '{"study_id":"ambiguous",', 1), encoding="utf-8")

    with pytest.raises(ValueError, match="duplicate JSON key"):
        reader_bundle.load_reader_response_bundle(root, expected_request_path=request_path)


def test_selected_bootstrap_draws_reject_missing_candidate(tmp_path: Path) -> None:
    values = {
        "r00": 0.0,
        "r10": 1.0,
        "r01": 0.5,
        "r11": 1.5,
        "b00": 0.0,
        "b10": 0.2,
        "b01": 0.1,
        "b11": 0.3,
    }
    bundle = reader_bundle.ReaderResponseBundle(
        root=tmp_path,
        manifest_path=tmp_path / "manifest.json",
        manifest={"primary_reduction_id": "primary"},
        designs=pd.DataFrame(),
        bootstrap_draws=pd.DataFrame(
            [
                {
                    "experiment_id": "exp",
                    "design_id": "present",
                    "reduction_id": "primary",
                    "draw_index": 0,
                    "is_reference": False,
                    **values,
                }
            ]
        ),
        wells=pd.DataFrame(),
        traces=pd.DataFrame(),
        events=pd.DataFrame(),
    )
    candidate_identity_bindings = pd.DataFrame(
        [
            {"id": "candidate", "design_id": "missing", "reader_experiment_id": "exp"},
        ]
    )

    with pytest.raises(ValueError, match="lacks one or more selected labels"):
        selected_reader_rows.build_selected_bootstrap_draws(
            bundle,
            candidate_identity_bindings=candidate_identity_bindings,
        )


def _bundle_fixture(tmp_path: Path) -> Path:
    root = tmp_path / "reader-bundle"
    tables = root / "tables"
    tables.mkdir(parents=True)
    values = {"r00": 0.0, "r10": 1.0, "r01": 0.5, "r11": 1.5, "b00": 0.0, "b10": 0.2, "b01": 0.1, "b11": 0.3}
    provenance = {
        f"{prefix}{state}_{suffix}": False if suffix.startswith("has_") else "exact"
        for prefix in ("r", "b")
        for state in ("00", "10", "01", "11")
        for suffix in ("has_policy_clipping", "has_instrument_overflow", "bound_kind")
    }
    event_provenance = {
        f"{prefix}{state}_event_sensitivity_has_{cause}": False
        for prefix in ("r", "b")
        for state in ("00", "10", "01", "11")
        for cause in ("policy_clipping", "instrument_overflow")
    }
    designs = pd.DataFrame(
        [
            {
                "experiment_id": "exp",
                "design_id": "d",
                "reduction_id": "primary",
                "reduction_role": "primary",
                "is_reference": False,
                **values,
                **provenance,
                **event_provenance,
            },
            {
                "experiment_id": "exp",
                "design_id": "ref",
                "reduction_id": "primary",
                "reduction_role": "primary",
                "is_reference": True,
                **values,
                **provenance,
                **event_provenance,
            },
        ]
    )
    draws = designs.assign(draw_index=[0, 0])
    events = pd.DataFrame(
        [
            {
                "experiment_id": "exp",
                "event_id": "event",
                "event_interval_start_assay_h": 1.0,
                "event_interval_end_assay_h": 1.5,
                "event_time_estimate_assay_h": 1.25,
            }
        ]
    )
    frames = {
        "designs": designs,
        "bootstrap_draws": draws,
        "events": events,
        "wells": pd.DataFrame(
            {
                "experiment_id": ["exp", "exp"],
                "design_id": ["d", "ref"],
                "reduction_id": ["primary", "primary"],
                "state": ["00", "00"],
                "position": ["A1", "A2"],
                "response_well": [0.0, 0.0],
                "magnitude_well": [0.0, 0.0],
                "response_policy_clipped_point_count": [0, 0],
                "response_instrument_overflow_point_count": [0, 0],
                "response_bound_kind": ["exact", "exact"],
                "magnitude_policy_clipped_point_count": [0, 0],
                "magnitude_instrument_overflow_point_count": [0, 0],
                "magnitude_bound_kind": ["exact", "exact"],
                "is_reference": [False, True],
            }
        ),
        "traces": pd.DataFrame(
            {
                "experiment_id": ["exp", "exp"],
                "design_id": ["d", "ref"],
                "position": ["A1", "A2"],
                "state": ["00", "00"],
                "time_from_event_h": [1.0, 1.0],
                "value": [0.2, 0.3],
                "value_policy_clipped": [False, False],
                "value_instrument_overflow": [False, False],
                "value_bound_kind": ["exact", "exact"],
                "signal_kind": ["growth", "growth"],
                "is_reference": [False, True],
            }
        ),
    }
    for record_id, frame in frames.items():
        frame.to_parquet(tables / f"{record_id}.parquet", index=False)
    artifacts = {}
    for path in tables.glob("*.parquet"):
        relative = path.relative_to(root).as_posix()
        artifacts[relative] = {"path": relative, "sha256": _sha256(path), "bytes": path.stat().st_size}
    manifest = {
        "schema_version": reader_bundle.READER_BUNDLE_SCHEMA,
        "study_id": "stress_ethanol_cipro_growth",
        "request_id": "test",
        "request": {"artifact_id": "request.yaml", "sha256": "pending"},
        "state_order": ["00", "10", "01", "11"],
        "display": {
            "schema_version": reader_bundle.READER_DISPLAY_SCHEMA,
            "study_label": "Example response study",
            "event_label": "Stress addition",
            "state_labels": {
                "00": "No stress",
                "10": "Ethanol",
                "01": "Ciprofloxacin",
                "11": "Ethanol + ciprofloxacin",
            },
            "channels": {
                "response_ratio": "YFP/CFP",
                "magnitude_ratio": "YFP/OD600",
                "growth": "OD600",
                "reference_design_id": "ref",
            },
            "examples": [
                {"design_id": "ref", "label": "Reference anchor", "role": "reference_anchor"},
                {"design_id": "d", "label": "Response example", "role": "response_example"},
            ],
        },
        "primary_reduction_id": "primary",
        "contracts": reader_bundle.EXPECTED_CONTRACTS,
        "records": {
            record_id: {
                "contract_id": reader_bundle.EXPECTED_CONTRACTS[record_id],
                "artifact_id": artifact_id,
            }
            for record_id, artifact_id in reader_bundle.EXPECTED_RECORD_ARTIFACTS.items()
        },
        "counts": {
            "design_rows": 2,
            "bootstrap_draw_rows": 2,
            "well_rows": 2,
            "trace_rows": 2,
            "experiments": 1,
        },
        "artifacts": artifacts,
    }
    (root / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    return root


def _attach_request_digest(root: Path, request_path: Path) -> None:
    bundled_request = root / "request.yaml"
    bundled_request.write_bytes(request_path.read_bytes())
    manifest_path = root / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    digest = _sha256(bundled_request)
    manifest["request"]["sha256"] = digest
    manifest["artifacts"]["request.yaml"] = {
        "path": "request.yaml",
        "sha256": digest,
        "bytes": bundled_request.stat().st_size,
    }
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")


def _sha256(path: Path) -> str:
    return "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()
