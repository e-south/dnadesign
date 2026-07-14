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
    reader_response_bundle,
)


def test_reader_bundle_loads_only_after_contract_and_digest_verification(tmp_path: Path) -> None:
    root = _bundle_fixture(tmp_path)
    request_path = tmp_path / "request.yaml"
    request_path.write_text("schema_version: reader.response_window.request.v3\n", encoding="utf-8")
    _attach_request_digest(root, request_path)

    bundle = reader_response_bundle.load_reader_response_bundle(root, expected_request_path=request_path)

    assert bundle.primary_reduction_id == "primary"
    assert len(bundle.designs) == 2
    assert len(bundle.bootstrap_draws) == 2
    assert bundle.response_examples == {"d": "Response example"}


def test_reader_bundle_rejects_record_digest_drift(tmp_path: Path) -> None:
    root = _bundle_fixture(tmp_path)
    request_path = tmp_path / "request.yaml"
    request_path.write_text("schema_version: reader.response_window.request.v3\n", encoding="utf-8")
    _attach_request_digest(root, request_path)
    pd.DataFrame({"changed": [1]}).to_parquet(root / "tables" / "designs.parquet", index=False)

    with pytest.raises(ValueError, match="digest mismatch"):
        reader_response_bundle.load_reader_response_bundle(root, expected_request_path=request_path)


def test_reader_bundle_rejects_study_request_drift(tmp_path: Path) -> None:
    root = _bundle_fixture(tmp_path)
    request_path = tmp_path / "request.yaml"
    request_path.write_text("schema_version: reader.response_window.request.v3\n", encoding="utf-8")
    _attach_request_digest(root, request_path)
    request_path.write_text("schema_version: reader.response_window.request.v3\nstudy_id: changed\n", encoding="utf-8")

    with pytest.raises(ValueError, match="request digest disagrees"):
        reader_response_bundle.load_reader_response_bundle(root, expected_request_path=request_path)


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
        reader_response_bundle.load_reader_response_bundle(root, expected_request_path=request_path)


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
    bundle = reader_response_bundle.ReaderResponseBundle(
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
        events=pd.DataFrame(),
    )
    candidate_identity_bindings = pd.DataFrame(
        [
            {"id": "candidate", "design_id": "missing", "reader_experiment_id": "exp"},
        ]
    )

    with pytest.raises(ValueError, match="lacks one or more selected labels"):
        reader_response_bundle.build_selected_bootstrap_draws(
            bundle,
            candidate_identity_bindings=candidate_identity_bindings,
        )


def _bundle_fixture(tmp_path: Path) -> Path:
    root = tmp_path / "reader-bundle"
    tables = root / "tables"
    tables.mkdir(parents=True)
    values = {"r00": 0.0, "r10": 1.0, "r01": 0.5, "r11": 1.5, "b00": 0.0, "b10": 0.2, "b01": 0.1, "b11": 0.3}
    designs = pd.DataFrame(
        [
            {
                "experiment_id": "exp",
                "design_id": "d",
                "reduction_id": "primary",
                "reduction_role": "primary",
                "is_reference": False,
                **values,
            },
            {
                "experiment_id": "exp",
                "design_id": "ref",
                "reduction_id": "primary",
                "reduction_role": "primary",
                "is_reference": True,
                **values,
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
        "wells": pd.DataFrame({"placeholder": [1]}),
        "traces": pd.DataFrame({"placeholder": [1]}),
    }
    for record_id, frame in frames.items():
        frame.to_parquet(tables / f"{record_id}.parquet", index=False)
    artifacts = {}
    for path in tables.glob("*.parquet"):
        relative = path.relative_to(root).as_posix()
        artifacts[relative] = {"path": relative, "sha256": _sha256(path), "bytes": path.stat().st_size}
    manifest = {
        "schema_version": reader_response_bundle.READER_BUNDLE_SCHEMA,
        "study_id": "stress_ethanol_cipro_growth",
        "request_id": "test",
        "request": {"artifact_id": "request.yaml", "sha256": "pending"},
        "state_order": ["00", "10", "01", "11"],
        "display": {
            "schema_version": reader_response_bundle.READER_DISPLAY_SCHEMA,
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
        "contracts": reader_response_bundle.EXPECTED_CONTRACTS,
        "records": {
            record_id: {
                "contract_id": reader_response_bundle.EXPECTED_CONTRACTS[record_id],
                "artifact_id": artifact_id,
            }
            for record_id, artifact_id in reader_response_bundle.EXPECTED_RECORD_ARTIFACTS.items()
        },
        "counts": {"design_rows": 2, "bootstrap_draw_rows": 2, "experiments": 1},
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
