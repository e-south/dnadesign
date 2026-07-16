"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/opal/tests/storage/test_observed_label_promotion.py

Verification contracts for manifest-pinned observed-label artifacts.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import json
from copy import deepcopy
from dataclasses import replace
from hashlib import sha256
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq
import pytest

from dnadesign.opal.src.core.utils import OpalError, file_sha256
from dnadesign.opal.src.storage.label_sources import ObservedLabelStore
from dnadesign.opal.src.storage.observed_label_promotion import (
    OBSERVED_LABEL_PROMOTION_SCHEMA_VERSION,
    CandidateExclusionSetBinding,
    ObservedLabelPromotionBinding,
    verify_observed_label_promotion,
)

LABEL_RELATIVE_PATH = "_opal/observed_labels.parquet"
MANIFEST_RELATIVE_PATH = "_opal/observed_labels.manifest.json"
PROVENANCE_RELATIVE_PATH = "_opal/study_label_provenance.json"
CANDIDATE_RELATIVE_PATH = "records.parquet"


def _label_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "id": ["candidate_a", "candidate_b"],
            "observed_round": [0, 0],
            "batch_id": ["batch_0", "batch_0"],
            "y_space": ["response_window_vector_v1", "response_window_vector_v1"],
            "y_obs": [[0.0] * 8, [1.0] * 8],
            "src": ["study_promotion", "study_promotion"],
            "ts": ["2026-07-14T00:00:00Z", "2026-07-14T00:00:00Z"],
        }
    )


def _write_promotion(dataset_root: Path) -> tuple[Path, Path, dict]:
    candidate_path = dataset_root / CANDIDATE_RELATIVE_PATH
    pd.DataFrame(
        {
            "id": ["candidate_a", "candidate_b", "candidate_unmeasured"],
            "sequence": ["ACGT", "TGCA", "AAAA"],
            "x_feature": [[0.0, 1.0], [1.0, 0.0], [0.5, 0.5]],
        }
    ).to_parquet(candidate_path, index=False)
    label_path = dataset_root / LABEL_RELATIVE_PATH
    label_path.parent.mkdir(parents=True, exist_ok=True)
    _label_frame().to_parquet(label_path, index=False)
    provenance_path = dataset_root / PROVENANCE_RELATIVE_PATH
    provenance_path.write_text('{"schema_version":"stress-study.labels.v1"}\n', encoding="utf-8")
    payload = {
        "schema_version": OBSERVED_LABEL_PROMOTION_SCHEMA_VERSION,
        "campaign_slug": "promoter",
        "study_id": "stress_promoter",
        "y_space": "response_window_vector_v1",
        "study_provenance": {
            "schema_id": "stress-study.labels.v1",
            "path": PROVENANCE_RELATIVE_PATH,
            "sha256": file_sha256(provenance_path),
        },
        "candidate_exclusion_projection": _candidate_exclusion_projection([]),
        "candidate_artifact": _candidate_artifact(candidate_path),
        "label_artifact": {
            "path": LABEL_RELATIVE_PATH,
            "sha256": file_sha256(label_path),
            "row_count": 2,
        },
    }
    manifest_path = dataset_root / MANIFEST_RELATIVE_PATH
    manifest_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return label_path, manifest_path, payload


def _binding(dataset_root: Path) -> ObservedLabelPromotionBinding:
    return ObservedLabelPromotionBinding(
        dataset_root=dataset_root,
        manifest_path=MANIFEST_RELATIVE_PATH,
        label_path=LABEL_RELATIVE_PATH,
        campaign_slug="promoter",
        study_id="stress_promoter",
        y_space="response_window_vector_v1",
        candidate_x_column="x_feature",
    )


def _candidate_artifact(path: Path) -> dict[str, object]:
    parquet = pq.ParquetFile(path)
    schema = parquet.schema_arrow
    return {
        "path": CANDIDATE_RELATIVE_PATH,
        "sha256": file_sha256(path),
        "row_count": int(parquet.metadata.num_rows),
        "columns": schema.names,
        "schema_sha256": sha256(schema.serialize().to_pybytes()).hexdigest(),
    }


def _candidate_exclusion_projection(entries: list[dict[str, str]]) -> dict[str, object]:
    canonical = json.dumps(
        sorted(entries, key=lambda entry: entry["candidate_id"]),
        sort_keys=True,
        separators=(",", ":"),
    )
    return {
        "exclusion_set_id": "study_observation_dispositions_v1",
        "entries_sha256": sha256(canonical.encode("utf-8")).hexdigest(),
        "entry_count": len(entries),
    }


def _store_with_label_frame(dataset_root: Path, frame: pd.DataFrame) -> ObservedLabelStore:
    label_path, manifest_path, payload = _write_promotion(dataset_root)
    frame.to_parquet(label_path, index=False)
    payload["label_artifact"]["sha256"] = file_sha256(label_path)
    payload["label_artifact"]["row_count"] = len(frame)
    manifest_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return ObservedLabelStore(
        path=label_path,
        y_space="response_window_vector_v1",
        promotion=_binding(dataset_root),
    )


def test_verify_observed_label_promotion_accepts_exact_artifact(tmp_path: Path) -> None:
    label_path, manifest_path, _ = _write_promotion(tmp_path)

    verified = verify_observed_label_promotion(_binding(tmp_path))

    assert verified.manifest_path == manifest_path.resolve()
    assert verified.label_path == label_path.resolve()
    assert verified.label_sha256 == file_sha256(label_path)
    assert verified.row_count == 2
    assert verified.study_provenance_schema_id == "stress-study.labels.v1"
    assert verified.study_provenance_path == (tmp_path / PROVENANCE_RELATIVE_PATH).resolve()
    assert verified.candidate_path == (tmp_path / CANDIDATE_RELATIVE_PATH).resolve()
    assert verified.candidate_row_count == 3
    assert verified.candidate_columns == ("id", "sequence", "x_feature")


def test_verify_observed_label_promotion_binds_campaign_candidate_exclusions(tmp_path: Path) -> None:
    _, manifest_path, payload = _write_promotion(tmp_path)
    entries = [{"candidate_id": "candidate_unmeasured", "reason": "study_excluded"}]
    payload["candidate_exclusion_projection"] = _candidate_exclusion_projection(entries)
    manifest_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    matching = replace(
        _binding(tmp_path),
        candidate_exclusion_sets=(
            CandidateExclusionSetBinding(
                exclusion_set_id="study_observation_dispositions_v1",
                entries=entries,
            ),
        ),
    )

    verified = verify_observed_label_promotion(matching)

    assert verified.candidate_exclusion_set_id == "study_observation_dispositions_v1"
    assert verified.candidate_exclusion_entry_count == 1


@pytest.mark.parametrize(
    ("sets", "message"),
    [
        ((), "missing candidate exclusion set"),
        (
            (
                CandidateExclusionSetBinding(
                    exclusion_set_id="study_observation_dispositions_v1",
                    entries=[{"candidate_id": "candidate_unmeasured", "reason": "different_reason"}],
                ),
            ),
            "digest mismatch",
        ),
    ],
)
def test_verify_observed_label_promotion_rejects_campaign_candidate_exclusion_drift(
    tmp_path: Path,
    sets: tuple[CandidateExclusionSetBinding, ...],
    message: str,
) -> None:
    _, manifest_path, payload = _write_promotion(tmp_path)
    entries = [{"candidate_id": "candidate_unmeasured", "reason": "study_excluded"}]
    payload["candidate_exclusion_projection"] = _candidate_exclusion_projection(entries)
    manifest_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    with pytest.raises(OpalError, match=message):
        verify_observed_label_promotion(replace(_binding(tmp_path), candidate_exclusion_sets=sets))


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (lambda payload: payload.update({"unexpected": "field"}), "fields must be exactly"),
        (lambda payload: payload.update({"schema_version": "opal.observed_label_promotion.v2"}), "schema_version"),
        (lambda payload: payload.update({"campaign_slug": "other"}), "campaign_slug"),
        (lambda payload: payload.update({"study_id": "other"}), "study_id"),
        (lambda payload: payload.update({"y_space": "other"}), "y_space"),
        (lambda payload: payload["study_provenance"].update({"schema_id": ""}), "schema_id"),
        (lambda payload: payload["study_provenance"].update({"path": "../outside.json"}), "path"),
        (lambda payload: payload["study_provenance"].update({"sha256": "0" * 64}), "SHA-256"),
        (
            lambda payload: payload["study_provenance"].update({"unexpected": "field"}),
            "study_provenance fields must be exactly",
        ),
        (lambda payload: payload["candidate_artifact"].update({"path": "other.parquet"}), "path"),
        (lambda payload: payload["candidate_artifact"].update({"sha256": "0" * 64}), "SHA-256"),
        (lambda payload: payload["candidate_artifact"].update({"row_count": 2}), "row_count"),
        (lambda payload: payload["candidate_artifact"].update({"columns": ["id"]}), "candidate/X columns"),
        (lambda payload: payload["candidate_artifact"].update({"schema_sha256": "0" * 64}), "schema identity"),
        (
            lambda payload: payload["candidate_artifact"].update({"unexpected": "field"}),
            "candidate_artifact fields must be exactly",
        ),
        (lambda payload: payload["label_artifact"].update({"path": "_opal/other.parquet"}), "path"),
        (lambda payload: payload["label_artifact"].update({"sha256": "0" * 64}), "SHA-256"),
        (lambda payload: payload["label_artifact"].update({"row_count": 3}), "row_count"),
        (
            lambda payload: payload["label_artifact"].update({"unexpected": "field"}),
            "label_artifact fields must be exactly",
        ),
    ],
)
def test_verify_observed_label_promotion_rejects_contract_drift(
    tmp_path: Path,
    mutation,
    message: str,
) -> None:
    _, manifest_path, payload = _write_promotion(tmp_path)
    mutated = deepcopy(payload)
    mutation(mutated)
    manifest_path.write_text(json.dumps(mutated), encoding="utf-8")

    with pytest.raises(OpalError, match=message):
        verify_observed_label_promotion(_binding(tmp_path))


def test_verify_observed_label_promotion_rejects_dataset_escape(tmp_path: Path) -> None:
    _write_promotion(tmp_path)
    binding = ObservedLabelPromotionBinding(
        dataset_root=tmp_path,
        manifest_path="../observed_labels.manifest.json",
        label_path=LABEL_RELATIVE_PATH,
        campaign_slug="promoter",
        study_id="stress_promoter",
        y_space="response_window_vector_v1",
    )

    with pytest.raises(OpalError, match="manifest_path must remain within the USR dataset root"):
        verify_observed_label_promotion(binding)


def test_verify_observed_label_promotion_rejects_duplicate_json_keys(tmp_path: Path) -> None:
    _, manifest_path, _ = _write_promotion(tmp_path)
    raw = manifest_path.read_text(encoding="utf-8")
    raw = raw.replace(
        '"campaign_slug": "promoter",',
        '"campaign_slug": "promoter",\n  "campaign_slug": "other",',
        1,
    )
    manifest_path.write_text(raw, encoding="utf-8")

    with pytest.raises(OpalError, match="duplicate JSON key.*campaign_slug"):
        verify_observed_label_promotion(_binding(tmp_path))


def test_verify_observed_label_promotion_rejects_nested_duplicate_json_keys(tmp_path: Path) -> None:
    _, manifest_path, _ = _write_promotion(tmp_path)
    raw = manifest_path.read_text(encoding="utf-8")
    raw = raw.replace(
        f'"path": "{PROVENANCE_RELATIVE_PATH}",',
        f'"path": "{PROVENANCE_RELATIVE_PATH}",\n    "path": "_opal/other.json",',
        1,
    )
    manifest_path.write_text(raw, encoding="utf-8")

    with pytest.raises(OpalError, match="duplicate JSON key.*path"):
        verify_observed_label_promotion(_binding(tmp_path))


def test_verify_observed_label_promotion_reports_non_file_artifact_as_contract_error(tmp_path: Path) -> None:
    label_path, _, _ = _write_promotion(tmp_path)
    label_path.unlink()
    label_path.mkdir()

    with pytest.raises(OpalError, match="Failed to hash observed-label promotion artifact"):
        verify_observed_label_promotion(_binding(tmp_path))


def test_manifest_pinned_store_rejects_generic_append_without_mutation(tmp_path: Path) -> None:
    label_path, _, _ = _write_promotion(tmp_path)
    before = file_sha256(label_path)
    store = ObservedLabelStore(
        path=label_path,
        y_space="response_window_vector_v1",
        promotion=_binding(tmp_path),
    )

    with pytest.raises(OpalError, match="manifest-pinned and immutable"):
        store.append_labels(
            pd.DataFrame({"id": ["candidate_a"], "y": [[2.0] * 8]}),
            observed_round=1,
            batch_id="batch_1",
            src="ingest_y",
            if_exists="fail",
            known_ids={"candidate_a", "candidate_b"},
        )

    assert file_sha256(label_path) == before


def test_manifest_pinned_store_reverifies_before_each_load(tmp_path: Path) -> None:
    label_path, _, _ = _write_promotion(tmp_path)
    store = ObservedLabelStore(
        path=label_path,
        y_space="response_window_vector_v1",
        promotion=_binding(tmp_path),
    )
    assert len(store.load()) == 2

    changed = _label_frame()
    changed.at[0, "y_obs"] = [9.0] * 8
    changed.to_parquet(label_path, index=False)

    with pytest.raises(OpalError, match="SHA-256"):
        store.load()


@pytest.mark.parametrize("mutation", ["sequence", "id", "x"], ids=["sequence", "candidate-id", "x-vector"])
def test_manifest_pinned_store_rejects_candidate_snapshot_drift(tmp_path: Path, mutation: str) -> None:
    label_path, _, _ = _write_promotion(tmp_path)
    store = ObservedLabelStore(
        path=label_path,
        y_space="response_window_vector_v1",
        promotion=_binding(tmp_path),
    )
    assert len(store.load()) == 2
    candidate_path = tmp_path / CANDIDATE_RELATIVE_PATH
    candidates = pd.read_parquet(candidate_path)
    if mutation == "sequence":
        candidates.loc[0, "sequence"] = "CCCC"
    elif mutation == "id":
        candidates.loc[0, "id"] = "candidate_changed"
    else:
        candidates.at[0, "x_feature"] = [9.0, 9.0]
    candidates.to_parquet(candidate_path, index=False)

    with pytest.raises(OpalError, match="candidate artifact SHA-256"):
        store.load()


def test_manifest_pinned_store_rejects_store_path_outside_verified_binding(tmp_path: Path) -> None:
    _write_promotion(tmp_path)
    other_path = tmp_path / "_opal" / "other_labels.parquet"
    _label_frame().to_parquet(other_path, index=False)
    store = ObservedLabelStore(
        path=other_path,
        y_space="response_window_vector_v1",
        promotion=_binding(tmp_path),
    )

    with pytest.raises(OpalError, match="store path does not match"):
        store.load()


def test_manifest_pinned_store_rejects_mixed_y_spaces(tmp_path: Path) -> None:
    frame = _label_frame()
    frame.loc[1, "y_space"] = "sfxi.vec8.v3"
    store = _store_with_label_frame(tmp_path, frame)

    with pytest.raises(OpalError, match="one Y space"):
        store.observed_ids()


def test_manifest_pinned_store_rejects_nested_vector_shape(tmp_path: Path) -> None:
    frame = _label_frame()
    frame["y_obs"] = [[[0.0] * 4, [1.0] * 4], [[1.0] * 4, [2.0] * 4]]
    store = _store_with_label_frame(tmp_path, frame)

    with pytest.raises(OpalError, match="one-dimensional"):
        store.observed_ids()


@pytest.mark.parametrize("rounds", [[1.5, 0.0], [-1, 0], [True, False]])
def test_manifest_pinned_store_rejects_invalid_observed_rounds(
    tmp_path: Path,
    rounds: list[object],
) -> None:
    frame = _label_frame()
    frame["observed_round"] = rounds
    store = _store_with_label_frame(tmp_path, frame)

    with pytest.raises(OpalError, match="nonnegative integers"):
        store.observed_ids()
