"""Contract tests for the study-owned SFXI reference-overlay recipe."""

from __future__ import annotations

import ast
import hashlib
import inspect
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import yaml

from dnadesign.opal.api.sfxi import score_vec8
from dnadesign.studies.units.stress_ethanol_cipro_growth.decision.opal.sfxi_reference_overlay import (
    reader_records,
    recipe,
)


def _checkout_root() -> Path:
    return next(parent for parent in Path(__file__).resolve().parents if (parent / "pyproject.toml").is_file())


def _live_checkout_root() -> Path:
    checkout = _checkout_root()
    git_marker = checkout / ".git"
    if not git_marker.is_file():
        return checkout
    git_dir = Path(git_marker.read_text(encoding="utf-8").removeprefix("gitdir:").strip())
    if not git_dir.is_absolute():
        git_dir = (checkout / git_dir).resolve()
    return git_dir.parents[1].parent


def _revision_digest(record: dict[str, object]) -> str:
    raw = json.dumps(record, ensure_ascii=True, sort_keys=True, separators=(",", ":"), default=str).encode()
    return f"sha256:{hashlib.sha256(raw).hexdigest()}"


def _write_verified_fixture(tmp_path: Path) -> tuple[Path, Path, Path]:
    usr_root = tmp_path / "usr"
    dataset = usr_root / recipe.DEFAULT_OUTPUT_DATASET
    dataset.mkdir(parents=True)
    sequences = ["ACGTACGT", "AACCGGTT", "AAAACCCC", "GGGGTTTT", "ACACGTGT"]
    pq.write_table(pa.table({"id": [f"usr-{i}" for i in range(5)], "sequence": sequences}), dataset / "records.parquet")

    reader_root = tmp_path / "reader"
    experiment_id = "20260101_fixture"
    experiment_root = reader_root / f"experiments/2026/{experiment_id}"
    outputs = experiment_root / "outputs"
    artifact = outputs / "artifacts/four_state_vector.transform_four_state_vector/vector.parquet"
    artifact.parent.mkdir(parents=True)
    frame = pa.table(
        {
            "design_id": [f"design-{i}" for i in range(5)],
            "sequence": sequences,
            "time_selected_h": [12.0] * 5,
            "reference_design_id": ["reference"] * 5,
            "r_logic": [2.0] * 5,
            "v00": [0.0] * 5,
            "v10": [0.0] * 5,
            "v01": [0.0] * 5,
            "v11": [1.0] * 5,
            "y00_star": [0.0] * 5,
            "y10_star": [0.0] * 5,
            "y01_star": [0.0] * 5,
            "y11_star": [0.0, 1.0, 2.0, 3.0, 4.0],
            "flat_logic": [False] * 5,
        }
    )
    pq.write_table(frame, artifact)
    content_digest = f"sha256:{hashlib.sha256(artifact.read_bytes()).hexdigest()}"
    manifest = outputs / "manifests/records.json"
    manifest.parent.mkdir(parents=True)
    record = {
        "record_id": "four_state_vector/vector",
        "kind": "dataframe_artifact",
        "contract_id": "logic.four_state_vector.v1",
        "schema_version": 6,
        "content_digest": content_digest,
        "config_digest": f"sha256:{'1' * 64}",
        "code_digest": f"sha256:{'2' * 64}",
        "producer_config_digest": f"sha256:{'3' * 64}",
        "build_identity": {"reader_version": "1.0.0", "source_digest": f"sha256:{'2' * 64}"},
        "path": "artifacts/four_state_vector.transform_four_state_vector/vector.parquet",
        "producer": {
            "id": "four_state_vector",
            "kind": "pipeline",
            "plugin": "transform/four_state_vector",
        },
    }
    manifest.write_text(
        json.dumps(
            {
                "schema_version": 4,
                "latest": {"four_state_vector/vector": record},
                "history": {"four_state_vector/vector": [record]},
            }
        ),
        encoding="utf-8",
    )
    (experiment_root / "config.yaml").write_text(
        (
            f"schema: reader/v8\nexperiment:\n  id: {experiment_id}\n"
            "evidence:\n  replicate_kind: unknown\n  replicate_identity_field: null\n"
        ),
        encoding="utf-8",
    )
    revision_digest = _revision_digest(record)
    selection = tmp_path / "selection.json"
    selection.write_text(
        json.dumps(
            {
                "schema_version": "stress_sfxi_reader_record_selection.v1",
                "selection_id": "fixture-selection",
                "records": [
                    {
                        "manifest": f"experiments/2026/{experiment_id}/outputs/manifests/records.json",
                        "record_id": "four_state_vector/vector",
                        "revision": 1,
                        "revision_digest": revision_digest,
                        "contract_id": "logic.four_state_vector.v1",
                        "record_schema_version": 6,
                        "content_digest": content_digest,
                        "config_digest": f"sha256:{'1' * 64}",
                        "code_digest": f"sha256:{'2' * 64}",
                        "design_ids": [f"design-{i}" for i in range(5)],
                    }
                ],
            },
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    return usr_root, reader_root, selection


def test_verified_preview_has_literal_full_table_parity_and_does_not_write(tmp_path: Path) -> None:
    usr_root, reader_root, selection = _write_verified_fixture(tmp_path)
    preview = recipe.build_overlay_preview(
        usr_root=usr_root,
        dataset_name=recipe.DEFAULT_OUTPUT_DATASET,
        reader_root=reader_root,
        selection_path=selection,
    )

    selection_digest = hashlib.sha256(selection.read_bytes()).hexdigest()
    expected_source = f"reader-record-selection:fixture-selection@sha256:{selection_digest}"
    effect_raw = [1.0, 2.0, 4.0, 8.0, 16.0]
    denominator = 14.399999999999999
    effect_scaled = [value / denominator if value < denominator else 1.0 for value in effect_raw]
    rows = []
    for index in range(5):
        rows.append(
            {
                "id": f"usr-{index}",
                "sfxi_ref__reference_instance_id": f"design-{index}",
                "sfxi_ref__collection_id": recipe.DEFAULT_COLLECTION_ID,
                "sfxi_ref__batch_id": "20260101_fixture",
                "sfxi_ref__campaign_id": recipe.DEFAULT_CAMPAIGN_ID,
                "sfxi_ref__reader_experiment_id": "20260101_fixture",
                "sfxi_ref__reader_experiment_date": 20260101,
                "sfxi_ref__metric_id": recipe.DEFAULT_METRIC_ID,
                "sfxi_ref__metric_value": effect_scaled[index],
                "sfxi_ref__metric_provenance": recipe.HISTORICAL_METRIC_PROVENANCE,
                "sfxi_ref__source_ref": expected_source,
                "sfxi_ref__score_ref": recipe.DEFAULT_SCORE_REF,
                "sfxi_ref__objective_name": "sfxi_v1",
                "sfxi_ref__api_version": "1",
                "sfxi_ref__state_order": ["00", "10", "01", "11"],
                "sfxi_ref__setpoint_name": "and",
                "sfxi_ref__setpoint_vector": [0.0, 0.0, 0.0, 1.0],
                "sfxi_ref__denom_used": denominator,
                "sfxi_ref__denom_percentile": 95,
                "sfxi_ref__logic_fidelity": 1.0,
                "sfxi_ref__effect_raw": effect_raw[index],
                "sfxi_ref__effect_scaled": effect_scaled[index],
                "sfxi_ref__sfxi": effect_scaled[index],
                "sfxi_ref__r_logic": 2.0,
                "sfxi_ref__time_selected_h": 12.0,
                "sfxi_ref__reference_design_id": "reference",
                "sfxi_ref__sequence_source_id": f"usr-{index}",
                "sfxi_ref__clip_lo_mask": False,
                "sfxi_ref__clip_hi_mask": index == 4,
                "sfxi_ref__intensity_disabled": False,
                "sfxi_ref__flat_logic": False,
            }
        )
    expected = pa.Table.from_pylist(rows).sort_by([("id", "ascending")])

    assert preview.table.schema == expected.schema
    assert preview.table.equals(expected)
    assert list((usr_root / recipe.DEFAULT_OUTPUT_DATASET).glob("_derived/**/*")) == []


def test_verified_preview_leaves_unknown_replicate_identity_uninterpreted(tmp_path: Path) -> None:
    usr_root, reader_root, selection = _write_verified_fixture(tmp_path)

    preview = recipe.build_overlay_preview(
        usr_root=usr_root,
        dataset_name=recipe.DEFAULT_OUTPUT_DATASET,
        reader_root=reader_root,
        selection_path=selection,
    )

    assert preview.table.num_rows == 5
    assert not any("replicate" in name for name in preview.table.column_names)


def test_verified_preview_rejects_record_digest_drift_before_scoring(tmp_path: Path) -> None:
    usr_root, reader_root, selection = _write_verified_fixture(tmp_path)
    envelope = json.loads(selection.read_text(encoding="utf-8"))
    envelope["records"][0]["content_digest"] = f"sha256:{'4' * 64}"
    selection.write_text(json.dumps(envelope), encoding="utf-8")

    with pytest.raises(recipe.SchemaError, match="unverified content_digest"):
        recipe.build_overlay_preview(
            usr_root=usr_root,
            dataset_name=recipe.DEFAULT_OUTPUT_DATASET,
            reader_root=reader_root,
            selection_path=selection,
        )


def test_selection_source_ref_digests_the_exact_parsed_bytes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    usr_root, reader_root, selection = _write_verified_fixture(tmp_path)
    original = selection.read_bytes()
    original_digest = hashlib.sha256(original).hexdigest()
    real_json_loads = reader_records.json.loads
    replaced = False

    def parse_then_replace(payload, *args, **kwargs):
        nonlocal replaced
        parsed = real_json_loads(payload, *args, **kwargs)
        is_selection = isinstance(parsed, dict) and parsed.get("schema_version") == reader_records.SELECTION_SCHEMA
        if is_selection and not replaced:
            replaced = True
            mutated = dict(parsed)
            mutated["selection_id"] = "concurrently-mutated-selection"
            selection.write_text(json.dumps(mutated, sort_keys=True), encoding="utf-8")
        return parsed

    monkeypatch.setattr(reader_records.json, "loads", parse_then_replace)
    preview = recipe.build_overlay_preview(
        usr_root=usr_root,
        dataset_name=recipe.DEFAULT_OUTPUT_DATASET,
        reader_root=reader_root,
        selection_path=selection,
    )

    assert replaced
    assert preview.source_ref == f"reader-record-selection:fixture-selection@sha256:{original_digest}"


def test_reader_scoring_uses_the_exact_artifact_bytes_that_were_hashed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    usr_root, reader_root, selection = _write_verified_fixture(tmp_path)
    artifact = (
        reader_root / "experiments/2026/20260101_fixture/outputs/artifacts/"
        "four_state_vector.transform_four_state_vector/vector.parquet"
    )
    original_table = pq.read_table(artifact)
    mutated = original_table.set_column(
        original_table.schema.get_field_index("y11_star"),
        "y11_star",
        pa.array([4.0, 3.0, 2.0, 1.0, 0.0]),
    )
    replacement = tmp_path / "replacement.parquet"
    pq.write_table(mutated, replacement)
    replacement_bytes = replacement.read_bytes()
    real_read_bytes = Path.read_bytes
    replaced = False

    def read_then_replace(path: Path) -> bytes:
        nonlocal replaced
        payload = real_read_bytes(path)
        if path == artifact and not replaced:
            replaced = True
            artifact.write_bytes(replacement_bytes)
        return payload

    monkeypatch.setattr(Path, "read_bytes", read_then_replace)
    preview = recipe.build_overlay_preview(
        usr_root=usr_root,
        dataset_name=recipe.DEFAULT_OUTPUT_DATASET,
        reader_root=reader_root,
        selection_path=selection,
    )

    assert replaced
    assert preview.table["sfxi_ref__effect_raw"].to_pylist() == [1.0, 2.0, 4.0, 8.0, 16.0]


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("contract_id", "logic.four_state_vector.v0", "selection contract"),
        ("record_schema_version", 5, "selection record schema"),
        ("config_digest", f"sha256:{'4' * 64}", "unverified config_digest"),
        ("code_digest", f"sha256:{'4' * 64}", "unverified code_digest"),
    ],
)
def test_verified_preview_rejects_contract_and_revision_drift_before_scoring(
    tmp_path: Path,
    field: str,
    value: object,
    message: str,
) -> None:
    usr_root, reader_root, selection = _write_verified_fixture(tmp_path)
    envelope = json.loads(selection.read_text(encoding="utf-8"))
    envelope["records"][0][field] = value
    selection.write_text(json.dumps(envelope), encoding="utf-8")

    with pytest.raises(recipe.SchemaError, match=message):
        recipe.build_overlay_preview(
            usr_root=usr_root,
            dataset_name=recipe.DEFAULT_OUTPUT_DATASET,
            reader_root=reader_root,
            selection_path=selection,
        )


def test_verified_preview_rejects_catalog_schema_drift(tmp_path: Path) -> None:
    usr_root, reader_root, selection = _write_verified_fixture(tmp_path)
    manifest = reader_root / "experiments/2026/20260101_fixture/outputs/manifests/records.json"
    catalog = json.loads(manifest.read_text(encoding="utf-8"))
    catalog["schema_version"] = 5
    manifest.write_text(json.dumps(catalog), encoding="utf-8")

    with pytest.raises(recipe.SchemaError, match="catalog must use schema 4"):
        recipe.build_overlay_preview(
            usr_root=usr_root,
            dataset_name=recipe.DEFAULT_OUTPUT_DATASET,
            reader_root=reader_root,
            selection_path=selection,
        )


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("revision", 2, "revision count mismatch"),
        ("revision_digest", f"sha256:{'4' * 64}", "revision digest mismatch"),
        ("revision_digest", "sha256:not-a-digest", "must be a lowercase sha256 digest"),
    ],
)
def test_verified_preview_rejects_revision_selection_drift(
    tmp_path: Path,
    field: str,
    value: object,
    message: str,
) -> None:
    usr_root, reader_root, selection = _write_verified_fixture(tmp_path)
    envelope = json.loads(selection.read_text(encoding="utf-8"))
    envelope["records"][0][field] = value
    selection.write_text(json.dumps(envelope), encoding="utf-8")

    with pytest.raises(recipe.SchemaError, match=message):
        recipe.build_overlay_preview(
            usr_root=usr_root,
            dataset_name=recipe.DEFAULT_OUTPUT_DATASET,
            reader_root=reader_root,
            selection_path=selection,
        )


def test_verified_preview_rejects_latest_history_divergence(tmp_path: Path) -> None:
    usr_root, reader_root, selection = _write_verified_fixture(tmp_path)
    manifest = reader_root / "experiments/2026/20260101_fixture/outputs/manifests/records.json"
    catalog = json.loads(manifest.read_text(encoding="utf-8"))
    catalog["latest"]["four_state_vector/vector"] = {
        **catalog["latest"]["four_state_vector/vector"],
        "created_at": "drift",
    }
    manifest.write_text(json.dumps(catalog), encoding="utf-8")

    with pytest.raises(recipe.SchemaError, match="latest revision does not match history"):
        recipe.build_overlay_preview(
            usr_root=usr_root,
            dataset_name=recipe.DEFAULT_OUTPUT_DATASET,
            reader_root=reader_root,
            selection_path=selection,
        )


def test_verified_preview_rejects_malformed_persisted_digest(tmp_path: Path) -> None:
    usr_root, reader_root, selection = _write_verified_fixture(tmp_path)
    manifest = reader_root / "experiments/2026/20260101_fixture/outputs/manifests/records.json"
    catalog = json.loads(manifest.read_text(encoding="utf-8"))
    record = catalog["history"]["four_state_vector/vector"][-1]
    record["producer_config_digest"] = "sha256:malformed"
    catalog["latest"]["four_state_vector/vector"] = record
    manifest.write_text(json.dumps(catalog), encoding="utf-8")
    envelope = json.loads(selection.read_text(encoding="utf-8"))
    envelope["records"][0]["revision_digest"] = _revision_digest(record)
    selection.write_text(json.dumps(envelope), encoding="utf-8")

    with pytest.raises(recipe.SchemaError, match="Reader record producer_config_digest"):
        recipe.build_overlay_preview(
            usr_root=usr_root,
            dataset_name=recipe.DEFAULT_OUTPUT_DATASET,
            reader_root=reader_root,
            selection_path=selection,
        )


@pytest.mark.parametrize(
    ("manifest_path", "message"),
    [
        ("experiments/2026/20260101_fixture/records.json", "canonical experiment catalog location"),
        ("experiments/2026/not-a-date/outputs/manifests/records.json", "valid YYYYMMDD date"),
        ("experiments/2025/20260101_fixture/outputs/manifests/records.json", "year directory"),
    ],
)
def test_verified_preview_rejects_noncanonical_experiment_location(
    tmp_path: Path,
    manifest_path: str,
    message: str,
) -> None:
    usr_root, reader_root, selection = _write_verified_fixture(tmp_path)
    envelope = json.loads(selection.read_text(encoding="utf-8"))
    envelope["records"][0]["manifest"] = manifest_path
    selection.write_text(json.dumps(envelope), encoding="utf-8")

    with pytest.raises(recipe.SchemaError, match=message):
        recipe.build_overlay_preview(
            usr_root=usr_root,
            dataset_name=recipe.DEFAULT_OUTPUT_DATASET,
            reader_root=reader_root,
            selection_path=selection,
        )


@pytest.mark.parametrize(
    ("config", "message"),
    [
        ({"schema": "reader/v9", "experiment": {"id": "20260101_fixture"}}, "must use reader/v8"),
        ({"schema": "reader/v8", "experiment": {"id": "different"}}, "config id does not match"),
    ],
)
def test_verified_preview_rejects_experiment_config_drift(
    tmp_path: Path,
    config: dict[str, object],
    message: str,
) -> None:
    usr_root, reader_root, selection = _write_verified_fixture(tmp_path)
    config_path = reader_root / "experiments/2026/20260101_fixture/config.yaml"
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")

    with pytest.raises(recipe.SchemaError, match=message):
        recipe.build_overlay_preview(
            usr_root=usr_root,
            dataset_name=recipe.DEFAULT_OUTPUT_DATASET,
            reader_root=reader_root,
            selection_path=selection,
        )


@pytest.mark.parametrize(
    "evidence",
    [
        None,
        {"replicate_kind": "unknown", "replicate_identity_field": None},
        {"replicate_kind": "biological", "replicate_identity_field": "biological_replicate_id"},
    ],
    ids=["missing", "unknown", "declared-biological"],
)
def test_verified_preview_does_not_promote_acquisition_metadata_to_replicate_identity(
    tmp_path: Path,
    evidence: dict[str, object] | None,
) -> None:
    usr_root, reader_root, selection = _write_verified_fixture(tmp_path)
    config: dict[str, object] = {
        "schema": "reader/v8",
        "experiment": {"id": "20260101_fixture"},
    }
    if evidence is not None:
        config["evidence"] = evidence
    config_path = reader_root / "experiments/2026/20260101_fixture/config.yaml"
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")

    preview = recipe.build_overlay_preview(
        usr_root=usr_root,
        dataset_name=recipe.DEFAULT_OUTPUT_DATASET,
        reader_root=reader_root,
        selection_path=selection,
    )

    assert preview.table.num_rows == 5
    assert not any("replicate" in name for name in preview.table.column_names)


def test_recipe_exposes_no_scoring_override_and_pins_sfxi_v1() -> None:
    assert "scoring_config" not in inspect.signature(recipe.build_overlay_preview).parameters
    assert recipe.FIXED_SCORING_CONFIG.setpoint_vector == (0.0, 0.0, 0.0, 1.0)
    assert recipe.FIXED_SCORING_CONFIG.scaling_percentile == 95
    assert recipe.FIXED_SCORING_CONFIG.scaling_min_n == 5
    assert recipe.FIXED_SCORING_CONFIG.intensity_log2_offset_delta == 0.0


def test_publication_uses_atomic_usr_create_once_surface() -> None:
    source = inspect.getsource(recipe.publish_overlay)
    assert ".create_overlay(" in source
    assert ".write_overlay_part(" not in source
    assert ".list_overlays(" not in source


def test_live_23_row_selection_is_canonical_and_portable() -> None:
    live_checkout = _live_checkout_root()
    reader_root = live_checkout.parent / "reader"
    usr_root = live_checkout / "src/dnadesign/usr/datasets"
    if not reader_root.is_dir() or not (usr_root / recipe.DEFAULT_OUTPUT_DATASET / "records.parquet").is_file():
        pytest.skip("Private Reader records and USR dataset are unavailable in this checkout.")

    preview = recipe.build_overlay_preview(
        usr_root=usr_root,
        dataset_name=recipe.DEFAULT_OUTPUT_DATASET,
        reader_root=reader_root,
    )

    assert preview.table.num_rows == 23
    assert len(preview.record_digests) == 4
    assert preview.source_ref.startswith("reader-record-selection:stress-sfxi-reference-promoters-v1@sha256:")
    assert str(reader_root) not in preview.source_ref
    assert set(preview.table["sfxi_ref__source_ref"].to_pylist()) == {preview.source_ref}


def test_archived_23_row_math_replays_without_mutating_history() -> None:
    live_checkout = _live_checkout_root()
    vec8_path = (
        live_checkout.parent / "reader/experiments/2026/20260501_sfxi_promoter_setpoint_scatter/outputs/artifacts/"
        "sfxi_vec8_latest_preview_input.preview_import_csv/vec8.parquet"
    )
    historical_dir = live_checkout / "src/dnadesign/usr/datasets/usr_sfxi_pdual10_densegen_promoters/_derived/sfxi_ref"
    historical_paths = sorted(historical_dir.glob("part-*.parquet"))
    if not vec8_path.is_file() or len(historical_paths) != 1:
        pytest.skip("The private historical SFXI archive is unavailable in this checkout.")
    historical_path = historical_paths[0]
    before = hashlib.sha256(historical_path.read_bytes()).hexdigest()
    vec8 = pq.read_table(vec8_path).to_pandas()
    historical = pq.read_table(historical_path).to_pandas().set_index("sfxi_ref__reference_instance_id")
    historical = historical.loc[vec8["design_id"].astype(str)]

    result = score_vec8(vec8.loc[:, recipe.VEC8_COLUMNS].to_numpy(float), recipe.FIXED_SCORING_CONFIG)

    assert len(vec8) == len(historical) == 23
    assert np.max(np.abs(historical["sfxi_ref__denom_used"].to_numpy(float) - result.denom_used)) == 0.0
    for field in ("logic_fidelity", "effect_raw", "effect_scaled", "sfxi"):
        assert np.max(np.abs(historical[f"sfxi_ref__{field}"].to_numpy(float) - getattr(result, field))) == 0.0
    assert hashlib.sha256(historical_path.read_bytes()).hexdigest() == before


def test_recipe_uses_owner_public_surfaces_without_import_path_mutation() -> None:
    package = Path(recipe.__file__).parent
    imports: set[str] = set()
    source = "\n".join(path.read_text(encoding="utf-8") for path in package.glob("*.py"))
    for path in package.glob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imports.update(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module:
                imports.add(node.module)

    assert "dnadesign.opal.api.sfxi" in imports
    assert "dnadesign.usr" in imports
    assert not {module for module in imports if module.split(".", maxsplit=1)[0] in {"reader", "reader_workbench"}}
    assert "sys.path" not in source


def test_usr_join_rejects_one_id_bound_to_multiple_sequences() -> None:
    base = pd.DataFrame({"id": ["usr-a", "usr-a"], "sequence": ["AACCGGTT", "GGCCAATT"]})
    reader = pd.DataFrame({"design_id": ["design-a", "design-b"], "sequence": ["AACCGGTT", "GGCCAATT"]})

    with pytest.raises(recipe.SchemaError, match="duplicate USR ids"):
        recipe._join_usr_ids(base=base, reader=reader)
