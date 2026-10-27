"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/densegen/tests/integrations/test_dense_arrays_playback.py

Verify DenseGen-to-dense-arrays playback translation.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import yaml
from dense_arrays.playback import PlaybackDocument, reconstruct_playback
from dense_arrays.playback.theme import PlaybackPresentation
from dense_arrays.realized import Orientation, PlacementKind

from dnadesign.densegen.src.integrations.dense_arrays import publisher
from dnadesign.densegen.src.integrations.dense_arrays.baserender_projection import (
    AnchoredIllustrationPresentation,
    BaseRenderDuplexProjection,
    DuplexPresentation,
)
from dnadesign.densegen.src.integrations.dense_arrays.playback import (
    playback_plan_from_densegen_record,
    realized_array_from_densegen_record,
)
from dnadesign.densegen.src.integrations.dense_arrays.publisher import (
    _selected_records_sha256,
    _selected_rows,
)


def _publisher_row(
    *,
    record_id: str = "record-1",
    generated_at: str = "first",
    placement_label: str = "TF_A",
) -> dict[str, object]:
    return {
        "id": record_id,
        "sequence": "AAATTT",
        "densegen__used_tfbs_detail": [
            {
                "part_kind": "TFBS",
                "sequence": "AAA",
                "offset": 0,
                "offset_raw": 0,
                "end": 3,
                "orientation": "fwd",
                "tfbs_id": "site-1",
                "regulator": placement_label,
            }
        ],
        "densegen__schema_version": "2.9",
        "densegen__run_id": "run-1",
        "densegen__plan": "baseline",
        "densegen__input_name": "fixture",
        "densegen__sampling_library_hash": "library-1",
        "densegen__sampling_library_index": 0,
        "densegen__pad_used": False,
        "densegen__pad_bases": 0,
        "densegen__pad_end": "5prime",
        "generated_at": generated_at,
    }


def _write_endpoint(
    tmp_path: Path,
    *,
    scene: str = "clean_scene",
    formats: tuple[str, ...] = ("manifest.json",),
    placement_label: str = "TF_A",
    record: dict[str, object] | None = None,
) -> Path:
    workspace = tmp_path / "workspace"
    table_path = workspace / "outputs" / "tables" / "records.parquet"
    table_path.parent.mkdir(parents=True)
    row = record or _publisher_row(placement_label=placement_label)
    pq.write_table(pa.Table.from_pylist([row]), table_path)
    selected = _selected_rows(table_path, ("record-1",))
    selected_sha256 = _selected_records_sha256(selected, ("record-1",))
    config = {
        "schema": "densegen.solution_path_playback_endpoint.v2",
        "endpoint_id": "fixture",
        "title": "Fixture endpoint",
        "audience": "public",
        "source": {
            "kind": "densegen_records",
            "table": "outputs/tables/records.parquet",
            "selected_records_sha256": selected_sha256,
            "records": [{"id": "record-1", "scene": scene}],
        },
        "adapter": {
            "kind": "densegen_realized_array_v1",
            "display_coordinate": "offset",
            "solver_coordinate_provenance": "offset_raw",
        },
        "playback": {
            "authority": "placement_reconstructed",
            "ordering_policy": ["start", "shorter_first", "placement_id"],
            "graph_relation": "coordinate_precedence",
            "show_authority_notice": False,
        },
        "labels": {"forbidden_terms": ["sigma factor"], "overrides": {}},
        "presentation": {
            "layout": "graph_left_duplex_right",
            "collection_order": [scene],
        },
        "outputs": {
            "directory": "outputs/publication/fixture",
            "formats": list(formats),
        },
    }
    config_path = workspace / "playback.yaml"
    config_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    return config_path


@pytest.mark.parametrize("schema", ["densegen.solution_path_playback_endpoint.v1", "unknown", None])
def test_publisher_rejects_unsupported_schema_before_reading_records(tmp_path: Path, schema: object) -> None:
    config_path = _write_endpoint(tmp_path)
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    config["schema"] = schema
    config["source"]["table"] = "missing.parquet"
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")

    with pytest.raises(ValueError, match="expected 'densegen.solution_path_playback_endpoint.v2'") as error:
        publisher.publish_densegen_playback_endpoint(config_path)

    assert "endpoint-schema-migration" in str(error.value)
    assert not (config_path.parent / "outputs" / "publication").exists()


def test_reverse_placement_uses_realized_reverse_complement() -> None:
    record = {
        "id": "record-1",
        "sequence": "AAACCGT",
        "densegen__used_tfbs_detail": [
            {
                "part_kind": "tfbs",
                "sequence": "ACGG",
                "offset": 3,
                "offset_raw": 3,
                "end": 7,
                "orientation": "REV",
                "tfbs_id": "tfbs-1",
                "regulator": "TF_A",
            }
        ],
    }

    realized = realized_array_from_densegen_record(
        record,
        source_ref="fixture.parquet",
    )

    assert realized.placements[0].orientation is Orientation.REVERSE
    assert realized.placements[0].sequence == "CCGT"
    assert realized.placements[0].metadata["library_sequence"] == "ACGG"
    assert reconstruct_playback(realized).steps[0].placement_sequence == "CCGT"


def test_missing_legacy_part_kind_defaults_to_tfbs() -> None:
    record = _publisher_row()
    del record["densegen__used_tfbs_detail"][0]["part_kind"]

    realized = realized_array_from_densegen_record(record, source_ref="fixture.parquet")

    assert realized.placements[0].kind is PlacementKind.TFBS


def test_adapter_normalizes_uppercase_forward_orientation() -> None:
    record = _publisher_row()
    detail = record["densegen__used_tfbs_detail"][0]
    detail["part_kind"] = "TFBS"
    detail["orientation"] = "FWD"

    realized = realized_array_from_densegen_record(record, source_ref="fixture.parquet")

    assert realized.placements[0].kind is PlacementKind.TFBS
    assert realized.placements[0].orientation is Orientation.FORWARD


def test_fixed_element_recovers_sequence_consistent_raw_coordinate() -> None:
    record = {
        "id": "record-2",
        "sequence": "AAACCCGGG",
        "densegen__used_tfbs_detail": [
            {
                "part_kind": "fixed_element",
                "sequence": "CCC",
                "offset": 4,
                "offset_raw": 3,
                "pad_left": 1,
                "end": 7,
                "constraint_name": "anchor",
                "placement_index": 0,
                "role": "upstream",
            }
        ],
    }

    realized = realized_array_from_densegen_record(
        record,
        source_ref="fixture.parquet",
    )

    assert realized.placements[0].start == 3
    assert realized.placements[0].metadata["coordinate_source"] == "offset_raw"
    plan = playback_plan_from_densegen_record(record, source_ref="fixture.parquet")
    assert any(notice.code == "coordinate_recovered" for notice in plan.notices)


def test_selected_record_digest_ignores_unselected_runtime_columns(tmp_path: Path) -> None:
    first = tmp_path / "first.parquet"
    second = tmp_path / "second.parquet"
    pq.write_table(pa.Table.from_pylist([_publisher_row(generated_at="first")]), first)
    pq.write_table(pa.Table.from_pylist([_publisher_row(generated_at="second")]), second)

    first_rows = _selected_rows(first, ("record-1",))
    second_rows = _selected_rows(second, ("record-1",))

    assert _selected_records_sha256(first_rows, ("record-1",)) == _selected_records_sha256(
        second_rows,
        ("record-1",),
    )


def test_selected_rows_rejects_duplicate_in_later_batch(tmp_path: Path) -> None:
    table_path = tmp_path / "records.parquet"
    rows = [_publisher_row(record_id="target")]
    rows.extend(_publisher_row(record_id=f"filler-{index}") for index in range(2047))
    rows.append(_publisher_row(record_id="target", generated_at="duplicate"))
    pq.write_table(pa.Table.from_pylist(rows), table_path)

    with pytest.raises(ValueError, match="record id 'target' occurs more than once"):
        _selected_rows(table_path, ("target",))


def test_publisher_confines_replace_to_dedicated_workspace_output(tmp_path: Path) -> None:
    config_path = _write_endpoint(tmp_path)
    payload = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    payload["outputs"]["directory"] = "../escape"
    config_path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")

    with pytest.raises(ValueError, match="relative descendant"):
        publisher.publish_densegen_playback_endpoint(config_path, replace=True)

    payload["outputs"]["directory"] = "outputs/tables"
    config_path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
    with pytest.raises(ValueError, match="must not contain the configured source table"):
        publisher.publish_densegen_playback_endpoint(config_path, replace=True)


@pytest.mark.parametrize("value", [float("nan"), float("inf"), float("-inf")])
def test_publisher_rejects_nonfinite_timing_values(tmp_path: Path, value: float) -> None:
    config_path = _write_endpoint(tmp_path)
    payload = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    payload["presentation"]["hold_seconds"] = value
    config_path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")

    with pytest.raises(ValueError, match="presentation.hold_seconds must be non-negative"):
        publisher.publish_densegen_playback_endpoint(config_path)


def test_publisher_restores_prior_bundle_when_replacement_install_fails(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config_path = _write_endpoint(tmp_path)
    output_path = publisher.publish_densegen_playback_endpoint(config_path)
    marker = output_path / "prior-bundle.txt"
    marker.write_text("prior\n", encoding="utf-8")

    def _fail_exchange(_new_bundle: Path, _prior_bundle: Path) -> None:
        raise OSError("forced atomic exchange failure")

    monkeypatch.setattr(publisher, "_atomic_exchange_directories", _fail_exchange)

    with pytest.raises(OSError, match="forced atomic exchange failure"):
        publisher.publish_densegen_playback_endpoint(config_path, replace=True)

    assert marker.read_text(encoding="utf-8") == "prior\n"
    assert not tuple(output_path.parent.glob(f".{output_path.name}.backup-*"))


@pytest.mark.parametrize("audience", ["public", "study_publication"])
def test_publisher_removes_prior_bundle_after_successful_replacement(tmp_path: Path, audience: str) -> None:
    config_path = _write_endpoint(tmp_path)
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    config["audience"] = audience
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")
    output_path = publisher.publish_densegen_playback_endpoint(config_path)
    assert json.loads((output_path / "manifest.json").read_text())["audience"] == audience
    marker = output_path / "prior-bundle.txt"
    marker.write_text("prior\n", encoding="utf-8")

    replaced_path = publisher.publish_densegen_playback_endpoint(config_path, replace=True)

    assert replaced_path == output_path
    assert not marker.exists()
    assert (output_path / "manifest.json").is_file()
    assert not tuple(output_path.parent.glob(f".{output_path.name}.backup-*"))


def test_publisher_keeps_endpoint_present_across_atomic_exchange(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config_path = _write_endpoint(tmp_path)
    output_path = publisher.publish_densegen_playback_endpoint(config_path)
    marker = output_path / "prior-bundle.txt"
    marker.write_text("prior\n", encoding="utf-8")
    original_exchange = publisher._atomic_exchange_directories
    observed: list[tuple[bool, bool]] = []

    def _observe_exchange(new_bundle: Path, prior_bundle: Path) -> None:
        observed.append((new_bundle.is_dir(), prior_bundle.is_dir()))
        original_exchange(new_bundle, prior_bundle)
        observed.append((new_bundle.is_dir(), prior_bundle.is_dir()))

    monkeypatch.setattr(publisher, "_atomic_exchange_directories", _observe_exchange)

    replaced_path = publisher.publish_densegen_playback_endpoint(config_path, replace=True)

    assert replaced_path == output_path
    assert observed == [(True, True), (True, True)]
    assert not marker.exists()
    assert (output_path / "manifest.json").is_file()


def test_publisher_reports_success_when_only_retired_bundle_cleanup_fails(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config_path = _write_endpoint(tmp_path)
    output_path = publisher.publish_densegen_playback_endpoint(config_path)
    original_rmtree = publisher.shutil.rmtree

    def _fail_retired_bundle_cleanup(path: Path, *args, **kwargs) -> None:
        candidate = Path(path)
        if candidate.name.startswith(f".{output_path.name}.") and (candidate / "manifest.json").exists():
            raise OSError("simulated cleanup denial")
        original_rmtree(path, *args, **kwargs)

    monkeypatch.setattr(publisher.shutil, "rmtree", _fail_retired_bundle_cleanup)

    replaced_path = publisher.publish_densegen_playback_endpoint(config_path, replace=True)

    assert replaced_path == output_path
    assert (output_path / "manifest.json").is_file()
    assert len(tuple(output_path.parent.glob(f".{output_path.name}.*"))) == 1


def test_publisher_retries_retired_bundle_cleanup_on_next_publication(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config_path = _write_endpoint(tmp_path)
    output_path = publisher.publish_densegen_playback_endpoint(config_path)
    original_rmtree = publisher.shutil.rmtree
    denied_once = False

    def _deny_first_retired_bundle_cleanup(path: Path, *args, **kwargs) -> None:
        nonlocal denied_once
        candidate = Path(path)
        if not denied_once and candidate.name.startswith(f".{output_path.name}."):
            denied_once = True
            raise OSError("simulated transient cleanup denial")
        original_rmtree(path, *args, **kwargs)

    monkeypatch.setattr(publisher.shutil, "rmtree", _deny_first_retired_bundle_cleanup)

    publisher.publish_densegen_playback_endpoint(config_path, replace=True)

    retained = tuple(output_path.parent.glob(f".{output_path.name}.backup-*"))
    assert len(retained) == 1
    assert (retained[0] / "manifest.json").is_file()

    monkeypatch.setattr(publisher.shutil, "rmtree", original_rmtree)
    publisher.publish_densegen_playback_endpoint(config_path, replace=True)

    assert not tuple(output_path.parent.glob(f".{output_path.name}.backup-*"))


def test_publisher_recovers_prior_bundle_after_interrupted_rename(tmp_path: Path) -> None:
    config_path = _write_endpoint(tmp_path)
    output_path = publisher.publish_densegen_playback_endpoint(config_path)
    backup_path = output_path.parent / f".{output_path.name}.backup-interrupted"
    output_path.replace(backup_path)

    replaced_path = publisher.publish_densegen_playback_endpoint(config_path, replace=True)

    assert replaced_path == output_path
    assert (output_path / "manifest.json").is_file()
    assert not backup_path.exists()


def test_publisher_validates_default_display_text(tmp_path: Path) -> None:
    config_path = _write_endpoint(tmp_path, scene="sigma_factor_example")

    with pytest.raises(ValueError, match="forbidden term: 'sigma factor'"):
        publisher.publish_densegen_playback_endpoint(config_path)


def test_publisher_rejects_unknown_label_fields(tmp_path: Path) -> None:
    config_path = _write_endpoint(tmp_path)
    payload = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    payload["labels"] = {"forbidden_term": ["sigma factor"], "overrides": {}}
    config_path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")

    with pytest.raises(ValueError, match=r"labels contains unsupported fields: \['forbidden_term'\]"):
        publisher.publish_densegen_playback_endpoint(config_path)


@pytest.mark.parametrize("config_name", ["playback.yaml", "playback-constraints.yaml"])
def test_packaged_playback_configs_follow_supported_schema(config_name: str) -> None:
    config_path = Path(__file__).resolve().parents[2] / "workspaces" / "demo_dense_array_showcase" / config_name
    endpoint = publisher._load_endpoint(config_path).endpoint
    source = publisher._required_mapping(endpoint["source"], field_name="source")
    labels = publisher._required_mapping(endpoint["labels"], field_name="labels")

    assert publisher._SCENE_ID.fullmatch(str(endpoint["endpoint_id"]))
    assert endpoint["audience"] == "public"
    assert source["kind"] == "densegen_records"
    publisher._strict_fields(source, publisher._SOURCE_FIELDS, field_name="source")
    publisher._strict_fields(labels, publisher._LABEL_FIELDS, field_name="labels")
    for index, record in enumerate(source["records"]):
        publisher._strict_fields(record, publisher._RECORD_SPEC_FIELDS, field_name=f"source.records[{index}]")


def test_publisher_rejects_unknown_top_level_endpoint_fields(tmp_path: Path) -> None:
    config_path = _write_endpoint(tmp_path)
    payload = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    payload["unexpected"] = "ignored"
    config_path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")

    with pytest.raises(ValueError, match=r"endpoint contains unsupported fields: \['unexpected'\]"):
        publisher.publish_densegen_playback_endpoint(config_path)


@pytest.mark.parametrize(
    ("endpoint_id", "message"),
    [(None, "endpoint_id must be a non-empty string"), ("Invalid ID", "endpoint_id must match")],
)
def test_publisher_requires_valid_endpoint_id(tmp_path: Path, endpoint_id: str | None, message: str) -> None:
    config_path = _write_endpoint(tmp_path)
    payload = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if endpoint_id is None:
        del payload["endpoint_id"]
    else:
        payload["endpoint_id"] = endpoint_id
    config_path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")

    with pytest.raises(ValueError, match=message):
        publisher.publish_densegen_playback_endpoint(config_path)


@pytest.mark.parametrize(
    ("field_path", "value", "message"),
    [
        (("endpoint_id",), 123, "endpoint_id must be a string"),
        (("source", "records", 0, "title"), 7, "source.records\\[\\].title must be a string"),
        (
            ("labels", "forbidden_terms", 0),
            False,
            "labels.forbidden_terms\\[\\] must be a string",
        ),
        (
            ("labels", "overrides", "TF_A"),
            False,
            "labels.overrides\\[TF_A\\] must be a string",
        ),
        (
            ("presentation", "collection_order", 0),
            False,
            "presentation.collection_order\\[\\] must be a string",
        ),
    ],
)
def test_publisher_rejects_non_string_yaml_text_scalars(
    tmp_path: Path,
    field_path: tuple[str | int, ...],
    value: object,
    message: str,
) -> None:
    config_path = _write_endpoint(tmp_path)
    payload = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    target = payload
    for key in field_path[:-1]:
        target = target[key]
    target[field_path[-1]] = value
    config_path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")

    with pytest.raises(TypeError, match=message):
        publisher.publish_densegen_playback_endpoint(config_path)


@pytest.mark.parametrize(
    ("field_path", "message"),
    [
        (("labels",), "labels must be a mapping"),
        (("labels", "overrides"), "labels.overrides must be a mapping"),
        (("labels", "forbidden_terms"), "labels.forbidden_terms must be a list"),
    ],
)
def test_publisher_rejects_false_yaml_label_containers(
    tmp_path: Path,
    field_path: tuple[str, ...],
    message: str,
) -> None:
    config_path = _write_endpoint(tmp_path)
    payload = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    target = payload
    for key in field_path[:-1]:
        target = target[key]
    target[field_path[-1]] = False
    config_path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")

    with pytest.raises(TypeError, match=message):
        publisher.publish_densegen_playback_endpoint(config_path)


@pytest.mark.parametrize("value", [False, 0, [], ""])
def test_publisher_rejects_explicit_non_mapping_duplex(tmp_path: Path, value: object) -> None:
    config_path = _write_endpoint(tmp_path)
    payload = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    payload["duplex"] = value
    config_path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")

    with pytest.raises(TypeError, match="duplex must be a mapping"):
        publisher.publish_densegen_playback_endpoint(config_path)


def test_publisher_rejects_unknown_audience(tmp_path: Path) -> None:
    config_path = _write_endpoint(tmp_path)
    payload = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    payload["audience"] = "internal"
    config_path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")

    with pytest.raises(ValueError, match=r"audience must be one of \['public', 'study_publication'\]"):
        publisher.publish_densegen_playback_endpoint(config_path)


def test_publisher_rejects_unsupported_source_kind(tmp_path: Path) -> None:
    config_path = _write_endpoint(tmp_path)
    payload = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    payload["source"]["kind"] = "unsupported_kind"
    config_path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")

    with pytest.raises(ValueError, match=r"source.kind must be one of \['densegen_records'\]"):
        publisher.publish_densegen_playback_endpoint(config_path)


def test_publisher_rejects_unknown_record_fields(tmp_path: Path) -> None:
    config_path = _write_endpoint(tmp_path)
    payload = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    payload["source"]["records"][0]["subtitel"] = "misspelled"
    config_path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")

    with pytest.raises(ValueError, match=r"source.records\[0\] contains unsupported fields: \['subtitel'\]"):
        publisher.publish_densegen_playback_endpoint(config_path)


def test_publisher_validates_record_derived_placement_labels(tmp_path: Path) -> None:
    config_path = _write_endpoint(tmp_path, placement_label="Sigma factor RpoD")

    with pytest.raises(ValueError, match="record-derived placement label contains forbidden term: 'sigma factor'"):
        publisher.publish_densegen_playback_endpoint(config_path)


def test_publisher_validates_record_derived_constraint_labels(tmp_path: Path) -> None:
    row = _publisher_row()
    row["densegen__used_tfbs_detail"] = [
        {
            "part_kind": "fixed_element",
            "sequence": "AAA",
            "offset": 0,
            "offset_raw": 0,
            "end": 3,
            "orientation": "fwd",
            "constraint_name": "Sigma factor spacing",
            "placement_index": 0,
            "role": "upstream",
            "variant_id": "upstream",
            "spacer_length": 0,
        },
        {
            "part_kind": "fixed_element",
            "sequence": "TTT",
            "offset": 3,
            "offset_raw": 3,
            "end": 6,
            "orientation": "fwd",
            "constraint_name": "Sigma factor spacing",
            "placement_index": 0,
            "role": "downstream",
            "variant_id": "downstream",
            "spacer_length": 0,
        },
    ]
    config_path = _write_endpoint(tmp_path, record=row)

    with pytest.raises(ValueError, match="record-derived constraint label contains forbidden term: 'sigma factor'"):
        publisher.publish_densegen_playback_endpoint(config_path)


def test_publisher_validates_persisted_plan_with_explicit_subtitle(tmp_path: Path) -> None:
    row = _publisher_row()
    row["densegen__plan"] = "sigma factor baseline"
    config_path = _write_endpoint(tmp_path, record=row)
    payload = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    payload["source"]["records"][0]["subtitle"] = "Clean public subtitle"
    config_path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")

    with pytest.raises(ValueError, match="record-derived plan name contains forbidden term: 'sigma factor'"):
        publisher.publish_densegen_playback_endpoint(config_path)


@pytest.mark.parametrize("field", ["densegen__run_id", "densegen__input_name"])
def test_publisher_validates_serialized_record_provenance(tmp_path: Path, field: str) -> None:
    row = _publisher_row()
    row[field] = "private sigma factor provenance"
    config_path = _write_endpoint(tmp_path, record=row)

    with pytest.raises(ValueError, match="serialized realized-array payload contains forbidden term: 'sigma factor'"):
        publisher.publish_densegen_playback_endpoint(config_path)


def test_publisher_validates_serialized_manifest_fields(tmp_path: Path) -> None:
    config_path = _write_endpoint(tmp_path)
    payload = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    payload["endpoint_id"] = "private_sigma_factor_endpoint"
    payload["labels"]["forbidden_terms"] = ["sigma_factor"]
    config_path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")

    with pytest.raises(ValueError, match="serialized manifest payload contains forbidden term: 'sigma_factor'"):
        publisher.publish_densegen_playback_endpoint(config_path)


def test_publisher_validates_record_derived_variant_annotations(tmp_path: Path) -> None:
    row = _publisher_row()
    row["densegen__used_tfbs_detail"] = [
        {
            "part_kind": "fixed_element",
            "sequence": "AAA",
            "offset": 0,
            "offset_raw": 0,
            "end": 3,
            "orientation": "fwd",
            "constraint_name": "anchor",
            "placement_index": 0,
            "role": "upstream",
            "variant_id": "Sigma factor variant",
            "spacer_length": 0,
        },
        {
            "part_kind": "fixed_element",
            "sequence": "TTT",
            "offset": 3,
            "offset_raw": 3,
            "end": 6,
            "orientation": "fwd",
            "constraint_name": "anchor",
            "placement_index": 0,
            "role": "downstream",
            "variant_id": "consensus",
            "spacer_length": 0,
        },
    ]
    config_path = _write_endpoint(tmp_path, record=row)
    payload = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    payload["duplex"] = {"fixed_element_annotations": "variant"}
    config_path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")

    with pytest.raises(ValueError, match="record-derived variant annotation contains forbidden term: 'sigma factor'"):
        publisher.publish_densegen_playback_endpoint(config_path)


def test_publisher_honors_requested_render_formats(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config_path = _write_endpoint(tmp_path, formats=("manifest.json", "poster.png"))
    calls: list[str] = []

    monkeypatch.setattr(
        publisher,
        "BaseRenderDuplexProjection",
        lambda *_args, **_kwargs: SimpleNamespace(render_rgba=lambda *_inner: None),
    )

    def _poster(_documents, path: Path, **_kwargs) -> None:
        calls.append("poster.png")
        path.write_bytes(b"poster")

    def _unexpected_mp4(*_args, **_kwargs) -> None:
        raise AssertionError("MP4 renderer must not run for a poster-only endpoint")

    monkeypatch.setattr(publisher, "render_collection_poster_png", _poster)
    monkeypatch.setattr(publisher, "render_collection_mp4", _unexpected_mp4)

    output = publisher.publish_densegen_playback_endpoint(config_path)
    manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))

    assert calls == ["poster.png"]
    assert (output / "poster.png").read_bytes() == b"poster"
    assert not (output / "playback.mp4").exists()
    assert manifest["endpoint_id"] == "fixture"
    assert manifest["audience"] == "public"
    assert manifest["requested_formats"] == ["manifest.json", "poster.png"]


def test_manifest_digest_binds_endpoint_bytes_used_before_render_mutation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config_path = _write_endpoint(tmp_path, formats=("manifest.json", "poster.png"))
    endpoint_bytes = config_path.read_bytes()
    expected_digest = hashlib.sha256(endpoint_bytes).hexdigest()

    monkeypatch.setattr(
        publisher,
        "BaseRenderDuplexProjection",
        lambda *_args, **_kwargs: SimpleNamespace(render_rgba=lambda *_inner: None),
    )

    def _mutating_poster(_documents, path: Path, **_kwargs) -> None:
        config_path.write_bytes(endpoint_bytes + b"\n# mutated during render\n")
        path.write_bytes(b"poster")

    monkeypatch.setattr(publisher, "render_collection_poster_png", _mutating_poster)

    output = publisher.publish_densegen_playback_endpoint(config_path)
    manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))

    assert manifest["endpoint_spec_sha256"] == expected_digest
    assert manifest["endpoint_spec_sha256"] != hashlib.sha256(config_path.read_bytes()).hexdigest()


def test_baserender_projection_omits_disabled_distance_bracket() -> None:
    record = {
        "id": "record-constraint",
        "sequence": "AAATTTCCC",
        "densegen__used_tfbs_detail": [
            {
                "part_kind": "fixed_element",
                "sequence": "AAA",
                "offset": 0,
                "offset_raw": 0,
                "end": 3,
                "orientation": "fwd",
                "constraint_name": "anchor",
                "placement_index": 0,
                "role": "upstream",
                "variant_id": "a",
                "spacer_length": 3,
            },
            {
                "part_kind": "fixed_element",
                "sequence": "CCC",
                "offset": 6,
                "offset_raw": 6,
                "end": 9,
                "orientation": "fwd",
                "constraint_name": "anchor",
                "placement_index": 0,
                "role": "downstream",
                "variant_id": "a",
                "spacer_length": 3,
            },
        ],
    }
    realized = realized_array_from_densegen_record(record, source_ref="fixture.parquet")
    plan = reconstruct_playback(realized)
    document = PlaybackDocument(
        plan=plan,
        title="Constraint fixture",
        presentation=PlaybackPresentation(show_distance_bracket="never"),
    )

    projection = BaseRenderDuplexProjection(
        (document,),
        realized_arrays={plan.realization_digest: realized},
    )

    assert all(
        effect.kind != "span_link"
        for projected_record in projection._records[plan.realization_digest]
        for effect in projected_record.effects
    )


def test_baserender_projection_keeps_unplaced_internal_coordinates_dimmed() -> None:
    record = {
        "id": "record-gap",
        "sequence": "AAATTTCCC",
        "densegen__used_tfbs_detail": [
            {
                "part_kind": "tfbs",
                "sequence": "AAA",
                "offset": 0,
                "offset_raw": 0,
                "end": 3,
                "orientation": "fwd",
                "tfbs_id": "site-1",
                "regulator": "TF_A",
            },
            {
                "part_kind": "tfbs",
                "sequence": "CCC",
                "offset": 6,
                "offset_raw": 6,
                "end": 9,
                "orientation": "fwd",
                "tfbs_id": "site-2",
                "regulator": "TF_B",
            },
        ],
    }
    realized = realized_array_from_densegen_record(record, source_ref="fixture.parquet")
    plan = reconstruct_playback(realized)
    document = PlaybackDocument(plan=plan, title="Gap fixture")

    projection = BaseRenderDuplexProjection((document,))
    final_record = projection._records[plan.realization_digest][-1]

    assert "base_hidden_indices" not in final_record.meta
    assert final_record.meta["dim_base_indices"] == {
        "primary": (3, 4, 5),
        "complement": (3, 4, 5),
    }


def test_baserender_projection_bounds_long_record_raster_memory() -> None:
    sequence = "A" * 100
    details = []
    for index, (start, end) in enumerate(((0, 33), (33, 66), (66, 100)), start=1):
        details.append(
            {
                "part_kind": "tfbs",
                "sequence": sequence[start:end],
                "offset": start,
                "offset_raw": start,
                "end": end,
                "orientation": "fwd",
                "tfbs_id": f"site-{index}",
                "regulator": f"TF_{index}",
            }
        )
    realized = realized_array_from_densegen_record(
        {
            "id": "record-long",
            "sequence": sequence,
            "densegen__used_tfbs_detail": details,
        },
        source_ref="fixture.parquet",
    )
    plan = reconstruct_playback(realized)
    document = PlaybackDocument(plan=plan, title="Long fixture")
    projection = BaseRenderDuplexProjection((document,))

    frames = tuple(projection.render_rgba(document, index) for index in range(len(plan.steps)))

    assert all(max(frame.shape[:2]) <= 2400 for frame in frames)
    assert all(frame.shape[0] * frame.shape[1] <= 3_000_000 for frame in frames)
    assert len(projection._rgba_cache) <= 2


def _frame_geometry_fixture(sequence_length: int = 60):
    sequence = "A" * sequence_length
    details = [
        {
            "part_kind": "tfbs",
            "sequence": sequence[start:end],
            "offset": start,
            "end": end,
            "orientation": "fwd",
            "tfbs_id": f"site-{index}",
            "regulator": f"TF_{index}",
        }
        for index, (start, end) in enumerate(((0, 15), (sequence_length - 6, sequence_length)))
    ]
    details.extend(
        {
            "part_kind": "fixed_element",
            "sequence": sequence[start : start + 6],
            "offset": start,
            "end": start + 6,
            "orientation": "fwd",
            "constraint_name": "anchor",
            "placement_index": 0,
            "role": role,
            "variant_id": "a",
            "spacer_length": 19,
        }
        for role, start in (("upstream", 20), ("downstream", 45))
    )
    realized = realized_array_from_densegen_record(
        {"id": f"geometry-{sequence_length}", "sequence": sequence, "densegen__used_tfbs_detail": details},
        source_ref="fixture.parquet",
    )
    plan = reconstruct_playback(realized)
    document = PlaybackDocument(
        plan=plan,
        title="Geometry fixture",
        color_overrides={plan.steps[0].placement_id: "#FF0000"},
        presentation=PlaybackPresentation(show_distance_bracket="always"),
    )
    return document, realized


@pytest.mark.parametrize("anchored", [False, True])
def test_baserender_projection_keeps_scene_crop_scale_and_alignment(anchored: bool) -> None:
    document, realized = _frame_geometry_fixture()
    projection = BaseRenderDuplexProjection(
        (document,),
        realized_arrays={document.plan.realization_digest: realized},
        presentation=DuplexPresentation(
            anchored_illustration=(AnchoredIllustrationPresentation("rnap_sigma70", "anchor") if anchored else None)
        ),
    )
    shapes, cap_heights, first_placement_boxes = [], [], []
    for index in range(len(document.plan.steps)):
        frame = projection.render_rgba(document, index)
        shapes.append(frame.shape)
        cap_heights.append(projection.native_nucleotide_cap_height_px)
        red_pixels = (frame[:, :, 0] > 220) & (frame[:, :, 1] < 80) & (frame[:, :, 2] < 80)
        rows, columns = np.where(red_pixels)
        assert rows.size > 0
        first_placement_boxes.append((rows.min(), rows.max(), columns.min(), columns.max()))

    assert len(set(shapes)) == 1
    assert len(set(cap_heights)) == 1
    assert len(set(first_placement_boxes)) == 1
    assert len(projection._rgba_cache) <= 2


def test_baserender_projection_restores_scene_native_metric_on_cache_hit() -> None:
    short_document, _ = _frame_geometry_fixture()
    long_document, _ = _frame_geometry_fixture(100)
    projection = BaseRenderDuplexProjection((short_document, long_document))
    short_frame = projection.render_rgba(short_document, 0)
    short_cap_height = projection.native_nucleotide_cap_height_px
    projection.render_rgba(long_document, 0)
    assert projection.native_nucleotide_cap_height_px != short_cap_height

    assert projection.render_rgba(short_document, 0) is short_frame
    assert projection.native_nucleotide_cap_height_px == short_cap_height
    assert len(projection._rgba_cache) == 2


@pytest.mark.parametrize("recovered", [False, True])
def test_densegen_owner_emits_only_evidenced_coordinate_recovery(recovered: bool) -> None:
    record = _publisher_row()
    detail = record["densegen__used_tfbs_detail"][0]
    if recovered:
        detail["offset"] = 1
        detail["offset_raw"] = -1
        detail["pad_left"] = 1
    plan = playback_plan_from_densegen_record(record, source_ref="fixture.parquet")
    notices = [notice for notice in plan.notices if notice.code == "coordinate_recovered"]
    assert bool(notices) is recovered
    if recovered:
        assert "1 placement" in notices[0].message
        assert "offset_raw_plus_pad" in notices[0].message
        assert notices[0].level.value == "warning"


def test_publisher_preserves_producer_coordinate_recovery_notice(tmp_path: Path) -> None:
    record = _publisher_row()
    record["densegen__used_tfbs_detail"][0]["offset"] = 1
    config_path = _write_endpoint(tmp_path, record=record)
    output = publisher.publish_densegen_playback_endpoint(config_path)
    plan = json.loads((output / "plans" / "clean_scene.json").read_text(encoding="utf-8"))
    recovered = [notice for notice in plan["notices"] if notice["code"] == "coordinate_recovered"]
    assert len(recovered) == 1
    assert "offset_raw" in recovered[0]["message"]


def test_publisher_projects_producer_labels_to_document_placement_ids(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config_path = _write_endpoint(tmp_path, formats=("manifest.json", "poster.png"))
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    config["labels"]["overrides"] = {"TF_A": "Custom binding site"}
    config["presentation"]["colors_by_label"] = {"TF_A": "#123456"}
    config["presentation"]["show_legend"] = True
    config["presentation"]["legend_entries"] = [{"key": "binding", "label": "Custom binding site", "color": "#123456"}]
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")
    captured = []
    monkeypatch.setattr(
        publisher,
        "BaseRenderDuplexProjection",
        lambda *_args, **_kwargs: SimpleNamespace(render_rgba=lambda *_inner: None),
    )

    def render(documents, path, **kwargs):
        captured.extend(documents)
        path.write_bytes(b"poster")

    monkeypatch.setattr(publisher, "render_collection_poster_png", render)
    publisher.publish_densegen_playback_endpoint(config_path)
    document = captured[0]
    assert document.step_label(0) == "Custom binding site"
    assert document.step_color(0) == "#123456"
    assert [(entry.label, entry.color) for entry in document.presentation.legend_entries] == [
        ("Custom binding site", "#123456")
    ]
    assert set(document.color_overrides) == {document.plan.steps[0].placement_id}
    assert set(document.label_overrides) == {document.plan.steps[0].placement_id}


def test_baserender_projection_preserves_explicit_placement_colors() -> None:
    plan = playback_plan_from_densegen_record(_publisher_row(), source_ref="fixture.parquet")
    document = PlaybackDocument(
        plan=plan,
        title="Caller palette",
        color_overrides={plan.steps[0].placement_id: "#123456"},
    )
    projection = BaseRenderDuplexProjection((document,))
    _, colors = projection._record_for_step(document, 0)
    assert set(colors.values()) == {"#123456"}


@pytest.mark.parametrize("color", ["red", "#12345", False])
def test_publisher_rejects_invalid_explicit_colors_before_output(tmp_path: Path, color: object) -> None:
    config_path = _write_endpoint(tmp_path)
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    config["presentation"]["colors_by_label"] = {"TF_A": color}
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")
    with pytest.raises(ValueError, match="#RRGGBB"):
        publisher.publish_densegen_playback_endpoint(config_path)
    assert not (config_path.parent / "outputs" / "publication").exists()


def test_publisher_validates_explicit_legend_text_before_output(tmp_path: Path) -> None:
    config_path = _write_endpoint(tmp_path)
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    config["presentation"]["show_legend"] = True
    config["presentation"]["legend_entries"] = [
        {"key": "binding", "label": "Forbidden sigma factor label", "color": "#123456"}
    ]
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")
    with pytest.raises(ValueError, match="presentation text contains forbidden term"):
        publisher.publish_densegen_playback_endpoint(config_path)
    assert not (config_path.parent / "outputs" / "publication").exists()


def test_publisher_requires_explicit_entries_for_enabled_legend(tmp_path: Path) -> None:
    config_path = _write_endpoint(tmp_path)
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    config["presentation"]["show_legend"] = True
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")
    with pytest.raises(ValueError, match="show_legend requires non-empty presentation.legend_entries"):
        publisher.publish_densegen_playback_endpoint(config_path)
    assert not (config_path.parent / "outputs" / "publication").exists()
