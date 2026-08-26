"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/densegen/tests/integrations/test_dense_arrays_playback.py

Verify DenseGen-to-dense-arrays playback translation.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import yaml
from dense_arrays.playback import PlaybackDocument, reconstruct_playback
from dense_arrays.playback.theme import PlaybackPresentation
from dense_arrays.realized import Orientation, PlacementKind

from dnadesign.densegen.src.integrations.dense_arrays import publisher
from dnadesign.densegen.src.integrations.dense_arrays.baserender_projection import (
    BaseRenderDuplexProjection,
)
from dnadesign.densegen.src.integrations.dense_arrays.playback import (
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
        "schema": "densegen.solution_path_playback_endpoint.v1",
        "endpoint_id": "fixture",
        "title": "Fixture endpoint",
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
    plan = reconstruct_playback(realized)
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


def test_publisher_removes_prior_bundle_after_successful_replacement(tmp_path: Path) -> None:
    config_path = _write_endpoint(tmp_path)
    output_path = publisher.publish_densegen_playback_endpoint(config_path)
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
    payload["endpoint_id"] = "private sigma factor endpoint"
    config_path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")

    with pytest.raises(ValueError, match="serialized manifest payload contains forbidden term: 'sigma factor'"):
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
    assert manifest["requested_formats"] == ["manifest.json", "poster.png"]


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


def test_baserender_projection_keeps_unplaced_internal_coordinates_hidden() -> None:
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

    assert final_record.meta["base_hidden_indices"] == {
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
