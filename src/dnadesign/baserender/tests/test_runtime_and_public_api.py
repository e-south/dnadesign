"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/baserender/tests/test_runtime_and_public_api.py

Tests for explicit runtime bootstrap and stable public API helpers.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import shutil
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pytest

import dnadesign.baserender as baserender
from dnadesign.baserender import load_records_from_parquet, render_parquet_record_figure
from dnadesign.baserender.src.core import ContractError, RenderingError
from dnadesign.baserender.src.core.record import Display, Feature, Record
from dnadesign.baserender.src.core.registry import (
    clear_feature_effect_contracts,
    get_effect_contract,
    get_feature_contract,
)
from dnadesign.baserender.src.core.types import Span
from dnadesign.baserender.src.outputs.images import _grid_max_rows_for_records, _grid_ncols_for_records
from dnadesign.baserender.src.render.effects.registry import clear_effect_drawers, get_effect_drawer
from dnadesign.baserender.src.runtime import initialize_runtime

from .conftest import densegen_job_payload, write_job, write_parquet


def test_runtime_bootstrap_is_explicit_and_idempotent() -> None:
    clear_feature_effect_contracts()
    clear_effect_drawers()
    import dnadesign.baserender.src.render as _render  # noqa: F401

    with pytest.raises(ContractError, match="Unknown feature kind"):
        get_feature_contract("kmer")
    with pytest.raises(ContractError, match="Unknown effect kind"):
        get_effect_contract("span_link")
    with pytest.raises(RenderingError, match="Unknown effect kind"):
        get_effect_drawer("span_link")

    initialize_runtime()
    assert get_feature_contract("kmer")
    assert get_effect_contract("span_link")
    assert get_effect_drawer("span_link")

    initialize_runtime()
    assert get_feature_contract("kmer")
    assert get_effect_contract("motif_logo")


def test_public_parquet_render_helper_renders_record_figure(tmp_path) -> None:
    parquet = write_parquet(
        tmp_path / "input.parquet",
        [
            {
                "id": "r1",
                "sequence": "TTGACAAAAAAAAAAAAAAAATATAAT",
                "densegen__used_tfbs_detail": [
                    {"regulator": "lexA", "orientation": "fwd", "sequence": "TTGACA", "offset": 0},
                    {"regulator": "cpxR", "orientation": "fwd", "sequence": "TATAAT", "offset": 23},
                ],
                "details": "row1",
            }
        ],
    )

    fig = render_parquet_record_figure(
        dataset_path=parquet,
        record_id="r1",
        adapter_kind="densegen_tfbs",
        adapter_columns={
            "sequence": "sequence",
            "annotations": "densegen__used_tfbs_detail",
            "id": "id",
            "overlay_text": "details",
        },
    )
    assert fig is not None
    plt.close(fig)


def test_public_sequence_panel_contract_renders_image() -> None:
    row = {
        "id": "r1",
        "sequence": "TTGACAAAAAAAAAAAAAAAATATAAT",
        "densegen__used_tfbs_detail": [
            {"regulator": "lexA", "orientation": "fwd", "sequence": "TTGACA", "offset": 0},
            {"regulator": "cpxR", "orientation": "fwd", "sequence": "TATAAT", "offset": 23},
        ],
    }

    result = baserender.render_sequence_panel_image(
        row,
        adapter_kind="densegen_tfbs",
        target_width_px=420,
        target_height_px=140,
    )

    assert baserender.BASERENDER_SEQUENCE_PANEL_CONTRACT_VERSION == "1"
    assert result.image.shape == (140, 420, 4)
    assert result.diagnostics.contract_id == "dnadesign.baserender.sequence_panel.v1"
    assert result.diagnostics.style_profile == "promoter_compact_slide.v1"
    assert result.diagnostics.strand_count == 2


def test_public_sequence_panel_contract_uses_white_canvas_under_dark_rc() -> None:
    row = {
        "id": "r1",
        "sequence": "TTGACAAAAAAAAAAAAAAAATATAAT",
        "densegen__used_tfbs_detail": [
            {"regulator": "lexA", "orientation": "fwd", "sequence": "TTGACA", "offset": 0},
            {"regulator": "cpxR", "orientation": "fwd", "sequence": "TATAAT", "offset": 23},
        ],
    }

    with plt.rc_context(
        {
            "figure.facecolor": "black",
            "axes.facecolor": "black",
            "savefig.facecolor": "black",
        }
    ):
        result = baserender.render_sequence_panel_image(
            row,
            adapter_kind="densegen_tfbs",
            target_width_px=420,
            target_height_px=140,
        )

    rgb = np.asarray(result.image)[:, :, :3]
    near_black_fraction = float((rgb.max(axis=2) <= 24).mean())
    assert near_black_fraction < 0.01


def test_public_sequence_panel_contract_renders_empty_densegen_annotations() -> None:
    row = {
        "id": "r1",
        "sequence": "TTGACAAAAAAAAAAAAAAAATATAAT",
        "densegen__used_tfbs_detail": [],
    }

    result = baserender.render_sequence_panel_image(
        row,
        adapter_kind="densegen_tfbs",
        target_width_px=420,
        target_height_px=140,
    )

    assert result.image.shape == (140, 420, 4)
    assert result.diagnostics.feature_count == 0


def test_public_sequence_panel_contract_renders_usr_genbank_image() -> None:
    row = {
        "id": "seq1",
        "sequence": "AACCGGTTGACATTTTTTTTTATAATGGCC",
        "usr_label__primary": "demoP",
        "seq_annot__features": [
            {
                "feature_id": "feat_promoter",
                "feature_order": 1,
                "feature_type": "misc_feature",
                "label": "pred. demoP",
                "role_hint": None,
                "start_0": 2,
                "end_0": 28,
                "strand": 1,
                "confidence": "high",
            },
            {
                "feature_id": "feat_m35",
                "feature_order": 2,
                "feature_type": "misc_feature",
                "label": "-35",
                "role_hint": "sigma70_minus35",
                "start_0": 6,
                "end_0": 12,
                "strand": 1,
                "confidence": "high",
            },
            {
                "feature_id": "feat_tfbs",
                "feature_order": 3,
                "feature_type": "misc_feature",
                "label": "LexA-",
                "role_hint": "TFBS",
                "start_0": 10,
                "end_0": 18,
                "strand": -1,
                "confidence": "high",
            },
            {
                "feature_id": "feat_m10",
                "feature_order": 4,
                "feature_type": "misc_feature",
                "label": "-10",
                "role_hint": "sigma70_minus10",
                "start_0": 20,
                "end_0": 26,
                "strand": 1,
                "confidence": "high",
            },
        ],
    }

    result = baserender.render_sequence_panel_image(
        row,
        adapter_kind="usr_genbank_annotations_v1",
        target_width_px=420,
        target_height_px=140,
    )

    assert result.image.shape == (140, 420, 4)
    assert result.diagnostics.contract_id == "dnadesign.baserender.sequence_panel.v1"
    assert result.diagnostics.adapter_kind == "usr_genbank_annotations_v1"
    assert result.diagnostics.feature_count == 4
    assert result.diagnostics.strand_count == 2


def test_public_sequence_panel_contract_rejects_invalid_profile_and_adapter() -> None:
    with pytest.raises(baserender.SchemaError, match="Unknown sequence panel profile"):
        baserender.sequence_panel_config_for_adapter("densegen_tfbs", style_profile="missing_profile")

    with pytest.raises(baserender.SchemaError, match="Unsupported sequence panel adapter kind"):
        baserender.sequence_panel_config_for_adapter("missing_adapter")


def test_public_style_helpers_are_root_exports() -> None:
    assert "presentation_default" in baserender.list_style_presets()
    style = baserender.resolve_style(preset=None, overrides={})
    assert isinstance(style, baserender.Style)


def test_image_grid_uses_record_max_rows_hint() -> None:
    records = [Record(id=f"r{index}", alphabet="DNA", sequence="ACGT", meta={"grid_max_rows": 5}) for index in range(8)]

    assert _grid_max_rows_for_records(records) == 5
    assert _grid_ncols_for_records(records[:5], default_ncols=1) == 1
    assert _grid_ncols_for_records(records, default_ncols=1) == 2


def test_public_batch_parquet_record_loader_returns_requested_order(tmp_path) -> None:
    parquet = write_parquet(
        tmp_path / "input.parquet",
        [
            {
                "id": "r1",
                "sequence": "TTGACAAAAAAAAAAAAAAAATATAAT",
                "densegen__used_tfbs_detail": [
                    {"regulator": "lexA", "orientation": "fwd", "sequence": "TTGACA", "offset": 0},
                ],
            },
            {
                "id": "r2",
                "sequence": "TTGACAAAAAAAAAAAAAAAATATAAT",
                "densegen__used_tfbs_detail": [
                    {"regulator": "cpxR", "orientation": "fwd", "sequence": "TATAAT", "offset": 23},
                ],
            },
        ],
    )
    records = load_records_from_parquet(
        dataset_path=parquet,
        record_ids=["r2", "r1"],
        adapter_kind="densegen_tfbs",
        adapter_columns={
            "sequence": "sequence",
            "annotations": "densegen__used_tfbs_detail",
            "id": "id",
        },
    )
    assert [record.id for record in records] == ["r2", "r1"]


def test_public_batch_parquet_record_loader_raises_on_missing_record_ids(tmp_path) -> None:
    parquet = write_parquet(
        tmp_path / "input.parquet",
        [
            {
                "id": "r1",
                "sequence": "TTGACAAAAAAAAAAAAAAAATATAAT",
                "densegen__used_tfbs_detail": [
                    {"regulator": "lexA", "orientation": "fwd", "sequence": "TTGACA", "offset": 0},
                ],
            }
        ],
    )
    with pytest.raises(baserender.SchemaError, match="Records not found"):
        load_records_from_parquet(
            dataset_path=parquet,
            record_ids=["r1", "missing"],
            adapter_kind="densegen_tfbs",
            adapter_columns={
                "sequence": "sequence",
                "annotations": "densegen__used_tfbs_detail",
                "id": "id",
            },
        )


def test_public_adapter_helpers_adapt_in_memory_contract_rows() -> None:
    row = {
        "contract_kind": "sequence_evidence_map_v1",
        "state_id": "assembled_payload",
        "topology_kind": "linear_dsdna",
        "alphabet": "iupac_dna",
        "primary_sequence": "CTCTATATCTGATATAGAG",
        "complement_sequence": "GAGATATAGTGTATATCTC",
        "owners": [],
        "effect_tags": [],
        "boundaries": [],
        "pairings": [],
        "display": {"title": "Assembled payload"},
        "meta": {},
    }

    record = baserender.adapt_record(row, adapter_kind="sequence_evidence_map_v1", alphabet="IUPAC_DNA")
    records = baserender.adapt_records([row, row], adapter_kind="sequence_evidence_map_v1", alphabet="IUPAC_DNA")

    assert record.id == "assembled_payload"
    assert [item.id for item in records] == ["assembled_payload", "assembled_payload"]


def test_public_sequence_evidence_adapter_rejects_unequal_complement_length() -> None:
    row = {
        "contract_kind": "sequence_evidence_map_v1",
        "state_id": "assembled_payload",
        "topology_kind": "linear_dsdna",
        "alphabet": "iupac_dna",
        "primary_sequence": "ACGT",
        "complement_sequence": "TGCAT",
        "owners": [],
        "effect_tags": [],
        "boundaries": [],
        "pairings": [],
        "display": {"title": "Assembled payload"},
        "meta": {},
    }

    with pytest.raises(baserender.SchemaError, match="complement_sequence length must match primary_sequence"):
        baserender.adapt_record(row, adapter_kind="sequence_evidence_map_v1", alphabet="IUPAC_DNA")


def test_public_adapter_helpers_reject_unknown_adapter_columns() -> None:
    row = {"sequence": "ACGT", "features": []}

    with pytest.raises(baserender.SchemaError, match="Unknown keys in input.adapter.columns"):
        baserender.adapt_record(
            row,
            adapter_kind="generic_features",
            adapter_columns={"sequence": "sequence", "features": "features", "bogus": "ignored"},
        )


def test_public_adapter_helpers_reject_unknown_adapter_policies() -> None:
    row = {"sequence": "ACGT", "features": []}

    with pytest.raises(baserender.SchemaError, match="Unknown keys in input.adapter.policies"):
        baserender.adapt_record(
            row,
            adapter_kind="generic_features",
            adapter_columns={"sequence": "sequence", "features": "features"},
            adapter_policies={"typo_policy": True},
        )


def test_public_adapter_helpers_enforce_adapter_alphabet_compatibility() -> None:
    row = {"sequence": "ACGU", "features": []}

    with pytest.raises(baserender.SchemaError, match="input.adapter.kind.*input.alphabet"):
        baserender.adapt_record(
            row,
            adapter_kind="generic_features",
            adapter_columns={"sequence": "sequence", "features": "features"},
            alphabet="RNA",
        )


def test_public_parquet_render_helper_rejects_legacy_densegen_tfbs_keys(tmp_path) -> None:
    parquet = write_parquet(
        tmp_path / "input.parquet",
        [
            {
                "id": "r1",
                "sequence": "TTGACAAAAAAAAAAAAAAAATATAAT",
                "densegen__used_tfbs_detail": [
                    {"tf": "lexA", "orientation": "fwd", "tfbs": "TTGACA", "offset": 0},
                ],
            }
        ],
    )

    with pytest.raises(baserender.SchemaError, match="regulator"):
        render_parquet_record_figure(
            dataset_path=parquet,
            record_id="r1",
            adapter_kind="densegen_tfbs",
            adapter_columns={
                "sequence": "sequence",
                "annotations": "densegen__used_tfbs_detail",
                "id": "id",
            },
            adapter_policies={"on_invalid_row": "error"},
        )


def test_public_api_does_not_export_tool_specific_helpers() -> None:
    assert not hasattr(baserender, "render_densegen_record_figure")


def test_public_api_exports_palette_for_sibling_tools() -> None:
    palette = baserender.Palette({})
    assert palette is not None


def test_public_record_grid_render_helper_renders_multi_panel_figure() -> None:
    from dnadesign.baserender import Record, render_record_grid_figure

    records = (
        Record(
            id="r1",
            alphabet="DNA",
            sequence="TTGACAAAAAAAAAAAAAAAATATAAT",
            features=(
                Feature(
                    id="f1",
                    kind="kmer",
                    span=Span(start=0, end=6, strand="fwd"),
                    label="TTGACA",
                    tags=("tf:lexA",),
                    render={"priority": 10},
                ),
            ),
            display=Display(overlay_text="elite-1", tag_labels={"tf:lexA": "lexA"}),
        ),
        Record(
            id="r2",
            alphabet="DNA",
            sequence="TTGACAAAAAAAAAAAAAAAATATAAT",
            features=(
                Feature(
                    id="f1",
                    kind="kmer",
                    span=Span(start=21, end=27, strand="fwd"),
                    label="TATAAT",
                    tags=("tf:cpxR",),
                    render={"priority": 10},
                ),
            ),
            display=Display(overlay_text="elite-2", tag_labels={"tf:cpxR": "cpxR"}),
        ),
    )

    fig = render_record_grid_figure(records, ncols=2)
    assert fig is not None
    plt.close(fig)


def test_public_api_exposes_generic_job_entrypoints(tmp_path) -> None:
    parquet = write_parquet(
        tmp_path / "input.parquet",
        [
            {
                "id": "r1",
                "sequence": "TTGACAAAAAAAAAAAAAAAATATAAT",
                "densegen__used_tfbs_detail": [
                    {"regulator": "lexA", "orientation": "fwd", "sequence": "TTGACA", "offset": 0},
                ],
                "details": "",
            }
        ],
    )
    payload = densegen_job_payload(
        parquet_path=parquet,
        results_root=tmp_path / "outputs",
        outputs=[{"kind": "images", "fmt": "png"}],
    )
    job_path = write_job(tmp_path / "job.yaml", payload)

    assert hasattr(baserender, "validate_job")
    assert hasattr(baserender, "run_job")
    assert hasattr(baserender, "RenderJobV3")
    assert hasattr(baserender, "validate_sequence_rows_job")
    assert hasattr(baserender, "run_sequence_rows_job")
    assert hasattr(baserender, "adapt_record")
    assert hasattr(baserender, "adapt_records")
    assert hasattr(baserender, "list_adapters")
    assert hasattr(baserender, "list_renderers")
    assert hasattr(baserender, "list_render_contracts")
    assert hasattr(baserender, "get_adapter_descriptor")
    assert hasattr(baserender, "get_renderer_descriptor")
    assert hasattr(baserender, "get_render_contract_descriptor")
    assert hasattr(baserender, "render")

    validated = baserender.validate_job(job_path, caller_root=tmp_path)
    report = baserender.run_job(job_path, caller_root=tmp_path)
    assert validated.version == 3
    assert isinstance(validated, baserender.RenderJobV3)
    assert "images_dir" in report.outputs

    adapter_kinds = baserender.list_adapters()
    renderer_names = baserender.list_renderers()
    assert "yiu_topology_cartoon_v1" in adapter_kinds
    assert "sequence_evidence_map_v1" in adapter_kinds
    assert "topology_cartoon" in renderer_names
    assert "nucleotide_evidence_map" in renderer_names
    assert "snapback_map" in renderer_names
    adapter_descriptor = baserender.get_adapter_descriptor("yiu_topology_cartoon_v1")
    renderer_descriptor = baserender.get_renderer_descriptor("topology_cartoon")
    assert adapter_descriptor.owner_tool == "yiu"
    assert "topology_cartoon" in adapter_descriptor.supported_renderers
    assert renderer_descriptor.name == "topology_cartoon"
    assert "DNA" in renderer_descriptor.accepted_alphabets
    evidence_adapter_descriptor = baserender.get_adapter_descriptor("sequence_evidence_map_v1")
    evidence_renderer_descriptor = baserender.get_renderer_descriptor("nucleotide_evidence_map")
    snapback_adapter_descriptor = baserender.get_adapter_descriptor("snapback_visual_v1")
    snapback_renderer_descriptor = baserender.get_renderer_descriptor("snapback_map")
    assert evidence_adapter_descriptor.owner_tool is None
    assert evidence_adapter_descriptor.required_source_columns == ()
    assert evidence_adapter_descriptor.supported_renderers == ("nucleotide_evidence_map",)
    assert evidence_renderer_descriptor.name == "nucleotide_evidence_map"
    assert "span_link" in evidence_renderer_descriptor.optional_record_features
    assert snapback_adapter_descriptor.supported_renderers == ("snapback_map",)
    assert snapback_renderer_descriptor.name == "snapback_map"

    contract_kinds = baserender.list_render_contracts()
    assert "base_render_job_v3" in contract_kinds
    assert "sequence_rows_render_v3" in contract_kinds
    assert "usr_genbank_annotation_render_v1" in contract_kinds
    assert "nucleotide_evidence_map_render_v3" in contract_kinds
    sequence_contract = baserender.get_render_contract_descriptor("sequence_rows_v3")
    generic_contract = baserender.get_render_contract_descriptor("render_job_v3")
    assert sequence_contract.kind == "sequence_rows_render_v3"
    assert sequence_contract.accepted_renderers == ("sequence_rows",)
    assert generic_contract.kind == "base_render_job_v3"
    assert "nucleotide_evidence_map" in generic_contract.accepted_renderers


def test_public_api_rejects_unknown_renderer_lookup() -> None:
    from dnadesign.baserender.src.render.renderer import get_renderer

    with pytest.raises(RenderingError, match="Unknown renderer: missing"):
        get_renderer("missing")
    with pytest.raises(RenderingError, match="Unknown renderer: missing"):
        baserender.get_renderer_descriptor("missing")


def test_public_api_accepts_in_memory_job_mapping(tmp_path) -> None:
    parquet = write_parquet(
        tmp_path / "input.parquet",
        [
            {
                "id": "r1",
                "sequence": "TTGACAAAAAAAAAAAAAAAATATAAT",
                "densegen__used_tfbs_detail": [
                    {"regulator": "lexA", "orientation": "fwd", "sequence": "TTGACA", "offset": 0},
                ],
                "details": "",
            }
        ],
    )
    payload = densegen_job_payload(
        parquet_path=Path("input.parquet"),
        results_root=Path("results"),
        outputs=[{"kind": "images", "fmt": "png"}],
    )

    validated = baserender.validate_job(payload, caller_root=tmp_path)
    report = baserender.run_job(payload, caller_root=tmp_path)
    assert validated.path == (tmp_path / "inline_job.yaml").resolve()
    assert validated.input.path == parquet.resolve()
    assert "images_dir" in report.outputs


def test_public_render_accepts_preset_only_style_mapping() -> None:
    record = baserender.Record(
        id="r1",
        alphabet="DNA",
        sequence="TTGACAAAAAAAAAAAAAAAATATAAT",
        features=(
            baserender.Feature(
                id="f1",
                kind="kmer",
                span=baserender.Span(start=0, end=6, strand="fwd"),
                label="TTGACA",
                tags=("tf:lexA",),
                render={"priority": 10},
            ),
        ),
        display=baserender.Display(overlay_text="elite-1", tag_labels={"tf:lexA": "lexA"}),
    )
    fig = baserender.render(record, style={"preset": "presentation_default"})
    assert fig is not None
    plt.close(fig)


def test_public_render_rejects_unknown_grid_keys() -> None:
    records = (
        baserender.Record(
            id="r1",
            alphabet="DNA",
            sequence="TTGACAAAAAAAAAAAAAAAATATAAT",
            features=(
                baserender.Feature(
                    id="f1",
                    kind="kmer",
                    span=baserender.Span(start=0, end=6, strand="fwd"),
                    label="TTGACA",
                    tags=("tf:lexA",),
                    render={"priority": 10},
                ),
            ),
            display=baserender.Display(overlay_text="elite-1", tag_labels={"tf:lexA": "lexA"}),
        ),
    )
    with pytest.raises(baserender.SchemaError, match="grid contains unknown keys"):
        baserender.render(records, grid={"cols": 2})


def test_public_render_defaults_to_single_row_for_record_lists(monkeypatch: pytest.MonkeyPatch) -> None:
    records = (
        baserender.Record(
            id="r1",
            alphabet="DNA",
            sequence="TTGACAAAAAAAAAAAAAAAATATAAT",
            features=(),
            display=baserender.Display(overlay_text="elite-1", tag_labels={}),
        ),
        baserender.Record(
            id="r2",
            alphabet="DNA",
            sequence="TTGACAAAAAAAAAAAAAAAATATAAT",
            features=(),
            display=baserender.Display(overlay_text="elite-2", tag_labels={}),
        ),
        baserender.Record(
            id="r3",
            alphabet="DNA",
            sequence="TTGACAAAAAAAAAAAAAAAATATAAT",
            features=(),
            display=baserender.Display(overlay_text="elite-3", tag_labels={}),
        ),
    )
    seen: dict[str, int] = {}

    def _fake_grid(
        _records,
        *,
        renderer_name: str,
        style_preset,
        style_overrides,
        ncols: int,
    ):
        seen["ncols"] = int(ncols)
        assert renderer_name == "sequence_rows"
        return plt.figure(figsize=(2, 2), dpi=100)

    monkeypatch.setattr("dnadesign.baserender.src.public.api.render_record_grid_figure", _fake_grid)
    fig = baserender.render(records)
    assert seen["ncols"] == 3
    plt.close(fig)


def test_public_api_rejects_unknown_kind() -> None:
    with pytest.raises(baserender.SchemaError, match="kind must be one of"):
        baserender.validate_job("densegen_job", kind="v4")


def test_public_api_accepts_render_job_v3_kind_alias(tmp_path: Path) -> None:
    parquet = write_parquet(
        tmp_path / "input.parquet",
        [
            {
                "id": "r1",
                "sequence": "TTGACAAAAAAAAAAAAAAAATATAAT",
                "densegen__used_tfbs_detail": [
                    {"regulator": "lexA", "orientation": "fwd", "sequence": "TTGACA", "offset": 0},
                ],
                "details": "",
            }
        ],
    )
    payload = densegen_job_payload(
        parquet_path=parquet,
        results_root=tmp_path / "outputs",
        outputs=[{"kind": "images", "fmt": "png"}],
    )
    job_path = write_job(tmp_path / "job.yaml", payload)

    validated = baserender.validate_job(job_path, kind="render_job_v3", caller_root=tmp_path)
    report = baserender.run_job(job_path, kind="render_job_v3", caller_root=tmp_path)

    assert validated.version == 3
    assert "images_dir" in report.outputs


def test_public_api_kind_descriptor_rejects_incompatible_renderer(tmp_path: Path) -> None:
    json_path = tmp_path / "input.json"
    json_path.write_text("[]")
    payload = {
        "version": 3,
        "results_root": str(tmp_path / "outputs"),
        "input": {
            "kind": "json",
            "path": str(json_path),
            "adapter": {"kind": "sequence_evidence_map_v1", "columns": {}, "policies": {}},
            "alphabet": "DNA",
        },
        "render": {"renderer": "nucleotide_evidence_map", "style": {"preset": None, "overrides": {}}},
        "outputs": [{"kind": "images", "fmt": "png"}],
    }
    job_path = write_job(tmp_path / "job.yaml", payload)

    with pytest.raises(baserender.SchemaError, match="kind.*render.renderer"):
        baserender.validate_job(job_path, kind="sequence_rows_v3", caller_root=tmp_path)


def test_public_api_exposes_render_job_alias(tmp_path: Path) -> None:
    parquet = write_parquet(
        tmp_path / "input.parquet",
        [
            {
                "id": "r1",
                "sequence": "TTGACAAAAAAAAAAAAAAAATATAAT",
                "densegen__used_tfbs_detail": [
                    {"regulator": "lexA", "orientation": "fwd", "sequence": "TTGACA", "offset": 0},
                ],
                "details": "",
            }
        ],
    )
    payload = densegen_job_payload(
        parquet_path=parquet,
        results_root=tmp_path / "outputs",
        outputs=[{"kind": "images", "fmt": "png"}],
    )
    job_path = write_job(tmp_path / "job.yaml", payload)

    validated = baserender.validate_render_job(job_path, caller_root=tmp_path)
    report = baserender.run_render_job(job_path, caller_root=tmp_path)

    assert validated.version == 3
    assert "images_dir" in report.outputs


def test_public_api_runs_densegen_and_cruncher_contracts_end_to_end(tmp_path: Path) -> None:
    pkg_root = Path(__file__).resolve().parents[1]

    # Curated workspaces: copy into isolated temp root and run through public API.
    copied_root = tmp_path / "workspaces"
    copied_root.mkdir(parents=True, exist_ok=True)
    for ws_name in ("demo_densegen_render", "demo_cruncher_render"):
        src_ws = pkg_root / "workspaces" / ws_name
        dst_ws = copied_root / ws_name
        shutil.copytree(src_ws, dst_ws)
        job_path = dst_ws / "job.yaml"

        validated = baserender.validate_job(job_path, kind="sequence_rows_v3", caller_root=tmp_path)
        report = baserender.run_job(job_path, kind="sequence_rows_v3", caller_root=tmp_path)
        assert validated.version == 3
        expected_ext = next(cfg.fmt for cfg in validated.outputs if cfg.kind == "images")
        images_dir = Path(report.outputs["images_dir"])
        assert images_dir.exists()
        assert any(p.suffix.lower() == f".{expected_ext.lower()}" for p in images_dir.iterdir())

    # Contract examples: ensure source-like cruncher and densegen paths still work through stable API.
    for example in ("densegen_job.yaml", "cruncher_job.yaml"):
        job_path = pkg_root / "docs" / "examples" / example
        validated = baserender.validate_job(job_path, kind="cruncher_showcase_v3", caller_root=tmp_path)
        report = baserender.run_job(job_path, kind="cruncher_showcase_v3", caller_root=tmp_path)
        assert validated.version == 3
        expected_ext = next(cfg.fmt for cfg in validated.outputs if cfg.kind == "images")
        images_dir = Path(report.outputs["images_dir"])
        assert images_dir.exists()
        assert any(p.suffix.lower() == f".{expected_ext.lower()}" for p in images_dir.iterdir())
