"""
--------------------------------------------------------------------------------
<dnadesign project>
src/dnadesign/baserender/tests/test_runtime_and_public_api.py

Tests for explicit runtime bootstrap and stable public API helpers.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import shutil
from pathlib import Path

import matplotlib.pyplot as plt
import pytest

import dnadesign.baserender as baserender
from dnadesign.baserender import load_records_from_parquet, render_parquet_record_figure
from dnadesign.baserender.src.core import ContractError, RenderingError
from dnadesign.baserender.src.core.record import Display, Feature
from dnadesign.baserender.src.core.registry import (
    clear_feature_effect_contracts,
    get_effect_contract,
    get_feature_contract,
)
from dnadesign.baserender.src.core.types import Span
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
                    {"tf": "lexA", "orientation": "fwd", "tfbs": "TTGACA", "offset": 0},
                    {"tf": "cpxR", "orientation": "fwd", "tfbs": "TATAAT", "offset": 23},
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


def test_public_batch_parquet_record_loader_returns_requested_order(tmp_path) -> None:
    parquet = write_parquet(
        tmp_path / "input.parquet",
        [
            {
                "id": "r1",
                "sequence": "TTGACAAAAAAAAAAAAAAAATATAAT",
                "densegen__used_tfbs_detail": [
                    {"tf": "lexA", "orientation": "fwd", "tfbs": "TTGACA", "offset": 0},
                ],
            },
            {
                "id": "r2",
                "sequence": "TTGACAAAAAAAAAAAAAAAATATAAT",
                "densegen__used_tfbs_detail": [
                    {"tf": "cpxR", "orientation": "fwd", "tfbs": "TATAAT", "offset": 23},
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
                    {"tf": "lexA", "orientation": "fwd", "tfbs": "TTGACA", "offset": 0},
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


def test_public_api_does_not_export_tool_specific_helpers() -> None:
    assert not hasattr(baserender, "render_densegen_record_figure")


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
                    {"tf": "lexA", "orientation": "fwd", "tfbs": "TTGACA", "offset": 0},
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
    assert hasattr(baserender, "validate_sequence_rows_job")
    assert hasattr(baserender, "run_sequence_rows_job")
    assert hasattr(baserender, "render")

    validated = baserender.validate_job(job_path, caller_root=tmp_path)
    report = baserender.run_job(job_path, caller_root=tmp_path)
    assert validated.version == 3
    assert "images_dir" in report.outputs


def test_public_api_accepts_in_memory_job_mapping(tmp_path) -> None:
    parquet = write_parquet(
        tmp_path / "input.parquet",
        [
            {
                "id": "r1",
                "sequence": "TTGACAAAAAAAAAAAAAAAATATAAT",
                "densegen__used_tfbs_detail": [
                    {"tf": "lexA", "orientation": "fwd", "tfbs": "TTGACA", "offset": 0},
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

    monkeypatch.setattr("dnadesign.baserender.src.api.render_record_grid_figure", _fake_grid)
    fig = baserender.render(records)
    assert seen["ncols"] == 3
    plt.close(fig)


def test_public_api_rejects_unknown_kind() -> None:
    with pytest.raises(baserender.SchemaError, match="kind must be one of"):
        baserender.validate_job("densegen_job", kind="v4")


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
