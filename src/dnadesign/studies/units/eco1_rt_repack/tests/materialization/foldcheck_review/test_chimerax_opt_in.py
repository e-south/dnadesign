"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/studies/units/eco1_rt_repack/tests/materialization/foldcheck_review/test_chimerax_opt_in.py

ChimeraX opt-in rendering tests for Eco1 fold-check review.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from pathlib import Path

import yaml
from pytest import MonkeyPatch

from dnadesign.studies.units.eco1_rt_repack.operations.materialization.foldcheck_review import (
    materialize_foldcheck_review,
    structure_overlay,
)
from dnadesign.studies.units.eco1_rt_repack.tests.materialization.foldcheck_review.fixtures import (
    write_review_inputs,
)


def test_foldcheck_review_default_does_not_discover_or_launch_chimerax(
    tmp_path: Path,
    monkeypatch: MonkeyPatch,
) -> None:
    write_review_inputs(tmp_path, local_model_paths=True)

    def fail_if_called(*_args: object, **_kwargs: object) -> str:
        raise AssertionError("default foldcheck-review materialization must not touch ChimeraX")

    monkeypatch.setattr(structure_overlay, "_find_chimerax", fail_if_called)
    monkeypatch.setattr(structure_overlay, "_run_chimerax", fail_if_called)

    result = materialize_foldcheck_review(repo_root=Path.cwd(), output_root=tmp_path)

    manifest = yaml.safe_load(result.visual_manifest_path.read_text(encoding="utf-8"))
    overlay_row = next(plot for plot in manifest["plots"] if plot["plot_id"] == "structure_overlay_panel")
    assert overlay_row["status"] == "skipped_optional_render_disabled"


def test_foldcheck_review_default_removes_stale_overlay_png(tmp_path: Path) -> None:
    write_review_inputs(tmp_path, local_model_paths=True)
    first = materialize_foldcheck_review(repo_root=Path.cwd(), output_root=tmp_path)
    first_manifest = yaml.safe_load(first.visual_manifest_path.read_text(encoding="utf-8"))
    first_overlay = next(plot for plot in first_manifest["plots"] if plot["plot_id"] == "structure_overlay_panel")
    overlay_path = first.visual_manifest_path.parent / str(first_overlay["path"])
    overlay_path.parent.mkdir(parents=True, exist_ok=True)
    overlay_path.write_bytes(b"stale-chimerax-render")

    second = materialize_foldcheck_review(repo_root=Path.cwd(), output_root=tmp_path)
    second_manifest = yaml.safe_load(second.visual_manifest_path.read_text(encoding="utf-8"))
    second_overlay = next(plot for plot in second_manifest["plots"] if plot["plot_id"] == "structure_overlay_panel")

    assert second_overlay["status"] == "skipped_optional_render_disabled"
    assert not overlay_path.exists()
