"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/studies/units/eco1_rt_repack/tests/materialization/review_deliverables/test_chimerax_rendering.py

Eco1 review-deliverable ChimeraX rendering tests.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

from pytest import MonkeyPatch

from dnadesign.studies.units.eco1_rt_repack.operations.materialization.review_deliverables import (
    mask_tracks,
    materialize_review_deliverables,
)
from dnadesign.studies.units.eco1_rt_repack.tests.materialization.review_deliverables.fixtures import (
    write_deliverable_inputs,
)


def test_chimerax_render_uses_gui_backed_script_mode(tmp_path: Path, monkeypatch: MonkeyPatch) -> None:
    script_path = tmp_path / "mask_structure_context.cxc"
    script_path.write_text("exit\n", encoding="utf-8")
    recorded_args: list[str] = []

    def fake_run(args: list[str], **_kwargs: object) -> SimpleNamespace:
        recorded_args.extend(args)
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(mask_tracks.subprocess, "run", fake_run)

    assert mask_tracks._run_chimerax(
        executable="/Applications/ChimeraX.app/Contents/MacOS/ChimeraX",
        script_path=script_path,
    )
    assert recorded_args == [
        "/Applications/ChimeraX.app/Contents/MacOS/ChimeraX",
        "--script",
        str(script_path),
    ]
    assert "--nogui" not in recorded_args


def test_review_deliverables_default_does_not_discover_or_launch_chimerax(
    tmp_path: Path,
    monkeypatch: MonkeyPatch,
) -> None:
    write_deliverable_inputs(tmp_path)

    def fail_if_called(*_args: object, **_kwargs: object) -> str:
        raise AssertionError("default review-deliverables materialization must not touch ChimeraX")

    monkeypatch.setattr(mask_tracks, "_find_chimerax", fail_if_called)
    monkeypatch.setattr(mask_tracks, "_run_chimerax", fail_if_called)

    result = materialize_review_deliverables(repo_root=Path.cwd(), output_root=tmp_path)

    assert result.manifest_path.exists()
