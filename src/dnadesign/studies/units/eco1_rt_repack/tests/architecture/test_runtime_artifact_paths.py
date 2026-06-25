"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/studies/units/eco1_rt_repack/tests/architecture/test_runtime_artifact_paths.py

Architecture tests for Eco1 RT repack runtime artifact path ownership.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from dnadesign.studies.units.eco1_rt_repack.paths import DEFAULT_THREAD_OUTPUT_ROOT
from dnadesign.studies.units.eco1_rt_repack.tests._helpers import repo_root

_PACKAGE_ROOT = "src/dnadesign/studies/units/eco1_rt_repack"
_STUDY_OUTPUT_ROOT = str(DEFAULT_THREAD_OUTPUT_ROOT)


def test_runtime_artifact_paths_are_study_workspace_scoped() -> None:
    root = repo_root()
    forbidden_repo_root_output_paths = {
        "/".join(("outputs", "thread", "eco1_rt_conservative_v1")),
        "#$ -o " + "outputs/logs",
        "tail -f " + "outputs/logs",
    }
    scanned_roots = (
        root / _PACKAGE_ROOT,
        root / "docs/studies/eco1_rt_repack",
        root / "docs/bu-scc/jobs/eco1-colabfold-foldcheck.qsub",
        root / ".agents/skills/eco1-rt-repack-status",
    )

    offenders = []
    for scanned_root in scanned_roots:
        paths = [scanned_root] if scanned_root.is_file() else list(scanned_root.rglob("*"))
        for path in paths:
            if not path.is_file() or path.suffix not in {".py", ".md", ".yaml", ".yml", ".qsub"}:
                continue
            if "workspaces" in path.relative_to(root).parts:
                continue
            text = path.read_text(encoding="utf-8")
            if any(forbidden in text for forbidden in forbidden_repo_root_output_paths):
                offenders.append(str(path.relative_to(root)))

    assert offenders == []
    assert str(DEFAULT_THREAD_OUTPUT_ROOT) == _STUDY_OUTPUT_ROOT
    assert "DEFAULT_THREAD_OUTPUT_ROOT" in (root / _PACKAGE_ROOT / "operations/contracts/constants.py").read_text(
        encoding="utf-8"
    )
