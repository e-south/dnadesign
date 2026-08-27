"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/devtools/tests/docs/test_external_study_boundaries.py

Documentation guards for external study-owned operator surfaces.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import re
from pathlib import Path


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[5]


def test_bu_scc_job_index_only_references_existing_templates() -> None:
    root = _repo_root()
    jobs_dir = root / "docs" / "bu-scc" / "jobs"
    content = (jobs_dir / "README.md").read_text(encoding="utf-8")
    references = set(re.findall(r"`([^`\n]*\.qsub)`", content))

    missing = []
    for reference in sorted(references):
        candidate = root / reference if "/" in reference else jobs_dir / reference
        if not candidate.is_file():
            missing.append(reference)

    assert missing == []


def test_public_ops_docs_do_not_advertise_external_study_registry_ids() -> None:
    root = _repo_root()
    docs = [
        *(root / "docs" / "operations").rglob("*.md"),
        root / "src" / "dnadesign" / "infer" / "docs" / "operations" / "scc-evo2-gpu-uv-runbook.md",
    ]
    external_registry_prefixes = (
        "studies.stress-ethanol-cipro-growth",
        "studies.retron-hairpin-design",
    )
    violations = [
        f"{path.relative_to(root)}: {prefix}"
        for path in docs
        for prefix in external_registry_prefixes
        if prefix in path.read_text(encoding="utf-8")
    ]

    assert violations == []


def test_generic_test_fixtures_do_not_reuse_external_study_identity() -> None:
    root = _repo_root()
    paths = [
        *(root / "src" / "dnadesign" / "ops" / "tests").rglob("*.py"),
        *(root / "src" / "dnadesign" / "devtools" / "tests" / "docs").rglob("*.py"),
    ]
    forbidden = ("stress-ethanol-cipro-growth", "study_stress_ethanol_cipro")
    violations = [
        f"{path.relative_to(root)}: {token}"
        for path in paths
        if path != Path(__file__)
        for token in forbidden
        if token in path.read_text(encoding="utf-8")
    ]

    assert violations == []
