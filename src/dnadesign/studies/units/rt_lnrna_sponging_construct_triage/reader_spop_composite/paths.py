"""
Reader SPOP composite path and provenance helpers.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

DEFAULT_HAIRPIN_OUTPUT_DIR = Path("docs/studies/retron_hairpin_design/workbench/outputs/teto_pwm_trim_rescue_v1")
DEFAULT_OUTPUT_DIR = Path(
    "docs/studies/rt_lnrna_sponging_construct_triage/workbench/outputs/reader_spop_condition_structure_matrix_v1"
)


def resolve_repo_root(repo_root: Path | None) -> Path:
    if repo_root is not None:
        return Path(repo_root).expanduser().resolve()
    current = Path(__file__).resolve()
    for parent in current.parents:
        if (parent / "pyproject.toml").exists():
            return parent
    raise RuntimeError("repo root not found")


def relative_path(path: Path, root: Path) -> str:
    return Path(path).resolve().relative_to(Path(root).resolve()).as_posix()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return f"sha256:{digest.hexdigest()}"
