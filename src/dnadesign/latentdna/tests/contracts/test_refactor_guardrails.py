"""Guardrails for the destructive promoter-study refactor."""

from __future__ import annotations

from pathlib import Path


def _repo_root() -> Path:
    current = Path(__file__).resolve()
    for parent in current.parents:
        if (parent / "pyproject.toml").exists():
            return parent
    raise RuntimeError("repo root not found")


_TEXT_SUFFIXES = {".md", ".py", ".yaml", ".yml", ".json", ".toml", ".txt", ".svg", ".html"}
_WORKSPACE_OUTPUT_ROOT = "src/dnadesign/latentdna/workspaces/stress_ethanol_cipro_growth/outputs"


def _is_scan_text_file(path: Path) -> bool:
    return "outputs" not in path.parts and path.suffix.lower() in _TEXT_SUFFIXES


def _scan_files() -> list[Path]:
    repo_root = _repo_root()
    scan_roots = [
        repo_root / "src/dnadesign/latentdna",
        repo_root / "src/dnadesign/studies",
        repo_root / "docs/studies/stress_ethanol_cipro_growth",
    ]
    this_file = Path(__file__).resolve()
    return [
        path
        for root in scan_roots
        for path in root.rglob("*")
        if path.is_file() and path.resolve() != this_file and _is_scan_text_file(path)
    ]


def _scan_output_paths() -> list[str]:
    repo_root = _repo_root()
    output_root = repo_root / _WORKSPACE_OUTPUT_ROOT
    if not output_root.exists():
        return []
    return sorted(path.relative_to(repo_root).as_posix() for path in output_root.rglob("*"))


def test_removed_promoter_surface_ids_do_not_reappear() -> None:
    checked_files = _scan_files()
    output_paths = _scan_output_paths()
    forbidden_tokens = [
        "".join(["atlas", "_2x2_intermediate_main"]),
        "".join(["atlas", "_2x3_model_family"]),
        "".join(["geometry", "_switchboard_20b"]),
        "".join(["context_shift", "_vs_drag_primary"]),
        "".join(["x2", "_primary_20b"]),
        "".join(["delta", "20"]),
        "".join(["delta", "7"]),
        "".join(["drag", "20"]),
        "".join(["drag", "7"]),
        "".join(["z20", "_1k_anchor"]),
        "".join(["z7", "_1k_anchor"]),
        "".join(["logits20", "_1k_anchor"]),
        "".join(["logits7", "_1k_anchor"]),
        "".join(["outputs", "/latentdna"]),
        "".join(["landmark", "_atlas_committee"]),
    ]

    for forbidden in forbidden_tokens:
        hits = [
            path.as_posix() for path in checked_files if forbidden in path.read_text(encoding="utf-8", errors="ignore")
        ]
        hits.extend(path for path in output_paths if forbidden in path)
        assert hits == [], f"forbidden refactor residue {forbidden!r} still present in: {hits}"


def test_human_audience_flags_are_removed() -> None:
    checked_files = _scan_files()
    forbidden_tokens = [
        "".join(["--", "human"]),
        "".join(["human", "_mode"]),
        "".join(["human", "_readable"]),
    ]

    for forbidden in forbidden_tokens:
        hits = [
            path.as_posix() for path in checked_files if forbidden in path.read_text(encoding="utf-8", errors="ignore")
        ]
        assert hits == [], f"forbidden audience wording {forbidden!r} still present in: {hits}"


def test_generic_runtime_modules_do_not_branch_on_sigma35_semantics() -> None:
    repo_root = _repo_root()
    runtime_paths = [
        repo_root / "src/dnadesign/latentdna/src/notebooks/browser_runtime_projection.py",
        repo_root / "src/dnadesign/latentdna/src/notebooks/browser_runtime_plot_review.py",
        repo_root / "src/dnadesign/latentdna/src/notebooks/browser_runtime_support.py",
        repo_root / "src/dnadesign/latentdna/src/presentation/labels.py",
        repo_root / "src/dnadesign/latentdna/src/presentation/visual_style.py",
        repo_root / "src/dnadesign/latentdna/src/plots/render.py",
        repo_root / "src/dnadesign/latentdna/src/scalars/build.py",
        repo_root / "src/dnadesign/latentdna/src/scalars/common.py",
        repo_root / "src/dnadesign/latentdna/src/scalars/preassay.py",
        repo_root / "src/dnadesign/latentdna/src/scalars/preassay_common.py",
        repo_root / "src/dnadesign/latentdna/src/scalars/preassay_ordinal.py",
        repo_root / "src/dnadesign/latentdna/src/scalars/preassay_reference.py",
        repo_root / "src/dnadesign/latentdna/src/scalars/preassay_selection.py",
        repo_root / "src/dnadesign/latentdna/src/scalars/preassay_summary.py",
    ]
    forbidden_tokens = ["sig35", "sigma35", "sigma-35", "promoter-specific"]

    hits: dict[str, list[str]] = {}
    for path in runtime_paths:
        text = path.read_text(encoding="utf-8", errors="ignore").casefold()
        path_hits = [token for token in forbidden_tokens if token in text]
        if path_hits:
            hits[path.relative_to(repo_root).as_posix()] = path_hits

    assert hits == {}, f"generic runtime modules still carry Sigma-35 runtime coupling: {hits}"
