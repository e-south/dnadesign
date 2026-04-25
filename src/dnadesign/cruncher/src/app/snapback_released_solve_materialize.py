"""
Per-hit materialization helpers for released-product Snapback solve runs.
"""

from __future__ import annotations

from pathlib import Path

from dnadesign.cruncher.artifacts.atomic_write import atomic_write_json
from dnadesign.cruncher.snapback.released_artifacts import (
    released_pre_nick_site_json_path,
    released_projection_json_path,
    released_release_site_json_path,
    released_solve_hit_json_path,
    released_solve_hit_plot_context_path,
    released_solve_hit_plot_path,
    released_solve_hit_run_dir,
)
from dnadesign.cruncher.snapback.released_hit_plot import (
    build_released_hit_plot_context,
    render_released_hit_plot,
)
from dnadesign.cruncher.snapback.released_models import (
    ReleasedSolveHit,
    ReleasedSolveOutputConfig,
    ReleasedTargetSearchHit,
)
from dnadesign.cruncher.viz.mpl import ensure_workspace_mpl_cache


def _relative_to_workspace(path: Path, *, workspace_root: Path) -> str:
    return str(path.resolve().relative_to(workspace_root.resolve()))


def _ensure_materialized_hit_dirs(hit_run_dir: Path) -> None:
    hit_run_dir.mkdir(parents=True, exist_ok=True)
    (hit_run_dir / "analysis").mkdir(parents=True, exist_ok=True)
    (hit_run_dir / "plots").mkdir(parents=True, exist_ok=True)


def _validate_rendered_plot_artifact(path: Path, *, fmt: str) -> None:
    if not path.exists():
        raise FileNotFoundError(f"Released solve render missing expected plot: {path}")
    header = path.read_bytes()[:64]
    if fmt == "pdf" and not header.startswith(b"%PDF"):
        raise ValueError(f"Released solve render is not a valid PDF artifact: {path}")
    if fmt == "png" and not header.startswith(b"\x89PNG\r\n\x1a\n"):
        raise ValueError(f"Released solve render is not a valid PNG artifact: {path}")
    if fmt == "svg":
        normalized = header.lstrip()
        if not (normalized.startswith(b"<?xml") or normalized.startswith(b"<svg")):
            raise ValueError(f"Released solve render is not a valid SVG artifact: {path}")


def materialize_released_solve_hit(
    *,
    hit: ReleasedTargetSearchHit,
    rank: int,
    run_dir: Path,
    workspace_root: Path,
    output: ReleasedSolveOutputConfig,
) -> ReleasedSolveHit:
    hit_run_dir = released_solve_hit_run_dir(run_dir, rank=rank)
    _ensure_materialized_hit_dirs(hit_run_dir)
    atomic_write_json(released_solve_hit_json_path(hit_run_dir), hit.model_dump(mode="json"))
    atomic_write_json(released_projection_json_path(hit_run_dir), hit.projection.model_dump(mode="json"))
    atomic_write_json(
        released_pre_nick_site_json_path(hit_run_dir),
        {
            "site": hit.pre_nick_site.model_dump(mode="json"),
            "event": hit.pre_nick_event.model_dump(mode="json"),
        },
    )
    atomic_write_json(
        released_release_site_json_path(hit_run_dir),
        {
            "site": hit.release_site.model_dump(mode="json"),
            "event": hit.release_event.model_dump(mode="json"),
        },
    )

    rendered_plot_path: Path | None = None
    if output.emit_renders:
        ensure_workspace_mpl_cache(workspace_root)
        rendered_plot_path = released_solve_hit_plot_path(hit_run_dir, fmt=output.render_format)
        plot_context = render_released_hit_plot(hit, rendered_plot_path)
        _validate_rendered_plot_artifact(rendered_plot_path, fmt=output.render_format)
    else:
        plot_context = build_released_hit_plot_context(hit)
    atomic_write_json(released_solve_hit_plot_context_path(hit_run_dir), plot_context)

    return ReleasedSolveHit(
        rank=rank,
        hit_kind=hit.hit_kind,
        nickase_variant_id=hit.nickase_variant_id,
        release_variant_id=hit.release_variant_id,
        materialized_run_dir=_relative_to_workspace(hit_run_dir, workspace_root=workspace_root),
        render_job_path=None,
        rendered_plot_path=(
            _relative_to_workspace(rendered_plot_path, workspace_root=workspace_root)
            if rendered_plot_path is not None
            else None
        ),
        target_search_hit=hit,
    )


def materialize_released_solve_hits(
    *,
    hits: list[ReleasedTargetSearchHit],
    run_dir: Path,
    workspace_root: Path,
    output: ReleasedSolveOutputConfig,
) -> list[ReleasedSolveHit]:
    return [
        materialize_released_solve_hit(
            hit=hit,
            rank=rank,
            run_dir=run_dir,
            workspace_root=workspace_root,
            output=output,
        )
        for rank, hit in enumerate(hits[: output.materialize_top_k], start=1)
    ]


__all__ = [
    "materialize_released_solve_hit",
    "materialize_released_solve_hits",
]
