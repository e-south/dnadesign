"""Candidate-representation set resolution for notebook and status surfaces."""

from __future__ import annotations

import numpy as np

from ..contracts.notebook import WorkspaceNotebookCandidateSet, WorkspaceNotebookCandidateView
from ..labels import humanize_candidate


def _normalized_role(value: object) -> str:
    return str(value or "").strip().lower()


def _view_tags(view) -> dict[str, str]:
    return {str(key): str(value) for key, value in dict(getattr(view, "tags", {}) or {}).items()}


def _matches_tags(view, include_tags: dict[str, str]) -> bool:
    tags = _view_tags(view)
    return all(tags.get(str(key)) == str(value) for key, value in include_tags.items())


def candidate_set_view_ids(context, candidate_set_id: str) -> list[str]:
    candidate_set = context.config.candidate_sets[candidate_set_id]
    explicit_view_ids = [str(view_id) for view_id in candidate_set.views]
    excluded_roles = {_normalized_role(role) for role in candidate_set.exclude_roles}
    resolved: list[str] = []
    seen: set[str] = set()

    for view_id in explicit_view_ids:
        if view_id not in context.config.views:
            continue
        view = context.config.views[view_id]
        role = _normalized_role(getattr(view, "role", None))
        if role in excluded_roles:
            continue
        resolved.append(view_id)
        seen.add(view_id)

    if candidate_set.include_tags:
        for view_id, view in context.config.views.items():
            if view_id in seen:
                continue
            role = _normalized_role(getattr(view, "role", None))
            if role in excluded_roles:
                continue
            if _matches_tags(view, candidate_set.include_tags):
                resolved.append(view_id)
                seen.add(view_id)

    return resolved


def _view_shape(context, view_id: str) -> tuple[int | None, int | None]:
    matrix_path = context.output_root / "views" / view_id / "matrix.npy"
    if not matrix_path.is_file():
        return None, None
    try:
        matrix = np.load(matrix_path, mmap_mode="r")
    except Exception:
        return None, None
    if len(matrix.shape) < 2:
        return int(matrix.shape[0]) if matrix.shape else None, None
    return int(matrix.shape[0]), int(matrix.shape[1])


def _is_materialized(context, view_id: str) -> bool:
    view_dir = context.output_root / "views" / view_id
    return (view_dir / "matrix.npy").is_file() and (view_dir / "rows.parquet").is_file()


def _candidate_status(context, view_id: str, *, role: str) -> str:
    if role in {"planned", "retired", "hidden"}:
        return role
    if _is_materialized(context, view_id):
        return "materialized"
    return "missing"


def _candidate_label(view_id: str, tags: dict[str, str]) -> str:
    label = humanize_candidate(tags)
    return label or humanize_candidate(view_id)


def _candidate_view_payload(context, *, candidate_set_id: str, view_id: str) -> WorkspaceNotebookCandidateView:
    view = context.require_view(view_id)
    role = _normalized_role(getattr(view, "role", None)) or None
    tags = _view_tags(view)
    rows, dims = _view_shape(context, view_id)
    candidate_set = context.config.candidate_sets[candidate_set_id]
    label = _candidate_label(view_id, tags)
    return WorkspaceNotebookCandidateView(
        view_id=view_id,
        label=label,
        panel_title=str(candidate_set.panel_titles.get(view_id) or label),
        status=_candidate_status(context, view_id, role=role or ""),
        role=role,
        model=tags.get("model"),
        family=tags.get("family"),
        scope=tags.get("scope"),
        coordinate_space_id=str(getattr(view, "coordinate_space_id", "") or "") or None,
        tags=tags,
        materialized=_is_materialized(context, view_id),
        rows=rows,
        dims=dims,
    )


def build_workspace_candidate_sets(
    context,
    *,
    notebook_id: str | None = None,
    visible_view_ids: set[str] | None = None,
) -> list[WorkspaceNotebookCandidateSet]:
    notebook = None
    if notebook_id is not None:
        notebook = context.require_notebook(notebook_id)
    elif "latent_geometry_browser" in context.config.notebooks:
        notebook = context.config.notebooks["latent_geometry_browser"]
    elif context.config.notebooks:
        notebook = next(iter(context.config.notebooks.values()))

    configured_ids = list(getattr(notebook, "candidate_sets", []) or []) if notebook is not None else []
    candidate_set_ids = configured_ids or list(context.config.candidate_sets)
    visible = set(visible_view_ids or ())
    payloads: list[WorkspaceNotebookCandidateSet] = []
    for candidate_set_id in candidate_set_ids:
        if candidate_set_id not in context.config.candidate_sets:
            continue
        candidate_set = context.config.candidate_sets[candidate_set_id]
        view_ids = candidate_set_view_ids(context, candidate_set_id)
        views = [
            _candidate_view_payload(context, candidate_set_id=candidate_set_id, view_id=view_id) for view_id in view_ids
        ]
        available_view_ids = [view_id for view_id in view_ids if not visible or view_id in visible]
        panel_titles_by_view = {view.view_id: view.panel_title for view in views}
        payloads.append(
            WorkspaceNotebookCandidateSet(
                candidate_set_id=candidate_set_id,
                label=candidate_set.label,
                description=candidate_set.description,
                view_ids=view_ids,
                available_view_ids=available_view_ids,
                panel_titles=[panel_titles_by_view.get(view_id, view_id) for view_id in available_view_ids],
                views=views,
            )
        )
    return payloads


__all__ = ["build_workspace_candidate_sets", "candidate_set_view_ids"]
