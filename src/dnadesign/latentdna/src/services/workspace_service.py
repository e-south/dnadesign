"""
Workspace services for latentdna.
"""

from __future__ import annotations

import shutil
from pathlib import Path

from ..contracts.errors import WorkspaceValidationError
from ..contracts.result import CommandResult
from ..workspaces.loader import load_workspace_config
from ..workspaces.paths import default_workspace_root
from ..workspaces.scaffold import scaffold_workspace
from ._artifacts import ARTIFACT_KIND_DIRS

_WORKSPACE_REFRESH_SPECIAL_TARGETS = ("runs", "catalog", "logs", "status")
_WORKSPACE_REFRESH_TARGETS = frozenset({"all", *ARTIFACT_KIND_DIRS.values(), *_WORKSPACE_REFRESH_SPECIAL_TARGETS})


def workspace_where() -> dict[str, str]:
    root, source = default_workspace_root()
    return {"workspace_root": root.as_posix(), "workspace_root_source": source}


def init_workspace(
    *,
    workspace: str | Path,
    template: str,
    from_study_dir: str | Path | None = None,
) -> dict[str, object]:
    workspace_dir = Path(workspace).resolve()
    scaffold_workspace(workspace_dir=workspace_dir, template=template, from_study_dir=from_study_dir)
    context = load_workspace_config(workspace_dir, validate_plot_semantics=True)
    result = CommandResult(
        command="workspace init",
        workspace_id=context.workspace_id,
        status="ok",
        artifact_kind="workspace",
        artifact_id=context.workspace_id,
        outputs=[workspace_dir.as_posix()],
        inputs={"template": template},
        metrics={
            "sources": len(context.config.sources),
            "views": len(context.config.views),
            "deliverables": len(context.config.deliverables),
        },
    )
    payload = result.model_dump(mode="json")
    payload["config_path"] = (workspace_dir / "config.yaml").as_posix()
    if context.config.study_binding is not None:
        payload["study_binding"] = context.config.study_binding.model_dump(mode="json")
    return payload


def list_workspaces(root: str | Path | None = None) -> list[dict[str, str]]:
    if root is None:
        root_path, _ = default_workspace_root()
    else:
        root_path = Path(root).resolve()
    if not root_path.exists():
        return []
    workspaces = []
    for candidate in sorted(root_path.iterdir()):
        if (candidate / "config.yaml").is_file():
            workspaces.append({"workspace_dir": candidate.resolve().as_posix(), "workspace_id": candidate.name})
    return workspaces


def show_workspace(workspace: str | Path) -> dict[str, str | int]:
    context = load_workspace_config(workspace, validate_plot_semantics=False)
    payload: dict[str, str | int | None] = {
        "workspace_id": context.workspace_id,
        "workspace_title": context.config.workspace.title,
        "workspace_dir": context.workspace_dir.as_posix(),
        "config_path": context.config_path.as_posix(),
        "sources": len(context.config.sources),
        "views": len(context.config.views),
    }
    if context.config.study_binding is not None:
        payload["study_binding_study_id"] = context.config.study_binding.study_id
        payload["study_binding_docs_root"] = context.config.study_binding.docs_root
    return payload


def _resolve_refresh_targets(targets: tuple[str, ...] | list[str] | None) -> list[str]:
    requested = [target.strip() for target in (targets or ["all"]) if target.strip()]
    if not requested:
        requested = ["all"]
    invalid = sorted(target for target in requested if target not in _WORKSPACE_REFRESH_TARGETS)
    if invalid:
        raise WorkspaceValidationError(
            "workspace refresh target must be one of: "
            f"{', '.join(sorted(_WORKSPACE_REFRESH_TARGETS))}; got {', '.join(invalid)}"
        )
    if "all" in requested:
        return [
            *sorted(set(ARTIFACT_KIND_DIRS.values())),
            "runs",
            "logs",
            "catalog",
            "status",
        ]
    return list(dict.fromkeys(requested))


def _assert_workspace_local_path(workspace_dir: Path, candidate: Path) -> Path:
    resolved = candidate.resolve()
    workspace_root = workspace_dir.resolve()
    if resolved != workspace_root and workspace_root not in resolved.parents:
        raise WorkspaceValidationError(f"workspace refresh target escapes workspace: {resolved}")
    return resolved


def _refresh_target_path(context, target: str) -> Path:
    if target == "catalog":
        return context.output_root / "catalog.json"
    return context.output_root / target


def _source_boundary_paths(context) -> list[str]:
    boundaries: set[str] = set()
    for source in context.config.sources.values():
        if getattr(source, "kind", None) == "usr":
            root = Path(str(source.root))
            if not root.is_absolute():
                root = context.workspace_dir / root
            boundaries.add(root.resolve().as_posix())
            continue
        raw_path = getattr(source, "path", None)
        if raw_path is None:
            continue
        candidate = Path(str(raw_path))
        if not candidate.is_absolute():
            candidate = context.workspace_dir / candidate
        boundaries.add(candidate.resolve().as_posix())
    return sorted(boundaries)


def refresh_workspace(
    workspace: str | Path,
    *,
    targets: tuple[str, ...] | list[str] | None = None,
    dry_run: bool = False,
) -> dict[str, object]:
    context = load_workspace_config(workspace)
    resolved_targets = _resolve_refresh_targets(targets)
    planned_paths = [
        _assert_workspace_local_path(context.workspace_dir, _refresh_target_path(context, target))
        for target in resolved_targets
    ]
    existing_paths = [path for path in planned_paths if path.exists()]

    if not dry_run:
        for path in existing_paths:
            if path.is_symlink():
                path.unlink()
            elif path.is_dir():
                shutil.rmtree(path)
            else:
                path.unlink()
        context.output_root.mkdir(parents=True, exist_ok=True)
        load_workspace_config(context.workspace_dir, validate_plot_semantics=True)
    source_paths = _source_boundary_paths(context)
    result = CommandResult(
        command="workspace refresh",
        workspace_id=context.workspace_id,
        status="ok",
        dry_run=dry_run,
        artifact_kind="workspace",
        artifact_id=context.workspace_id,
        outputs=[context.output_root.as_posix()],
        inputs={"targets": resolved_targets},
        metrics={
            "planned_paths": len(planned_paths),
            "removed_paths": len(existing_paths) if not dry_run else 0,
            "existing_paths": len(existing_paths),
        },
    )
    payload = result.model_dump(mode="json")
    payload["planned_removals"] = [path.as_posix() for path in planned_paths]
    payload["removed_paths"] = [] if dry_run else [path.as_posix() for path in existing_paths]
    payload["protected_paths"] = source_paths
    payload["post_refresh_validation"] = "skipped" if dry_run else "ok"
    return payload
