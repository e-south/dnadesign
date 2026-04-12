"""
Workspace services for latentdna.
"""

from __future__ import annotations

from pathlib import Path

from ..contracts.result import CommandResult
from ..workspaces.loader import default_workspace_root, load_workspace_config, scaffold_workspace


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
    context = load_workspace_config(workspace_dir)
    result = CommandResult(
        command="workspace init",
        workspace_id=context.workspace_id,
        status="ok",
        artifact_kind="workspace",
        artifact_id=context.workspace_id,
        outputs=[workspace_dir.as_posix()],
        inputs={
            "template": template,
            **({"study_dir": str(from_study_dir)} if from_study_dir is not None else {}),
        },
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
    context = load_workspace_config(workspace)
    payload: dict[str, str | int | None] = {
        "workspace_id": context.workspace_id,
        "workspace_dir": context.workspace_dir.as_posix(),
        "config_path": context.config_path.as_posix(),
        "sources": len(context.config.sources),
        "views": len(context.config.views),
    }
    if context.config.study_binding is not None:
        payload["study_binding_kind"] = context.config.study_binding.kind
        payload["study_binding_study_dir"] = context.config.study_binding.study_dir
    return payload
