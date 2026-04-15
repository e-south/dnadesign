"""
Study documentation reference helpers for latentdna.
"""

from __future__ import annotations

from pathlib import Path

from ..contracts.errors import WorkspaceValidationError
from ..workspaces.loader import WorkspaceContext
from ..workspaces.paths import resolve_repo_path


def resolve_docs_ref(context: WorkspaceContext, docs_ref: str) -> dict[str, str]:
    binding = context.config.study_binding
    if binding is None:
        raise WorkspaceValidationError(f"deliverable docs_ref requires study_binding: {docs_ref}")
    prefix = f"study:{binding.study_id}/"
    if not docs_ref.startswith(prefix):
        raise WorkspaceValidationError(
            f"docs_ref must start with {prefix!r} for workspace {context.workspace_id}: {docs_ref}"
        )
    relative_ref = docs_ref.removeprefix(prefix)
    docs_root = resolve_repo_path(binding.docs_root)
    markdown_path = docs_root / f"{relative_ref}.md"
    yaml_path = docs_root / f"{relative_ref}.yaml"
    if markdown_path.is_file():
        path = markdown_path
    elif yaml_path.is_file():
        path = yaml_path
    else:
        raise WorkspaceValidationError(f"docs_ref path does not exist under {docs_root}: {relative_ref}")
    return {
        "docs_ref": docs_ref,
        "relative_ref": relative_ref,
        "path": path.resolve().as_posix(),
    }


def read_docs_ref(context: WorkspaceContext, docs_ref: str) -> dict[str, str]:
    resolved = resolve_docs_ref(context, docs_ref)
    path = Path(resolved["path"])
    resolved["content"] = path.read_text(encoding="utf-8")
    return resolved
