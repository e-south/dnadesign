"""
Study documentation reference helpers for latentdna.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from ..contracts.errors import WorkspaceValidationError
from ..workspaces.paths import resolve_repo_path

if TYPE_CHECKING:
    from ..workspaces.loader import WorkspaceContext

_DOCS_REF_SUFFIXES = (".md", ".yaml")


def _relative_docs_ref(prefix: str, docs_ref: str, *, workspace_id: str) -> str:
    if not docs_ref.startswith(prefix):
        raise WorkspaceValidationError(f"docs_ref must start with {prefix!r} for workspace {workspace_id}: {docs_ref}")
    relative_ref = docs_ref.removeprefix(prefix)
    if not relative_ref:
        raise WorkspaceValidationError(f"docs_ref must name a document under {prefix!r}: {docs_ref}")
    parts = Path(relative_ref).parts
    if Path(relative_ref).is_absolute() or any(part in {"", ".", ".."} for part in parts):
        raise WorkspaceValidationError(f"docs_ref must stay under the study docs root: {docs_ref}")
    return relative_ref


def resolve_docs_ref_path(
    *,
    study_id: str,
    deliverable_docs_root: str | Path,
    docs_ref: str,
    workspace_id: str,
) -> dict[str, str]:
    prefix = f"study:{study_id}/"
    relative_ref = _relative_docs_ref(prefix, docs_ref, workspace_id=workspace_id)
    resolved_docs_root = resolve_repo_path(deliverable_docs_root)
    for suffix in _DOCS_REF_SUFFIXES:
        candidate = (resolved_docs_root / f"{relative_ref}{suffix}").resolve()
        if resolved_docs_root not in candidate.parents:
            raise WorkspaceValidationError(f"docs_ref must stay under {resolved_docs_root}: {docs_ref}")
        if candidate.is_file():
            return {
                "docs_ref": docs_ref,
                "relative_ref": relative_ref,
                "path": candidate.as_posix(),
            }
    raise WorkspaceValidationError(f"docs_ref path does not exist under {resolved_docs_root}: {relative_ref}")


def resolve_docs_ref(context: WorkspaceContext, docs_ref: str) -> dict[str, str]:
    binding = context.config.study_binding
    if binding is None:
        raise WorkspaceValidationError(f"deliverable docs_ref requires study_binding: {docs_ref}")
    return resolve_docs_ref_path(
        study_id=binding.study_id,
        deliverable_docs_root=binding.deliverable_docs_root,
        docs_ref=docs_ref,
        workspace_id=context.workspace_id,
    )


def read_docs_ref(context: WorkspaceContext, docs_ref: str) -> dict[str, str]:
    resolved = resolve_docs_ref(context, docs_ref)
    path = Path(resolved["path"])
    resolved["content"] = path.read_text(encoding="utf-8")
    return resolved
