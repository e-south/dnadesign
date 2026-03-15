"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/ops/runbooks/runbook_paths.py

Resolved-path rewriting for ops runbook contracts loaded from disk.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .schema import OrchestrationRunbookV1

DENSEGEN_POST_RUN_TEMPLATE_DEFAULT = Path("docs/bu-scc/jobs/densegen-analysis.qsub")


def _resolve_path_from_runbook_base(path_value: Path, *, runbook_base_dir: Path) -> Path:
    expanded = path_value.expanduser()
    if expanded.is_absolute():
        return expanded
    return (runbook_base_dir / expanded).resolve()


def _resolve_repo_root_from_module() -> Path | None:
    current = Path(__file__).resolve()
    for parent in current.parents:
        if (parent / "pyproject.toml").exists():
            return parent
    return None


def _resolve_densegen_post_run_template(path_value: Path, *, runbook_base_dir: Path) -> Path:
    expanded = path_value.expanduser()
    if expanded.is_absolute():
        return expanded
    if expanded == DENSEGEN_POST_RUN_TEMPLATE_DEFAULT:
        repo_root = _resolve_repo_root_from_module()
        if repo_root is not None:
            return (repo_root / expanded).resolve()
    return (runbook_base_dir / expanded).resolve()


def resolve_runbook_paths(runbook: "OrchestrationRunbookV1", *, runbook_base_dir: Path) -> "OrchestrationRunbookV1":
    densegen = runbook.densegen
    if densegen is not None:
        post_run = densegen.post_run.model_copy(
            update={
                "qsub_template": _resolve_densegen_post_run_template(
                    densegen.post_run.qsub_template,
                    runbook_base_dir=runbook_base_dir,
                )
            }
        )
        densegen = densegen.model_copy(
            update={
                "config": _resolve_path_from_runbook_base(densegen.config, runbook_base_dir=runbook_base_dir),
                "qsub_template": _resolve_path_from_runbook_base(
                    densegen.qsub_template, runbook_base_dir=runbook_base_dir
                ),
                "post_run": post_run,
            }
        )

    infer = runbook.infer
    if infer is not None:
        infer = infer.model_copy(
            update={
                "config": _resolve_path_from_runbook_base(infer.config, runbook_base_dir=runbook_base_dir),
                "qsub_template": _resolve_path_from_runbook_base(
                    infer.qsub_template, runbook_base_dir=runbook_base_dir
                ),
            }
        )

    notify = runbook.notify
    if notify is not None:
        notify = notify.model_copy(
            update={
                "profile": _resolve_path_from_runbook_base(notify.profile, runbook_base_dir=runbook_base_dir),
                "cursor": _resolve_path_from_runbook_base(notify.cursor, runbook_base_dir=runbook_base_dir),
                "spool_dir": _resolve_path_from_runbook_base(notify.spool_dir, runbook_base_dir=runbook_base_dir),
                "qsub_template": _resolve_path_from_runbook_base(
                    notify.qsub_template, runbook_base_dir=runbook_base_dir
                ),
            }
        )

    logging = runbook.logging.model_copy(
        update={
            "stdout_dir": _resolve_path_from_runbook_base(runbook.logging.stdout_dir, runbook_base_dir=runbook_base_dir)
        }
    )

    return runbook.model_copy(
        update={
            "workspace_root": _resolve_path_from_runbook_base(
                runbook.workspace_root, runbook_base_dir=runbook_base_dir
            ),
            "densegen": densegen,
            "infer": infer,
            "notify": notify,
            "logging": logging,
        }
    )
