"""
--------------------------------------------------------------------------------
<cruncher project>
src/dnadesign/cruncher/src/app/yiu_workflow/report.py

Top-level YIU report dispatch and validate entrypoints.

Module Author(s): OpenAI Codex
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from pathlib import Path

from dnadesign.cruncher.app.yiu_workflow.ship_v4 import _build_yiu_report_v4
from dnadesign.cruncher.yiu.catalog import load_yiu_catalogs
from dnadesign.cruncher.yiu.load import load_yiu_spec
from dnadesign.cruncher.yiu.models import YiuProcessSpecV4, YiuValidationReport


def _build_yiu_report(
    spec: YiuProcessSpecV4,
    *,
    catalogs=None,
) -> YiuValidationReport:
    return _build_yiu_report_v4(spec, catalogs=catalogs)


def validate_yiu_spec(path: str | Path) -> YiuValidationReport:
    spec, _spec_path, workspace_root = load_yiu_spec(path)
    catalogs = load_yiu_catalogs(spec, workspace_root=workspace_root)
    report = _build_yiu_report(spec, catalogs=catalogs)
    return report.model_copy(
        update={
            "metadata": report.metadata.model_copy(
                update={
                    "emitted_view_count": 0,
                }
            )
        }
    )
