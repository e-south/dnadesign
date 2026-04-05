"""
--------------------------------------------------------------------------------
<cruncher project>
src/dnadesign/cruncher/src/app/yiu_workflow/validate.py

Validate payload-centric YIU specs.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from pathlib import Path

from dnadesign.cruncher.yiu.bundle_models import YiuValidationReport, build_validation_report
from dnadesign.cruncher.yiu.load import load_yiu_spec
from dnadesign.cruncher.yiu.normalizer import normalize_payload


def validate_yiu_spec(path: str | Path) -> YiuValidationReport:
    spec, _resolved_spec_path, workspace_root = load_yiu_spec(path)
    normalized = normalize_payload(spec, workspace_root=workspace_root)
    return build_validation_report(spec_name=spec.yiu.name, normalized=normalized)
