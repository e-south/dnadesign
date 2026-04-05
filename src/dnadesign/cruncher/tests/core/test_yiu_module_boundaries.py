"""
--------------------------------------------------------------------------------
<cruncher project>
src/dnadesign/cruncher/tests/core/test_yiu_module_boundaries.py

Architecture invariants for YIU spec and payload-resolution seams.

Module Author(s): OpenAI Codex
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import ast
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def _read_yiu_source(path: str) -> str:
    return (ROOT / "src" / "yiu" / path).read_text(encoding="utf-8")


def _parse_yiu_source(path: str) -> ast.Module:
    return ast.parse(_read_yiu_source(path))


def test_spec_models_stays_a_public_facade() -> None:
    text = _read_yiu_source("spec_models.py")
    tree = _parse_yiu_source("spec_models.py")

    assert "from dnadesign.cruncher.yiu.spec_input_models import" in text
    assert "from dnadesign.cruncher.yiu.spec_pwm_models import" in text
    assert "from dnadesign.cruncher.yiu.spec_rendering_models import" in text
    assert "from pydantic import" not in text
    assert not [node for node in tree.body if isinstance(node, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef))]


def test_payload_resolution_stays_an_orchestration_seam() -> None:
    text = _read_yiu_source("payload_resolution.py")
    tree = _parse_yiu_source("payload_resolution.py")

    assert "from dnadesign.cruncher.yiu.input_payload_models import ResolvedInputPayload" in text
    assert "from dnadesign.cruncher.yiu.sample_hit_sources import metadata_text, resolve_sample_hit_payload" in text
    assert "import csv" not in text
    assert "read_parquet" not in text
    assert "DictReader" not in text
    assert "pyarrow" not in text
    assert "pandas" not in text
    function_names = [node.name for node in tree.body if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))]
    assert function_names == ["resolve_input_payload"]
