"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/cruncher/tests/core/test_yiu_module_boundaries.py

Architecture invariants for YIU spec and payload-resolution seams.

Module Author(s): Eric J. South
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


def test_pwm_context_stays_a_public_resolution_facade() -> None:
    text = _read_yiu_source("pwm_context.py")
    tree = _parse_yiu_source("pwm_context.py")

    assert "from dnadesign.cruncher.yiu.pwm_context_sources import resolve_context_model" in text
    assert "import yaml" not in text
    assert "import json" not in text
    assert "pyarrow" not in text
    assert "pandas" not in text
    function_names = [node.name for node in tree.body if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))]
    assert function_names == ["resolve_motif_context"]


def test_pwm_context_sources_keep_sample_context_loading_isolated() -> None:
    text = _read_yiu_source("pwm_context_sources.py")
    tree = _parse_yiu_source("pwm_context_sources.py")

    assert "from dnadesign.cruncher.yiu.pwm_context_sample_context import sample_context_to_model" in text
    assert "pyarrow" not in text
    assert "pandas" not in text
    function_names = [node.name for node in tree.body if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))]
    assert function_names == ["resolve_context_model"]


def test_pwm_context_sample_context_stays_an_orchestration_seam() -> None:
    text = _read_yiu_source("pwm_context_sample_context.py")
    tree = _parse_yiu_source("pwm_context_sample_context.py")

    assert "from dnadesign.cruncher.yiu.pwm_context_sample_motifs import (" in text
    assert "from dnadesign.cruncher.yiu.pwm_context_sample_occurrences import load_selected_occurrence_rows" in text
    assert "import json" not in text
    assert "import math" not in text
    assert "pyarrow" not in text
    assert "pandas" not in text
    function_names = [node.name for node in tree.body if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))]
    assert function_names == ["sample_context_to_model"]


def test_visual_system_stays_a_view_registry() -> None:
    text = _read_yiu_source("visual_system.py")
    tree = _parse_yiu_source("visual_system.py")

    assert "from dnadesign.cruncher.yiu.visual_directions import (" in text
    assert "bench_strip_style_foundation" not in text
    assert "_COMMON_STRIP_STYLE_OVERRIDES" not in text
    function_names = [node.name for node in tree.body if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))]
    assert function_names == ["_build_style_profile", "get_yiu_style_profile", "build_yiu_style_overrides"]
