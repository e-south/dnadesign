"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/baserender/tests/test_public_import_policy.py

Import policy tests ensuring sibling tools use only baserender public exports.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import ast
from pathlib import Path

_FORBIDDEN_BASERENDER_INTERNAL_PREFIX = "dnadesign.baserender.src."


def _violations_for_tool(tool_dir: Path) -> list[str]:
    violations: list[str] = []
    for path in sorted(tool_dir.rglob("*.py")):
        source = path.read_text(encoding="utf-8")
        if _FORBIDDEN_BASERENDER_INTERNAL_PREFIX not in source:
            continue
        tree = ast.parse(source, filename=str(path))
        matches: list[str] = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    if alias.name.startswith(_FORBIDDEN_BASERENDER_INTERNAL_PREFIX):
                        matches.append(alias.name)
            elif isinstance(node, ast.ImportFrom):
                module = node.module or ""
                if module.startswith(_FORBIDDEN_BASERENDER_INTERNAL_PREFIX):
                    matches.append(module)
        if matches:
            violations.append(f"{path}: {sorted(set(matches))}")
    return violations


def test_public_import_policy_prefilter_still_catches_forbidden_imports(tmp_path: Path) -> None:
    tool_dir = tmp_path / "tool"
    tool_dir.mkdir()
    (tool_dir / "safe.py").write_text("from dnadesign.baserender import RenderJob\n", encoding="utf-8")
    (tool_dir / "bad.py").write_text(
        "from dnadesign.baserender.src.render import render_record\n",
        encoding="utf-8",
    )

    violations = _violations_for_tool(tool_dir)

    assert len(violations) == 1
    assert "bad.py" in violations[0]
    assert "dnadesign.baserender.src.render" in violations[0]


def test_sibling_tools_do_not_import_baserender_internal_modules() -> None:
    repo_root = Path(__file__).resolve().parents[4]
    tools_root = repo_root / "src" / "dnadesign"

    violations: list[str] = []
    for tool_dir in sorted(tools_root.iterdir(), key=lambda p: p.name):
        if not tool_dir.is_dir():
            continue
        if tool_dir.name == "baserender":
            continue
        if tool_dir.name.startswith("."):
            continue
        violations.extend(_violations_for_tool(tool_dir))

    assert not violations, f"Found disallowed deep baserender imports in sibling tools: {violations}"
