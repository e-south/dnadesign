"""
--------------------------------------------------------------------------------
<dnadesign project>
src/dnadesign/baserender/tests/test_public_import_policy.py

Import policy tests ensuring sibling tools use only baserender public exports.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from pathlib import Path


def _violations_for_tool(tool_dir: Path) -> list[str]:
    violations: list[str] = []
    for path in sorted(tool_dir.rglob("*.py")):
        text = path.read_text()
        if "dnadesign.baserender.src." in text:
            violations.append(str(path))
    return violations


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
