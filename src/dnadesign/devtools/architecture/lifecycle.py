"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/devtools/architecture/lifecycle.py

Repository-only lifecycle checks for navigation and installed components.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import ast
import re
import tomllib
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

LIFECYCLE_PATH = Path(".github/tool-lifecycle.toml")


@dataclass(frozen=True)
class Component:
    status: str
    catalog: bool
    installed: bool
    replacement: str | None = None


def load_lifecycle(repo_root: Path) -> dict[str, Component]:
    """Read explicit declarations; directory discovery never grants support."""
    path = repo_root / LIFECYCLE_PATH
    raw = tomllib.loads(path.read_text(encoding="utf-8"))
    if (
        set(raw) != {"schema_version", "components"}
        or type(raw["schema_version"]) is not int
        or raw["schema_version"] != 1
    ):
        raise ValueError(f"{path}: expected lifecycle schema_version 1 and components")
    if not isinstance(raw["components"], dict) or not raw["components"]:
        raise ValueError(f"{path}: components must be a nonempty table")
    result = {}
    for name, fields in raw["components"].items():
        if not re.fullmatch(r"[a-z][a-z0-9_]*", name) or not isinstance(fields, dict):
            raise ValueError(f"{path}: invalid component {name!r}")
        if set(fields) - {"status", "catalog", "installed", "replacement"}:
            raise ValueError(f"{path}: unknown lifecycle fields for {name}")
        status, catalog = fields.get("status"), fields.get("catalog")
        if (
            not isinstance(status, str)
            or status not in {"active", "maintenance", "retired"}
            or not isinstance(catalog, bool)
        ):
            raise ValueError(f"{path}: {name} needs a valid status and boolean catalog flag")
        installed = fields.get("installed", status != "retired")
        if not isinstance(installed, bool):
            raise ValueError(f"{path}: {name}.installed must be boolean")
        if status == "retired" and installed:
            raise ValueError(f"{path}: retired {name} cannot be installed")
        if "replacement" in fields and (
            not isinstance(fields["replacement"], str) or not fields["replacement"].strip()
        ):
            raise ValueError(f"{path}: {name}.replacement must be a nonempty reference")
        if status == "retired" and catalog:
            raise ValueError(f"{path}: retired {name} requires catalog=false")
        result[name] = Component(**(fields | {"installed": installed}))
    return result


def _command_owner(repo_root: Path, target: str) -> str:
    if not isinstance(target, str):
        raise ValueError("console script target must be a string")
    module, separator, attribute = target.partition(":")
    if not separator or not attribute or not module.startswith("dnadesign."):
        raise ValueError(f"cannot determine DNADesign command owner for {target!r}")
    if module == "dnadesign._entrypoints":
        path = repo_root / "src/dnadesign/_entrypoints.py"
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        function = next((n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == attribute), None)
        calls = (
            []
            if function is None
            else [
                n
                for n in ast.walk(function)
                if isinstance(n, ast.Call) and isinstance(n.func, ast.Name) and n.func.id == "_run"
            ]
        )
        if len(calls) != 1 or not calls[0].args or not isinstance(calls[0].args[0], ast.Constant):
            raise ValueError(f"{path}: {attribute} must delegate to one explicit command target")
        delegated = calls[0].args[0].value
        if not isinstance(delegated, str) or delegated.startswith(module + ":"):
            raise ValueError(f"{path}: invalid command delegation for {attribute}")
        return _command_owner(repo_root, delegated)
    return module.split(".")[1]


def find_installation_issues(repo_root: Path) -> list[str]:
    """Check declared console scripts without importing optional tool runtimes."""
    components = load_lifecycle(repo_root)
    project = tomllib.loads((repo_root / "pyproject.toml").read_text(encoding="utf-8"))
    issues = []
    for command, target in project.get("project", {}).get("scripts", {}).items():
        try:
            owner = _command_owner(repo_root, target)
        except (OSError, ValueError, SyntaxError) as exc:
            issues.append(f"command {command}: {exc}")
            continue
        component = components.get(owner)
        if component is None or not component.installed:
            issues.append(
                f"command {command}: owner {owner} is unclassified, retired or repository-only; "
                "remove or reassign the entrypoint"
            )
    return issues


def wheel_issues(components: dict[str, Component], members: Iterable[str]) -> list[str]:
    """Validate emitted contents, including resources rather than just imports."""
    owners = {parts[1] for name in members if len(parts := name.split("/")) > 2 and parts[0] == "dnadesign"}
    installed = {name for name, component in components.items() if component.installed}
    issues = [
        f"wheel contains unclassified, retired or repository-only component: {name}"
        for name in sorted(owners - installed)
    ]
    issues.extend(f"wheel is missing installed component: {name}" for name in sorted(installed - owners))
    return issues
