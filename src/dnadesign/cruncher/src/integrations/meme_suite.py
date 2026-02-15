"""
--------------------------------------------------------------------------------
<cruncher project>
src/dnadesign/cruncher/src/integrations/meme_suite.py

Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import os
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class MemeToolStatus:
    tool: str
    status: str
    path: str
    version: str
    hint: str


def resolve_tool_path(tool_path: Path | None, *, config_path: Path | None) -> Path | None:
    if tool_path is None:
        return None
    resolved = tool_path.expanduser()
    if resolved.is_absolute() or config_path is None:
        return resolved
    return (config_path.parent / resolved).resolve()


def resolve_executable(tool: str, *, tool_path: Path | None) -> Path | None:
    if tool_path is not None:
        resolved = tool_path.expanduser()
        if resolved.is_dir():
            candidate = resolved / tool
        else:
            candidate = resolved
            if candidate.name != tool:
                raise FileNotFoundError(
                    f"Configured tool_path points to '{candidate.name}', expected '{tool}'. "
                    f"Provide a bin directory or the correct executable."
                )
        if candidate.exists():
            return candidate
        raise FileNotFoundError(f"Configured tool_path does not contain '{tool}': {candidate}")
    env_dir = os.getenv("MEME_BIN")
    if env_dir:
        candidate = Path(env_dir).expanduser() / tool
        if candidate.exists():
            return candidate
    found = shutil.which(tool)
    return Path(found) if found else None


def tool_version(exe: Path) -> str | None:
    for flag in ("--version", "-version"):
        result = subprocess.run(
            [str(exe), flag],
            capture_output=True,
            text=True,
            check=False,
        )
        if result.returncode == 0:
            output = (result.stdout or result.stderr).strip()
            if output:
                return output.splitlines()[0].strip()
    return None


def check_meme_tools(*, tool: str, tool_path: Path | None) -> tuple[bool, list[MemeToolStatus]]:
    if tool not in {"auto", "streme", "meme"}:
        raise ValueError("tool must be auto, streme, or meme.")
    if tool == "auto" and tool_path is not None and tool_path.expanduser().is_file():
        raise ValueError("tool_path points to a single executable; use --tool or provide a bin directory.")

    tools = ["streme", "meme"] if tool == "auto" else [tool]
    statuses: list[MemeToolStatus] = []
    ok = True
    for name in tools:
        try:
            exe = resolve_executable(name, tool_path=tool_path)
        except FileNotFoundError as exc:
            ok = False
            statuses.append(
                MemeToolStatus(
                    tool=name,
                    status="missing",
                    path="-",
                    version="-",
                    hint=str(exc),
                )
            )
            continue
        if exe is None:
            ok = False
            statuses.append(
                MemeToolStatus(
                    tool=name,
                    status="missing",
                    path="-",
                    version="-",
                    hint="Install MEME Suite and set MEME_BIN or discover.tool_path.",
                )
            )
            continue
        version = tool_version(exe) or "-"
        status = "ok" if version != "-" else "unknown"
        hint = "-" if status == "ok" else f"Run {name} --version to verify."
        statuses.append(
            MemeToolStatus(
                tool=name,
                status=status,
                path=str(exe),
                version=version,
                hint=hint,
            )
        )
    return ok, statuses
