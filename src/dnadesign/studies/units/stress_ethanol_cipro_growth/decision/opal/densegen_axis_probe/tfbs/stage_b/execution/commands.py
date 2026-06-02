"""OPAL command contracts for Stage B execution."""

from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Sequence


def opal_validate_command(config_path: Path) -> list[str]:
    return ["uv", "run", "opal", "validate", "-c", str(config_path), "--json"]


def opal_init_command(config_path: Path) -> list[str]:
    return ["uv", "run", "opal", "init", "-c", str(config_path), "--json"]


def opal_ingest_command(config_path: Path, labels_path: Path, *, round_index: int) -> list[str]:
    return [
        "uv",
        "run",
        "opal",
        "ingest-y",
        "-c",
        str(config_path),
        "--round",
        str(int(round_index)),
        "--in",
        str(labels_path),
        "--unknown-sequences",
        "error",
        "--apply",
        "--json",
    ]


def opal_run_command(config_path: Path, *, round_index: int, resume: bool) -> list[str]:
    command = ["uv", "run", "opal", "run", "-c", str(config_path), "--round", str(int(round_index)), "--json"]
    if resume:
        command.append("--resume")
    return command


def opal_status_command(config_path: Path) -> list[str]:
    return ["uv", "run", "opal", "status", "-c", str(config_path), "--with-ledger", "--json"]


def run_command(command: Sequence[str], *, cwd: Path) -> None:
    subprocess.run(list(map(str, command)), cwd=cwd, check=True)
