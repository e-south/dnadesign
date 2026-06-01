"""OPAL command contracts for Stage B TFBS campaign artifacts."""

from __future__ import annotations

from pathlib import Path


def opal_validate_command(config_path: Path) -> list[str]:
    """Return the deterministic OPAL validation command for one campaign config."""

    return ["uv", "run", "opal", "validate", "-c", str(config_path)]


def opal_ingest_command(config_path: Path, label_input_path: Path, *, round_index: int) -> list[str]:
    """Return the deterministic OPAL label-ingest command for one campaign round."""

    return [
        "uv",
        "run",
        "opal",
        "ingest-y",
        "-c",
        str(config_path),
        "--in",
        str(label_input_path),
        "--round",
        str(int(round_index)),
        "--unknown-sequences",
        "error",
        "--apply",
    ]
