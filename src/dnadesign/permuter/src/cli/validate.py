"""
--------------------------------------------------------------------------------
<dnadesign project>
src/dnadesign/permuter/src/cli/validate.py

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import hashlib
from pathlib import Path

from rich.console import Console

from dnadesign.permuter.src.core.storage import read_parquet

console = Console()
_CORE = ["id", "bio_type", "sequence", "alphabet", "length", "source", "created_at"]


def _sha1(bio_type: str, sequence: str) -> str:
    return hashlib.sha1(f"{bio_type}|{sequence}".encode("utf-8")).hexdigest()


def validate(data: Path, strict: bool = False):
    df = read_parquet(data)

    # core columns
    missing = [c for c in _CORE if c not in df.columns]
    if missing:
        raise ValueError(f"USR core columns missing: {missing}")

    # id integrity
    recomputed = df.apply(
        lambda r: _sha1(str(r["bio_type"]), str(r["sequence"])), axis=1
    )
    bad = (recomputed != df["id"]).sum()
    if bad:
        raise ValueError(f"{bad} row(s) have incorrect id for (bio_type|sequence)")

    # namespacing
    for c in df.columns:
        if c in _CORE:
            continue
        if "__" not in c:
            if strict:
                raise ValueError(f"Non-namespaced derived column in strict mode: {c}")

    # presence of required permuter columns for DMS
    required = [
        "permuter__job",
        "permuter__ref",
        "permuter__protocol",
        "permuter__var_id",
    ]
    miss = [c for c in required if c not in df.columns]
    if miss and strict:
        raise ValueError(f"Missing required permuter columns: {miss}")

    console.print(f"[green]✔[/green] Validation passed for {data}")
