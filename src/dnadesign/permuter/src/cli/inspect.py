"""
--------------------------------------------------------------------------------
<dnadesign project>
src/dnadesign/permuter/src/cli/inspect.py

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import shlex
import sys
from pathlib import Path

from rich.console import Console
from rich.table import Table

from dnadesign.permuter.src.core.paths import normalize_data_path
from dnadesign.permuter.src.core.storage import (
    append_record_md,
    read_parquet,
)

console = Console()


def inspect_(data: Path, head: int = 5):
    records = normalize_data_path(data)
    df = read_parquet(records)
    n = len(df)
    bio_types = sorted(df["bio_type"].dropna().unique().tolist())
    lengths = df["length"].describe()
    metric_cols = [c for c in df.columns if c.startswith("permuter__metric__")]

    table = Table(title="Permuter dataset snapshot")
    table.add_column("Property")
    table.add_column("Value")

    table.add_row("Rows", str(n))
    table.add_row("bio_type(s)", ", ".join(bio_types) or "—")
    table.add_row("length.mean", f"{lengths['mean']:.2f}")
    table.add_row("length.min", str(int(lengths["min"])))
    table.add_row("length.max", str(int(lengths["max"])))
    table.add_row("metrics", ", ".join(metric_cols) or "—")

    console.print(table)

    console.print("[bold]Head:[/bold]")
    console.print(df.head(head))
    try:
        cmd = shlex.join(sys.argv)
    except Exception:
        cmd = " ".join(sys.argv)
    append_record_md(records.parent, "inspect", cmd)
