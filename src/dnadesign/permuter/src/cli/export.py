"""
--------------------------------------------------------------------------------
<dnadesign project>
src/dnadesign/permuter/src/cli/export.py

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import shlex
import sys
from pathlib import Path

from dnadesign.permuter.src.core.paths import normalize_data_path
from dnadesign.permuter.src.core.storage import (
    append_journal,
    append_record_md,
    read_parquet,
)


def export_(data: Path, fmt: str, out: Path):
    records = normalize_data_path(data)
    df = read_parquet(records)
    out.parent.mkdir(parents=True, exist_ok=True)
    fmt = fmt.lower()
    if fmt == "csv":
        df.to_csv(out, index=False)
    elif fmt == "jsonl":
        with out.open("w", encoding="utf-8") as fh:
            for _, r in df.iterrows():
                fh.write(r.to_json() + "\n")
    else:
        raise ValueError("Unsupported export format (csv|jsonl)")
    try:
        cmd = shlex.join(sys.argv)
    except Exception:
        cmd = " ".join(sys.argv)
        append_record_md(records.parent, "export", cmd)
    append_journal(
        records.parent, "EXPORT", [f"fmt: {fmt}", f"out: {out}", f"command: {cmd}"]
    )
