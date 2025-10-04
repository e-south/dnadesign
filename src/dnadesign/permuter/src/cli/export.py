"""
--------------------------------------------------------------------------------
<dnadesign project>
src/dnadesign/permuter/src/cli/export.py

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from pathlib import Path

from dnadesign.permuter.src.core.storage import read_parquet


def export_(data: Path, fmt: str, out: Path):
    df = read_parquet(data)
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
