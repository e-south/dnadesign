"""
--------------------------------------------------------------------------------
<dnadesign project>
src/dnadesign/permuter/src/core/storage.py

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd


def ensure_output_dir(path: Path):
    path.mkdir(parents=True, exist_ok=True)


def atomic_write_parquet(df: pd.DataFrame, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".parquet.tmp")
    df.to_parquet(tmp, index=False)
    tmp.replace(path)


def read_parquet(path: Path) -> pd.DataFrame:
    return pd.read_parquet(path)


# --- reference sequence sidecar ---------------------------------------------


def write_ref_fasta(dataset_dir: Path, ref_name: str, sequence: str) -> Path:
    fasta = dataset_dir / "REF.fa"
    lines = [f">{ref_name}\n"]
    # wrap 80 cols
    seq = sequence.strip()
    lines += [seq[i : i + 80] + "\n" for i in range(0, len(seq), 80)]
    fasta.write_text("".join(lines), encoding="utf-8")
    return fasta


def read_ref_fasta(dataset_dir: Path) -> tuple[str, str] | None:
    p = dataset_dir / "REF.fa"
    if not p.exists():
        return None
    name = ""
    seq = []
    with p.open("r", encoding="utf-8") as fh:
        for line in fh:
            if line.startswith(">"):
                name = line[1:].strip()
            else:
                seq.append(line.strip())
    s = "".join(seq).strip()
    if not s:
        return None
    return name, s
