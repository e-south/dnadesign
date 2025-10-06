"""
--------------------------------------------------------------------------------
<dnadesign project>
src/dnadesign/permuter/src/core/storage.py

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from textwrap import dedent

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


# --- per-dataset journal -----------------------------------------------------


def append_journal(
    dataset_dir: Path, section: str, lines: list[str] | tuple[str, ...] = ()
) -> Path:
    """
    Append a structured entry to JOURNAL.md next to records.parquet.
    Users can also freely add their own notes to this file.
    """
    journal = dataset_dir / "JOURNAL.md"
    if not journal.exists():
        header = (
            "# Permuter Journal\n\n"
            "_Automatic command log and scratch pad for this dataset._\n\n"
        )
        journal.write_text(header, encoding="utf-8")
    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S %Z")
    sep = "\n".join(f"- {ln}" for ln in (lines or []))
    entry = f"\n## {section} · {now}\n\n{sep}\n"
    with journal.open("a", encoding="utf-8") as fh:
        fh.write(entry)
    return journal


# --- per-dataset RECORD.md (human-facing command log + scratch pad) ----------


def init_record_md(
    *,
    dataset_dir: Path,
    job_yaml: Path,
    job_name: str,
    ref_name: str,
    refs_csv: Path,
) -> Path:
    """
    Create RECORD.md if missing. This is the human-facing, append-only
    log of commands you run against this dataset, plus a scratch area.
    """
    path = dataset_dir / "RECORD.md"
    if path.exists():
        return path
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S %Z")
    text = (
        dedent(
            f"""
        # Permuter RECORD

        _Dataset created {ts} (UTC). This file is a lightweight, human‑editable record._

        **Job**: {job_name}
        **Reference**: {ref_name}
        **Job YAML**: {job_yaml}
        **Refs CSV**: {refs_csv}

        ## Commands
        (CLI invocations will be appended below.)

        ## Notes
        (Use this area as a scratch pad for observations.)
        """
        ).strip()
        + "\n"
    )
    path.write_text(text, encoding="utf-8")
    return path


def append_record_md(dataset_dir: Path, section: str, command: str) -> Path:
    path = dataset_dir / "RECORD.md"
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S %Z")
    entry = f"\n### {section} · {ts}\n\n```\n{command}\n```\n"
    with path.open("a", encoding="utf-8") as fh:
        fh.write(entry)
    return path
