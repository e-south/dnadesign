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


def atomic_write_parquet(df: pd.DataFrame, path: Path) -> None:
    """
    Write a Parquet file atomically using the pyarrow engine.
    No fallbacks: if pyarrow is missing or the write fails, raise with a clear message.
    """
    path = Path(path).expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".parquet.tmp")

    try:
        # Explicit hard requirement — see also read_parquet below.
        import pyarrow  # type: ignore  # noqa: F401
    except Exception as e:
        raise RuntimeError(
            "Permuter requires 'pyarrow' to write Parquet files. Install it with: pip install pyarrow"
        ) from e

    try:
        df.to_parquet(tmp, index=False, engine="pyarrow")
        tmp.replace(path)  # atomic on POSIX; same-device rename
    finally:
        # If anything went wrong before replace() or after, try to clean up the tmp.
        try:
            if tmp.exists():
                tmp.unlink(missing_ok=True)
        except Exception:
            pass


def read_parquet(path: Path) -> pd.DataFrame:
    """
    Read a Parquet file using the pyarrow engine (no fallbacks).
    """
    p = Path(path).expanduser().resolve()
    if not p.exists():
        raise FileNotFoundError(f"Parquet file not found: {p}")
    try:
        import pyarrow  # type: ignore  # noqa: F401
    except Exception as e:
        raise RuntimeError(
            "Permuter requires 'pyarrow' to read Parquet files. Install it with: pip install pyarrow"
        ) from e
    return pd.read_parquet(p, engine="pyarrow")


# --- reference sequence sidecar ---------------------------------------------


def write_ref_fasta(dataset_dir: Path, ref_name: str, sequence: str) -> Path:
    fasta = dataset_dir / "REF.fa"
    lines = [f">{ref_name}\n"]
    # wrap 80 cols
    seq = sequence.strip()
    lines += [seq[i : i + 80] + "\n" for i in range(0, len(seq), 80)]
    fasta.write_text("".join(lines), encoding="utf-8")
    return fasta


def write_ref_protein_fasta(dataset_dir: Path, ref_name: str, aa_sequence: str) -> Path:
    """Authoritative reference protein (one‑line FASTA)."""
    fasta = dataset_dir / "REF_AA.fa"
    seq = aa_sequence.strip()
    lines = [f">{ref_name}\n"]
    lines += [seq[i : i + 80] + "\n" for i in range(0, len(seq), 80)]
    fasta.write_text("".join(lines), encoding="utf-8")
    return fasta


def read_ref_protein_fasta(dataset_dir: Path) -> tuple[str, str] | None:
    p = dataset_dir / "REF_AA.fa"
    if not p.exists():
        return None
    name, seq = "", []
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


def append_record_md(dataset_dir: Path, action: str, command: str) -> Path:
    """
    Minimal, human-friendly record entry:
      ### <action> · <timestamp>
      <command in a code fence>
    """
    path = dataset_dir / "RECORD.md"
    if not path.exists():
        # fall back to a minimal header if dataset was created manually
        ts = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S %Z")
        path.write_text(
            f"# Permuter RECORD\n\n_Dataset created {ts} (UTC). This file is a lightweight, human-editable record._\n",
            encoding="utf-8",
        )
    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S %Z")
    entry = f"\n### {action} · {now}\n\n```\n{command}\n```\n"
    with path.open("a", encoding="utf-8") as fh:
        fh.write(entry)
    return path


def append_record_event(
    dataset_dir: Path,
    section: str,
    lines: list[str] | tuple[str, ...] = (),
    command: str | None = None,
) -> Path:
    """
    Append a single consolidated entry to RECORD.md:
      ## SECTION · timestamp
      - key: value
      ...
      ```bash
      <command>
      ```
    """
    path = dataset_dir / "RECORD.md"
    if not path.exists():
        # create a minimal header if user created dataset manually
        ts = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S %Z")
        text = f"# Permuter RECORD\n\n_Dataset created {ts} (UTC)._"
        path.write_text(text + "\n", encoding="utf-8")
    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S %Z")
    body = "\n".join(f"- {ln}" for ln in (lines or []))
    cmd = f"\n\n```bash\n{command}\n```\n" if command else "\n"
    entry = f"\n## {section} · {now}\n\n{body}{cmd}"
    with path.open("a", encoding="utf-8") as fh:
        fh.write(entry)
    return path


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
