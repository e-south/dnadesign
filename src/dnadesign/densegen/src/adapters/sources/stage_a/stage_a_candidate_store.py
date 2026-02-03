"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/densegen/src/adapters/sources/stage_a/stage_a_candidate_store.py

Stage-A debug candidate record persistence helpers.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from pathlib import Path

from .stage_a_paths import safe_label


def write_candidate_records(
    records: list[dict],
    *,
    debug_output_dir: Path,
    debug_label: str,
    motif_id: str,
    motif_hash: str | None = None,
) -> Path:
    suffix = ""
    if motif_hash:
        suffix = f"__{safe_label(motif_hash[:10])}"
    label = f"{safe_label(debug_label or motif_id)}{suffix}"
    debug_output_dir.mkdir(parents=True, exist_ok=True)
    path = debug_output_dir / f"candidates__{label}.parquet"
    import pandas as pd

    df = pd.DataFrame(records)
    if path.exists():
        try:
            existing = pd.read_parquet(path)
            if "candidate_id" not in existing.columns or "candidate_id" not in df.columns:
                raise ValueError(
                    f"Candidate append requires candidate_id in {path}. "
                    "Clear outputs/pools/candidates or use --fresh to reset."
                )
            if set(existing.columns) != set(df.columns):
                raise ValueError(
                    f"Candidate schema mismatch for {path}. Clear outputs/pools/candidates or use --fresh to reset."
                )
            df = df[existing.columns]
            df = pd.concat([existing, df], ignore_index=True)
            df = df.drop_duplicates(subset=["candidate_id"], keep="last")
        except Exception as exc:
            if isinstance(exc, ValueError):
                raise
            raise RuntimeError(f"Failed to append candidate records to {path}") from exc
    df.to_parquet(path, index=False)
    return path


def write_fimo_debug_tsv(
    lines: list[str],
    *,
    debug_path: Path,
) -> Path:
    if not lines:
        raise ValueError("FIMO debug TSV lines are required.")
    debug_path.parent.mkdir(parents=True, exist_ok=True)
    debug_path.write_text("\n".join(lines) + "\n")
    return debug_path
