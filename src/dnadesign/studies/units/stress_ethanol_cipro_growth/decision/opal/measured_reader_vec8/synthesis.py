"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/studies/units/stress_ethanol_cipro_growth/decision/opal/measured_reader_vec8/synthesis.py

Loads synthesis handoff mappings for measured Reader vec8 staging.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from .constants import BATCH0_HANDOFF_ID, STRESS_CAMPAIGN_SLUGS
from .contracts import MeasuredReaderVec8Error


def load_batch0_manifest_map(repo_root: Path) -> pd.DataFrame:
    """Return the exact pDual-10 reader design to OPAL candidate mapping."""

    root = Path(repo_root).expanduser().resolve()
    rows: list[pd.DataFrame] = []
    for slug in STRESS_CAMPAIGN_SLUGS:
        manifest_dir = (
            root
            / "src"
            / "dnadesign"
            / "opal"
            / "campaigns"
            / slug
            / "outputs"
            / "synthesis_handoff"
            / BATCH0_HANDOFF_ID
        )
        files = sorted(manifest_dir.glob("*synthesis_manifest.csv"))
        if len(files) != 1:
            raise MeasuredReaderVec8Error(
                f"Expected one synthesis manifest for {slug}; found {len(files)} under {manifest_dir}."
            )
        frame = pd.read_csv(files[0])
        required = {"id", "synthesis_name", "core_sequence", "campaign_slug", "validation_status"}
        missing = sorted(required - set(frame.columns))
        if missing:
            raise MeasuredReaderVec8Error(f"Synthesis manifest {files[0]} missing columns: {missing}")
        if not frame["validation_status"].astype(str).eq("pass").all():
            raise MeasuredReaderVec8Error(f"Synthesis manifest {files[0]} contains non-pass validation rows.")
        frame = frame.copy()
        frame["source_campaign_slug"] = frame["campaign_slug"].astype(str)
        frame["campaign_slug"] = slug
        frame["reader_design_id"] = "pDual-10-" + frame["synthesis_name"].astype(str)
        rows.append(frame)
    out = pd.concat(rows, ignore_index=True)
    duplicates = out.loc[out["reader_design_id"].duplicated(keep=False), "reader_design_id"].astype(str).unique()
    if len(duplicates):
        raise MeasuredReaderVec8Error(
            f"Duplicate reader design IDs in batch0 synthesis manifests: {sorted(duplicates)}"
        )
    return out
