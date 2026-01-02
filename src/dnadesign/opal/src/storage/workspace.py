"""
--------------------------------------------------------------------------------
<dnadesign project>
src/dnadesign/opal/src/storage/workspace.py

CampaignWorkspace centralizes all campaign paths to keep IO consistent and
decoupled from CLI or application logic.

Module Author(s): Eric J. South
Dunlop Lab
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from ..config.types import RootConfig


@dataclass(frozen=True)
class CampaignWorkspace:
    """Resolved campaign workspace paths (all absolute)."""

    config_path: Path
    workdir: Path

    @classmethod
    def from_config(cls, cfg: RootConfig, config_path: Path) -> "CampaignWorkspace":
        return cls(config_path=Path(config_path).resolve(), workdir=Path(cfg.campaign.workdir).resolve())

    @property
    def outputs_dir(self) -> Path:
        return self.workdir / "outputs"

    @property
    def inputs_dir(self) -> Path:
        return self.workdir / "inputs"

    @property
    def state_path(self) -> Path:
        return self.workdir / "state.json"

    @property
    def marker_path(self) -> Path:
        return self.workdir / ".opal" / "config"

    # --- Ledger sinks ---
    @property
    def ledger_dir(self) -> Path:
        return self.outputs_dir

    @property
    def ledger_predictions_dir(self) -> Path:
        return self.outputs_dir / "ledger.predictions"

    @property
    def ledger_runs_path(self) -> Path:
        return self.outputs_dir / "ledger.runs.parquet"

    @property
    def ledger_labels_path(self) -> Path:
        return self.outputs_dir / "ledger.labels.parquet"

    # --- Per-round ---
    def round_dir(self, round_index: int) -> Path:
        return self.outputs_dir / f"round_{int(round_index)}"
