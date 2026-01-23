# ABOUTME: Resolves canonical campaign workspace paths and output locations.
# ABOUTME: Centralizes layout for outputs, ledgers, and per-round artifacts.
"""
--------------------------------------------------------------------------------
<dnadesign project>
src/dnadesign/opal/src/storage/workspace.py

Module Author(s): Eric J. South
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
        return cls(
            config_path=Path(config_path).resolve(),
            workdir=Path(cfg.campaign.workdir).resolve(),
        )

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
        return self.outputs_dir / "ledger"

    @property
    def ledger_predictions_dir(self) -> Path:
        return self.ledger_dir / "predictions"

    @property
    def ledger_runs_path(self) -> Path:
        return self.ledger_dir / "runs.parquet"

    @property
    def ledger_labels_path(self) -> Path:
        return self.ledger_dir / "labels.parquet"

    # --- Per-round ---
    @property
    def rounds_dir(self) -> Path:
        return self.outputs_dir / "rounds"

    def round_dir(self, round_index: int) -> Path:
        return self.rounds_dir / f"round_{int(round_index)}"
