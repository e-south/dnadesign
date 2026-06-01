"""Stage B TFBS artifact layout contract."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from .semantics import stage_b_dataset_id


@dataclass(frozen=True)
class TfbsStageBLayout:
    """Filesystem ontology for one Stage B TFBS campaign-set materialization."""

    out_dir: Path
    split_id: str
    seed: int

    @property
    def manifests_dir(self) -> Path:
        return self.out_dir / "manifests"

    @property
    def config_manifest_path(self) -> Path:
        return self.manifests_dir / "stage_b_sentinel_config_manifest.json"

    @property
    def collection_manifest_path(self) -> Path:
        return self.manifests_dir / "stage_b_sentinel_campaign_collection.json"

    @property
    def validation_reports_dir(self) -> Path:
        return self.out_dir / "validation_reports"

    @property
    def scratch_usr_dir(self) -> Path:
        return self.out_dir / "scratch_usr"

    @property
    def dataset(self) -> str:
        return stage_b_dataset_id(split_id=self.split_id, seed=self.seed)

    @property
    def dataset_dir(self) -> Path:
        return self.scratch_usr_dir / self.dataset

    @property
    def records_path(self) -> Path:
        return self.dataset_dir / "records.parquet"

    @property
    def candidate_scope_path(self) -> Path:
        return self.dataset_dir / "candidate_scope_ids.parquet"

    def campaign_workdir(self, run_key: str) -> Path:
        return self.out_dir / "campaigns" / run_key

    def campaign_config_path(self, run_key: str) -> Path:
        return self.campaign_workdir(run_key) / "configs" / "campaign.yaml"

    def campaign_plot_config_path(self, run_key: str) -> Path:
        return self.campaign_workdir(run_key) / "configs" / "plots.yaml"

    def initial_label_input_path(self, run_key: str) -> Path:
        return self.campaign_workdir(run_key) / "inputs" / "r0" / "labels-b0.parquet"

    def sidecar_relative_path(self, run_key: str) -> str:
        return f"_opal/{run_key}/observed_labels.parquet"
