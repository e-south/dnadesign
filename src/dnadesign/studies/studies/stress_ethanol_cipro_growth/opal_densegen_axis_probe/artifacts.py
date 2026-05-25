"""Study-owned DenseGen axis OPAL probe package."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .constants import DEFAULT_TOP_K, SCRATCH_DATASET


@dataclass(frozen=True)
class AxisLabel:
    id: str
    axis_class: str | None
    logic4: list[int] | None
    effect4: list[int] | None
    vec8: list[int] | None
    quality_flag: str
    lexA_count: int = 0
    cpxR_count: int = 0
    baeR_count: int = 0
    background_count: int = 0
    cipro_axis: bool = False
    ethanol_axis: bool = False
    densegen_plan_class: str | None = None
    sigma35_variant: str | None = None
    densegen_plan: str | None = None
    expected_axis_class_from_plan: str | None = None


@dataclass(frozen=True)
class RunSpec:
    campaign_key: str
    oracle_id: str
    split_id: str
    run_key: str
    target_class: str
    workdir: Path
    config_path: Path
    label_input_path: Path
    sidecar_path: Path
    selection_k: int = DEFAULT_TOP_K
    seed: int = 7
    label_family_id: str = "sfxi_axis_vec8"
    max_x_matrix_gib: float | None = None
    score_batch_size: int | None = None


@dataclass(frozen=True)
class ProbePlan:
    run_root: Path
    initial_label_count: int
    selection_k: int
    seed: int
    rounds: int
    gate: str | None
    splits: tuple[str, ...]
    apply: bool
    max_x_matrix_gib: float | None = None
    score_batch_size: int | None = None
    stop_after: str = "status"
    suite_id: str = "densegen_motif_qa_k12_s3_v1"
    suite_seeds: tuple[int, ...] = (7, 17, 29)
    active_label_family: str = "sfxi_axis_vec8"
    passive_label_families: tuple[str, ...] = ("tf_family_presence", "tf_family_count", "densegen_plan_class")
    runs: list[RunSpec] = field(default_factory=list)
    commands: list[list[str]] = field(default_factory=list)


@dataclass(frozen=True)
class RunRootAudit:
    run_root: Path
    exists: bool
    decision: str | None
    status: str
    labels_present: bool
    splits_present: bool
    metrics_present: bool
    decision_present: bool
    scratch_records_present: bool
    planned_campaign_count: int
    shared_sidecar_present: bool
    problems: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "run_root": str(self.run_root),
            "exists": self.exists,
            "decision": self.decision,
            "status": self.status,
            "labels_present": self.labels_present,
            "splits_present": self.splits_present,
            "metrics_present": self.metrics_present,
            "decision_present": self.decision_present,
            "scratch_records_present": self.scratch_records_present,
            "planned_campaign_count": self.planned_campaign_count,
            "shared_sidecar_present": self.shared_sidecar_present,
            "problems": list(self.problems),
        }


@dataclass(frozen=True)
class ProbeArtifactLayout:
    run_root: Path

    @property
    def labels_dir(self) -> Path:
        return self.run_root / "labels"

    @property
    def densegen_labels_path(self) -> Path:
        return self.labels_dir / "densegen_part_axis_vec8.parquet"

    @property
    def null_labels_path(self) -> Path:
        return self.labels_dir / "permuted_densegen_part_axis_vec8.parquet"

    @property
    def label_family_manifest_path(self) -> Path:
        return self.labels_dir / "label_families.json"

    @property
    def null_provenance_path(self) -> Path:
        return self.labels_dir / "null_provenance.json"

    @property
    def splits_dir(self) -> Path:
        return self.run_root / "splits"

    @property
    def split_metadata_path(self) -> Path:
        return self.splits_dir / "split_metadata.json"

    @property
    def probe_plan_path(self) -> Path:
        return self.run_root / "probe_plan.json"

    @property
    def suite_manifest_path(self) -> Path:
        return self.run_root / "probe_suite.json"

    def train_ids_path(self, split_id: str) -> Path:
        return self.splits_dir / f"{split_id}_train_ids.parquet"

    def eval_ids_path(self, split_id: str) -> Path:
        return self.splits_dir / f"{split_id}_eval_ids.parquet"

    @property
    def reports_dir(self) -> Path:
        return self.run_root / "reports"

    @property
    def metrics_path(self) -> Path:
        return self.reports_dir / "metrics.json"

    @property
    def decision_path(self) -> Path:
        return self.reports_dir / "decision.md"

    @property
    def status_path(self) -> Path:
        return self.reports_dir / "status.json"

    @property
    def run_manifest_path(self) -> Path:
        return self.reports_dir / "run_manifest.json"

    @property
    def review_path(self) -> Path:
        return self.reports_dir / "review.md"

    @property
    def review_index_path(self) -> Path:
        return self.reports_dir / "index.html"

    @property
    def review_manifest_path(self) -> Path:
        return self.reports_dir / "review_manifest.json"

    @property
    def review_plots_dir(self) -> Path:
        return self.reports_dir / "plots"

    @property
    def scratch_campaigns_dir(self) -> Path:
        return self.run_root / "scratch_campaigns"

    @property
    def scratch_usr_dir(self) -> Path:
        return self.run_root / "scratch_usr"

    @property
    def scratch_dataset_dir(self) -> Path:
        return self.scratch_usr_dir / SCRATCH_DATASET

    def split_dataset(self, split_id: str) -> str:
        return f"{SCRATCH_DATASET}_{str(split_id)}"

    def split_dataset_dir(self, split_id: str) -> Path:
        return self.scratch_usr_dir / self.split_dataset(split_id)

    def split_records_path(self, split_id: str) -> Path:
        return self.split_dataset_dir(split_id) / "records.parquet"

    def campaign_workdir(self, run_key: str) -> Path:
        return self.scratch_campaigns_dir / run_key

    def campaign_config_path(self, run_key: str) -> Path:
        return self.campaign_workdir(run_key) / "configs" / "campaign.yaml"

    def campaign_label_input_path(self, run_key: str, round_index: int = 0) -> Path:
        return self.campaign_workdir(run_key) / "inputs" / f"r{int(round_index)}" / f"vec8-b{int(round_index)}.parquet"

    def campaign_sidecar_path(self, run_key: str, split_id: str) -> Path:
        return self.split_dataset_dir(split_id) / "_opal" / run_key / "observed_labels.parquet"
