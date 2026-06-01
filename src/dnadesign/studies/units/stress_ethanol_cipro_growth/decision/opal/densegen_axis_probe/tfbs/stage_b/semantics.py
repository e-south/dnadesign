"""Controlled vocabulary and identity helpers for TFBS learnability Stage B."""

from __future__ import annotations

from dataclasses import dataclass

from ...core.constants import SCRATCH_DATASET

TFBS_STAGE_B_PROBE_FAMILY = "densegen_tfbs_learnability_probe_v1"
TFBS_STAGE_B_STAGE = "B"
TFBS_STAGE_B_SCOPE = "stage_b_sentinel_initial"
TFBS_STAGE_B_SPLIT_ID = "random_id"
TFBS_STAGE_B_ORACLE_ROLES = ("positive", "matched_null")
TFBS_STAGE_B_LABEL_SOURCE_KIND = "usr_sidecar"
TFBS_STAGE_B_RETENTION_MODE = "production_review"
TFBS_STAGE_B_PREDICTION_LEDGER_RETENTION = "latest_full_plus_selected_history"
TFBS_STAGE_B_PLOT_TIDY_RETENTION = "compact"
TFBS_STAGE_B_MODEL_ARTIFACT_RETENTION = "latest"
TFBS_STAGE_B_TABULAR_FORMAT = "parquet_zstd"
TFBS_STAGE_B_SELECTION_TIE_HANDLING_MODES = ("competition_rank", "dense_rank", "ordinal")
TFBS_STAGE_B_DEFAULT_TIE_HANDLING = "ordinal"
TFBS_STAGE_B_EXACT_BUDGET_TIE_HANDLING = "ordinal"


@dataclass(frozen=True)
class TfbsStageBRunIdentity:
    """Stable identifiers for one Stage B label/oracle/split/seed campaign."""

    label_name: str
    oracle_role: str
    split_id: str
    seed: int

    def __post_init__(self) -> None:
        validate_stage_b_oracle_role(self.oracle_role)
        validate_stage_b_split_id(self.split_id)
        if int(self.seed) < 0:
            raise ValueError("Stage B seed must be non-negative")

    @property
    def run_key(self) -> str:
        return f"tfbs_{self.label_name}_{self.oracle_role}_{self.split_id}_seed{int(self.seed)}"

    @property
    def campaign_slug(self) -> str:
        return slug_token(f"tfbs_v1_{self.label_name}_{self.oracle_role}_{self.split_id}_seed{int(self.seed)}")


def validate_stage_b_oracle_role(role: str) -> str:
    value = str(role)
    if value not in TFBS_STAGE_B_ORACLE_ROLES:
        raise ValueError(f"unsupported Stage B oracle role: {value!r}")
    return value


def validate_stage_b_split_id(split_id: str) -> str:
    value = str(split_id)
    if value != TFBS_STAGE_B_SPLIT_ID:
        raise ValueError(f"Stage B v1 supports split_id={TFBS_STAGE_B_SPLIT_ID!r} only")
    return value


def validate_stage_b_tie_handling(tie_handling: str) -> str:
    value = str(tie_handling).strip().lower()
    if value not in TFBS_STAGE_B_SELECTION_TIE_HANDLING_MODES:
        allowed = "|".join(TFBS_STAGE_B_SELECTION_TIE_HANDLING_MODES)
        raise ValueError(f"Stage B selection tie_handling must be one of {allowed}; got {tie_handling!r}")
    return value


def stage_b_selection_budget_mode(*, tie_handling: str) -> str:
    validated = validate_stage_b_tie_handling(tie_handling)
    if validated == TFBS_STAGE_B_EXACT_BUDGET_TIE_HANDLING:
        return "exact_top_k"
    return "tie_expanding_rank"


def stage_b_dataset_id(*, split_id: str, seed: int) -> str:
    validate_stage_b_split_id(split_id)
    if int(seed) < 0:
        raise ValueError("Stage B seed must be non-negative")
    return f"{SCRATCH_DATASET}_tfbs_{slug_token(split_id)}_seed{int(seed)}"


def slug_token(value: str) -> str:
    out = []
    for char in str(value):
        lower = char.lower()
        if lower.isalnum():
            out.append(lower)
        elif lower in {"_", "-"}:
            out.append(lower)
        else:
            out.append("_")
    token = "".join(out).strip("_-")
    while "__" in token:
        token = token.replace("__", "_")
    if not token:
        raise ValueError(f"could not build slug token from {value!r}")
    return token
