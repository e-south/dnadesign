"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/studies/units/stress_ethanol_cipro_growth/decision/opal/densegen_axis_probe/tfbs/stage_b/seed.py

Initial-label seed policies for DenseGen TFBS Stage B probe campaigns.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import hashlib
from typing import TYPE_CHECKING, Sequence

if TYPE_CHECKING:
    import numpy as np
    import pandas as pd

TFBS_STAGE_B_INITIAL_SEED_POLICY_LABEL_VALUE_STRATIFIED_RANDOM = "label_value_stratified_random"
TFBS_STAGE_B_INITIAL_SEED_POLICY_UNIFORM_RANDOM = "uniform_random"
TFBS_STAGE_B_SHARED_INITIAL_SEED_CONTEXT_VERSION = "tfbs_stage_b_shared_initial_seed_v1"
TFBS_STAGE_B_INITIAL_SEED_POLICIES = (
    TFBS_STAGE_B_INITIAL_SEED_POLICY_LABEL_VALUE_STRATIFIED_RANDOM,
    TFBS_STAGE_B_INITIAL_SEED_POLICY_UNIFORM_RANDOM,
)


def validate_tfbs_stage_b_initial_seed_policy(value: str) -> str:
    """Return a supported Stage B initial seed policy or fail fast."""

    policy = str(value)
    if policy not in TFBS_STAGE_B_INITIAL_SEED_POLICIES:
        raise ValueError(
            "Unsupported Stage B initial seed policy "
            f"{policy!r}; expected one of {list(TFBS_STAGE_B_INITIAL_SEED_POLICIES)}"
        )
    return policy


def tfbs_stage_b_shared_initial_seed_context(*, label_name: str, split_id: str, seed: int) -> str:
    """Return the shared seed context used for one positive/null campaign pair."""

    label_text = str(label_name).strip()
    split_text = str(split_id).strip()
    if not label_text:
        raise ValueError("Stage B shared initial seed context requires label_name")
    if not split_text:
        raise ValueError("Stage B shared initial seed context requires split_id")
    return f"{TFBS_STAGE_B_SHARED_INITIAL_SEED_CONTEXT_VERSION}:label={label_text}:split={split_text}:seed={int(seed)}"


def select_tfbs_stage_b_initial_ids(
    frame: pd.DataFrame,
    *,
    label_name: str,
    initial_label_count: int,
    seed: int,
    policy: str,
    seed_context: str = "",
) -> tuple[str, ...]:
    """Select initial labeled IDs for a synthetic TFBS learnability probe campaign."""

    policy = validate_tfbs_stage_b_initial_seed_policy(policy)
    prepared = _seed_frame(frame, label_name=label_name)
    count = int(initial_label_count)
    if count <= 0:
        raise ValueError("Stage B initial_label_count must be positive")
    if count > len(prepared):
        raise ValueError(f"Stage B initial_label_count={count} exceeds row universe size {len(prepared)}")
    if policy == TFBS_STAGE_B_INITIAL_SEED_POLICY_UNIFORM_RANDOM:
        import numpy as np

        rng = np.random.default_rng(int(seed))
        return _uniform_random_ids(prepared["id"].tolist(), count=count, rng=rng)
    if policy == TFBS_STAGE_B_INITIAL_SEED_POLICY_LABEL_VALUE_STRATIFIED_RANDOM:
        import numpy as np

        rng = np.random.default_rng(_stable_seed(seed=seed, context=f"{policy}:{label_name}:{seed_context}"))
        return _label_value_stratified_random_ids(prepared, count=count, rng=rng)
    raise AssertionError(f"validated unsupported Stage B initial seed policy: {policy}")


def select_tfbs_stage_b_paired_initial_ids(
    positive_frame: pd.DataFrame,
    control_frame: pd.DataFrame,
    *,
    label_name: str,
    initial_label_count: int,
    seed: int,
    policy: str,
    seed_context: str = "",
) -> tuple[str, ...]:
    """Select shared initial IDs that are non-degenerate under positive and control labels."""

    policy = validate_tfbs_stage_b_initial_seed_policy(policy)
    positive = _seed_frame(positive_frame, label_name=label_name).rename(columns={label_name: "__positive__"})
    control = _seed_frame(control_frame, label_name=label_name).rename(columns={label_name: "__control__"})
    merged = positive.merge(control, on="id", how="inner", validate="one_to_one")
    if len(merged) != len(positive) or len(merged) != len(control):
        raise ValueError("Stage B paired initial seed selection requires matching positive/control candidate IDs")
    count = int(initial_label_count)
    if count <= 0:
        raise ValueError("Stage B initial_label_count must be positive")
    if count > len(merged):
        raise ValueError(f"Stage B initial_label_count={count} exceeds paired candidate scope size {len(merged)}")
    for attempt in range(256):
        ids = _select_candidate_ids_for_attempt(
            merged,
            label_name=label_name,
            count=count,
            seed=seed,
            policy=policy,
            seed_context=seed_context,
            attempt=attempt,
        )
        if _selected_has_multiple_values(merged, ids=ids, column="__positive__") and _selected_has_multiple_values(
            merged,
            ids=ids,
            column="__control__",
        ):
            return ids
    raise ValueError(
        "Stage B paired initial seed selection could not find a non-degenerate shared batch "
        f"for {label_name} under both positive and control labels"
    )


def _seed_frame(frame: pd.DataFrame, *, label_name: str) -> pd.DataFrame:
    import pandas as pd

    missing = sorted({"id", label_name} - set(frame.columns))
    if missing:
        raise ValueError(f"Stage B initial seed frame missing column(s): {missing}")
    out = frame.loc[:, ["id", label_name]].copy()
    out["id"] = out["id"].astype(str)
    if out["id"].isna().any():
        raise ValueError("Stage B initial seed ids must not be null")
    if out["id"].duplicated().any():
        duplicates = out.loc[out["id"].duplicated(), "id"].head(10).tolist()
        raise ValueError(f"Stage B initial seed frame contains duplicate id(s): {duplicates}")
    out[label_name] = pd.to_numeric(out[label_name], errors="raise")
    if out[label_name].isna().any():
        raise ValueError(f"Stage B initial seed label {label_name!r} must not contain nulls")
    return out.sort_values(["id"]).reset_index(drop=True)


def _uniform_random_ids(ids: Sequence[str], *, count: int, rng: np.random.Generator) -> tuple[str, ...]:
    unique_ids = sorted(set(map(str, ids)))
    if len(unique_ids) != len(ids):
        raise ValueError("Stage B uniform initial seed sampling requires unique ids")
    selected_indices = rng.choice(len(unique_ids), size=count, replace=False)
    return tuple(unique_ids[int(index)] for index in sorted(selected_indices.tolist()))


def _select_candidate_ids_for_attempt(
    frame: pd.DataFrame,
    *,
    label_name: str,
    count: int,
    seed: int,
    policy: str,
    seed_context: str,
    attempt: int,
) -> tuple[str, ...]:
    import numpy as np

    context = f"{policy}:{label_name}:{seed_context}:paired_attempt={int(attempt)}"
    rng = np.random.default_rng(_stable_seed(seed=seed, context=context))
    if policy == TFBS_STAGE_B_INITIAL_SEED_POLICY_UNIFORM_RANDOM:
        return _uniform_random_ids(frame["id"].tolist(), count=count, rng=rng)
    if policy == TFBS_STAGE_B_INITIAL_SEED_POLICY_LABEL_VALUE_STRATIFIED_RANDOM:
        prepared = frame.loc[:, ["id", "__positive__"]].rename(columns={"__positive__": label_name})
        return _label_value_stratified_random_ids(prepared, count=count, rng=rng)
    raise AssertionError(f"validated unsupported Stage B initial seed policy: {policy}")


def _selected_has_multiple_values(frame: pd.DataFrame, *, ids: Sequence[str], column: str) -> bool:
    wanted = set(map(str, ids))
    selected = frame.loc[frame["id"].astype(str).isin(wanted), column]
    return selected.nunique(dropna=False) >= 2


def _label_value_stratified_random_ids(
    frame: pd.DataFrame,
    *,
    count: int,
    rng: np.random.Generator,
) -> tuple[str, ...]:
    import numpy as np

    label_name = frame.columns.difference(["id"]).tolist()[0]
    ordered = frame.sort_values([label_name, "id"]).reset_index(drop=True)
    selected: list[str] = []
    for stratum_index in range(count):
        start = int(np.floor(stratum_index * len(ordered) / count))
        stop = int(np.floor((stratum_index + 1) * len(ordered) / count))
        if stop <= start:
            stop = start + 1
        stratum = ordered.iloc[start:stop]
        selected_row_index = int(rng.choice(stratum.index.to_numpy(dtype=int), size=1)[0])
        selected.append(str(ordered.loc[selected_row_index, "id"]))
    if len(set(selected)) != len(selected):
        raise AssertionError("Stage B stratified initial seed sampling produced duplicate ids")
    return tuple(selected)


def _stable_seed(*, seed: int, context: str) -> int:
    digest = hashlib.sha256(f"{int(seed)}:{context}".encode("utf-8")).digest()
    return int.from_bytes(digest[:8], byteorder="big", signed=False)
