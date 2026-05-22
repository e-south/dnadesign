"""
Training-label resolution and preflight checks for an OPAL round.
"""

from __future__ import annotations

import numpy as np

from ....core.leakage import assert_no_leakage_violations, build_shared_label_source_contamination_report
from ....core.utils import OpalError
from ....storage.label_sources import CampaignHistoryLabelSource, label_source_from_config
from ...preflight import preflight_run
from ...round_plan import plan_round
from ..contracts import RoundInputs, TrainingBundle
from .telemetry import log


def stage_training(inputs: RoundInputs) -> TrainingBundle:
    cfg = inputs.cfg
    req = inputs.req
    store = inputs.store
    df = inputs.df
    label_source = label_source_from_config(cfg, store)
    uses_campaign_history = isinstance(label_source, CampaignHistoryLabelSource)

    rep = preflight_run(
        store,
        df,
        int(req.as_of_round),
        cfg.safety.fail_on_mixed_biotype_or_alphabet,
        auto_backfill=False,
        check_manual_attach=uses_campaign_history,
        x_dim_override=req.x_dim_override,
    )
    if uses_campaign_history and getattr(rep, "manual_attach_count", 0):
        raise OpalError(
            f"Detected {rep.manual_attach_count} labels in '{store.y_col}' without label_hist. "
            "Run `opal ingest-y` (preferred) or explicitly reconcile the current Y column with "
            "`opal label-hist attach-from-y`."
        )
    assert_no_leakage_violations(build_shared_label_source_contamination_report(cfg=cfg, store=store, df=df))
    label_source.validate(df)

    plan = plan_round(
        store,
        df,
        cfg,
        int(req.as_of_round),
        warnings=list(rep.warnings or []),
        label_source=label_source,
    )
    train_df = plan.training_df
    if train_df.empty:
        raise OpalError(f"No labels ≤ round {req.as_of_round} for training.")

    train_ids = train_df["id"].astype(str).tolist()
    Y_train = np.stack(train_df["y"].map(lambda v: np.asarray(v, dtype=float)).to_list(), axis=0)
    R_train = train_df["r"].astype(int).to_numpy()
    y_dim = int(Y_train.shape[1])
    exp_len = cfg.data.y_expected_length
    if exp_len is not None and y_dim != int(exp_len):
        raise OpalError(f"Y length mismatch: expected {int(exp_len)}, got {y_dim}")

    log(
        req.verbose,
        f"[labels] as_of_round={int(req.as_of_round)} | n_train={len(train_df)} | y_dim={y_dim} | "
        f"policy(cumulative={bool(plan.training_policy.get('cumulative_training', True))}, "
        f"dedup={plan.training_dedup_policy})",
    )
    return TrainingBundle(
        rep=rep,
        plan=plan,
        train_df=train_df,
        train_ids=train_ids,
        Y_train=Y_train,
        R_train=R_train,
        y_dim=y_dim,
    )
