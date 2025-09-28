"""
--------------------------------------------------------------------------------
<dnadesign project>
src/dnadesign/opal/src/cli/commands/run.py

Module Author(s): Eric J. South
Dunlop Lab
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
import typer

from ...artifacts import (
    ArtifactPaths,
    append_events,
    append_round_log_event,
    events_path,
    round_dir,
    write_objective_meta,
    write_round_ctx,
    write_selection_csv,
)
from ...config import load_config
from ...locks import CampaignLock
from ...preflight import preflight_run
from ...registries.models import get_model
from ...registries.objectives import get_objective
from ...registries.selections import get_selection, normalize_selection_result
from ...round_context import RoundContext
from ...state import CampaignState, RoundEntry
from ...utils import (
    ExitCodes,
    OpalError,
    ensure_dir,
    file_sha256,
    now_iso,
)
from ..registry import cli_command
from ._common import internal_error, json_out, resolve_config_path, store_from_cfg


def _pct(arr: np.ndarray, ps: tuple[int, int, int] = (10, 50, 90)) -> List[float]:
    arr = np.asarray(arr, dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return [float("nan")] * len(ps)
    return [float(np.percentile(arr, p)) for p in ps]


def _echo_and_capture(lines: list[str], s: str) -> None:
    lines.append(s)
    typer.echo(s, err=True)


@cli_command(
    "run",
    help="Train on labels ≤ round, score, select top-k, write artifacts & records.",
)
def cmd_run(
    config: Path = typer.Option(
        None, "--config", "-c", envvar="OPAL_CONFIG", help="campaign.yaml"
    ),
    round: int = typer.Option(..., "--round", "-r"),
    k: Optional[int] = typer.Option(
        None, "--k", "-k", help="Top-k (default from YAML)"
    ),
    resume: bool = typer.Option(
        False, "--resume", help="Allow overwrites for this round"
    ),
    force: bool = typer.Option(False, "--force", help="Force write-backs if needed"),
    score_batch_size: Optional[int] = typer.Option(
        None, "--score-batch-size", help="Override batch size"
    ),
    no_backfill: bool = typer.Option(
        False,
        "--no-backfill",
        help="Do not auto-backfill legacy Y values into label history during preflight.",
    ),
) -> None:
    log_lines: list[str] = []
    try:
        cfg = load_config(resolve_config_path(config))
        _k_from_cli = k is not None
        if k is None:
            k = cfg.selection.top_k_default
        if k is None or int(k) <= 0:
            raise OpalError("--k must be a positive integer (or omit it to use YAML).")
        k = int(k)

        workdir = Path(cfg.campaign.workdir)
        rdir = round_dir(workdir, round)
        ensure_dir(rdir)

        # lock — single writer per campaign
        lock = CampaignLock(workdir)
        with lock:
            store = store_from_cfg(cfg)
            df = store.load()
            _echo_and_capture(
                log_lines, f"[run] campaign={cfg.campaign.slug} round={round}"
            )
            _echo_and_capture(log_lines, f"[data] loaded records: {len(df)} rows")

            # preflight
            _echo_and_capture(log_lines, "[preflight] starting checks…")
            rep = preflight_run(
                store,
                df,
                round,
                cfg.safety.fail_on_mixed_biotype_or_alphabet,
                auto_backfill=(not no_backfill),
            )

            if getattr(rep, "x_dim", None) is not None:
                _echo_and_capture(log_lines, f"[preflight] X dimension: {rep.x_dim}")
            if getattr(rep, "backfill", {}).get("backfilled", 0) > 0:
                _echo_and_capture(
                    log_lines,
                    f"[preflight] backfilled labels: {rep.backfill['backfilled']}",
                )

            # If preflight mutated the table (backfilled), reload so training sees it.
            if getattr(rep, "backfill", {}).get("backfilled", 0) > 0:
                df = store.load()

            # gather training labels ≤ round
            labels_eff = store.training_labels_from_y(df, round)
            if labels_eff.empty:
                raise OpalError("No effective labels to train on.")
            y_vals = labels_eff["y"].tolist()
            y_mat = np.array(
                [
                    (
                        np.array(v, dtype=float)
                        if isinstance(v, list)
                        else np.array([float(v)])
                    )
                    for v in y_vals
                ],
                dtype=float,
            )
            y_dim = int(y_mat.shape[1])
            _echo_and_capture(
                log_lines,
                f"[labels] y_column='{cfg.data.y_column_name}' length={y_dim}",
            )
            _echo_and_capture(
                log_lines,
                f"[labels] effective training examples: {len(labels_eff)} (≤ round {round})",
            )

            # Label history diagnostics
            lh_col = store.label_hist_col()
            rounds_seen: Dict[int, int] = {}
            entries_used = 0
            tr_id_set = set(labels_eff["id"].astype(str).tolist())
            if lh_col in df.columns:
                for _, row in df.iterrows():
                    if str(row["id"]) not in tr_id_set:
                        continue
                    for e in store._normalize_hist_cell(row.get(lh_col)):
                        try:
                            rr = int(e.get("r", 9_999_999))
                            if rr <= int(round):
                                rounds_seen[rr] = rounds_seen.get(rr, 0) + 1
                                entries_used += 1
                        except Exception:
                            continue

            rounds_str = ", ".join(
                f"{rk}: {rv}" for rk, rv in sorted(rounds_seen.items())
            )
            _echo_and_capture(
                log_lines,
                f"[history] {lh_col}: entries_used={entries_used}, rounds_seen={{{rounds_str}}}",
            )
            pol = cfg.training.policy or {}
            _echo_and_capture(
                log_lines,
                "[history] policy: "
                f"cumulative_training={bool(pol.get('cumulative_training', True))}, "
                f"dedup={pol.get('label_cross_round_deduplication_policy','latest_only')}",
            )

            tr_ids = labels_eff["id"].tolist()
            Xtr, _ = store.transform_matrix(df, tr_ids)

            _echo_and_capture(
                log_lines,
                f"[train] n_train={len(tr_ids)} Xtr_shape={tuple(Xtr.shape)} y_dim={y_dim}",
            )

            # candidate universe
            cand = store.candidate_universe(df, round)
            cand_ids = cand["id"].tolist()

            # model
            mdl = get_model(
                cfg.training.models.name,
                cfg.training.models.params,
                cfg.training.target_scaler.__dict__,
            )

            import time

            t0 = time.perf_counter()
            _echo_and_capture(
                log_lines,
                f"[model] {cfg.training.models.name} (params={cfg.training.models.params})",
            )
            # Scaler "mini-card"
            tsc = cfg.training.target_scaler
            _echo_and_capture(
                log_lines,
                "[scaler] target_scaler="
                f"{tsc.method} (center={tsc.center_statistic}, scale={tsc.scale_statistic}, "
                f"min_labels={tsc.minimum_labels_required})",
            )

            _echo_and_capture(log_lines, "[fit] fitting model…")
            metrics = mdl.fit(Xtr, y_mat)
            fit_dt = time.perf_counter() - t0

            # per-target |y| IQR distribution (10/50/90)
            try:
                abs_y = np.abs(y_mat)
                iqr_per_target = np.subtract(
                    np.percentile(abs_y, 75, axis=0),
                    np.percentile(abs_y, 25, axis=0),
                )
                p10, p50, p90 = _pct(iqr_per_target, ps=(10, 50, 90))
                iqr_summary = f"[{p10:.3g},{p50:.3g},{p90:.3g}]"
            except Exception:
                iqr_summary = "[nan,nan,nan]"

            oob_r2_txt = f"{metrics.oob_r2:.3g}" if metrics.oob_r2 is not None else "NA"
            _echo_and_capture(
                log_lines,
                "[fit] done in "
                f"{fit_dt:.2f}s | oob_r2={oob_r2_txt} | per-target_iqr: "
                f"[p10,p50,p90] of |y| = {iqr_summary}",
            )

            # predict in batches
            sbatch = int(score_batch_size or cfg.scoring.score_batch_size)
            if sbatch <= 0:
                raise OpalError("score_batch_size must be > 0")
            preds: list[tuple[str, list[float]]] = []
            stds: list[tuple[str, list[float]]] = []
            t1 = time.perf_counter()
            _echo_and_capture(
                log_lines,
                f"[score] scoring {len(cand_ids)} candidates in chunks of {sbatch}…",
            )
            with typer.progressbar(
                iterable=range(0, len(cand_ids), sbatch),
                label="Scoring",
                file=sys.stderr,
            ) as steps:
                for i in steps:
                    chunk_ids = cand_ids[i : i + sbatch]
                    Xc, _ = store.transform_matrix(df, chunk_ids)
                    yh = mdl.predict(Xc)  # (n,d)
                    per_tree = mdl.predict_per_tree(Xc)  # (T,n,d)
                    ystd = per_tree.std(axis=0)  # (n,d)
                    preds.extend(
                        [
                            (cid, list(map(float, row)))
                            for cid, row in zip(chunk_ids, yh)
                        ]
                    )
                    stds.extend(
                        [
                            (cid, list(map(float, row)))
                            for cid, row in zip(chunk_ids, ystd)
                        ]
                    )
            score_dt = time.perf_counter() - t1
            _echo_and_capture(log_lines, f"[score] done in {score_dt:.2f}s")

            scored = pd.DataFrame(preds, columns=["id", "y_pred_vec"]).merge(
                pd.DataFrame(stds, columns=["id", "y_pred_std_vec"]),
                on="id",
                how="left",
            )
            scored["uncertainty_mean_all_std"] = scored["y_pred_std_vec"].map(
                lambda v: (
                    float(np.mean(v))
                    if isinstance(v, list) and len(v) > 0
                    else float("nan")
                )
            )
            seq_map = df.set_index("id")["sequence"].to_dict()
            scored["sequence"] = scored["id"].map(seq_map)

            # quick uncertainty distribution signal
            unc_p10, unc_p50, unc_p90 = _pct(
                scored["uncertainty_mean_all_std"].to_numpy(), ps=(10, 50, 90)
            )
            _echo_and_capture(
                log_lines,
                "[uncertainty] mean(std per target) p10/p50/p90 = "
                f"{unc_p10:.3g}/{unc_p50:.3g}/{unc_p90:.3g}",
            )

            # objective
            obj_name = cfg.objective.objective.name
            obj_params = cfg.objective.objective.params
            obj_fn = get_objective(obj_name)

            ypred_mat = np.array(
                [np.array(v, dtype=float) for v in scored["y_pred_vec"].tolist()],
                dtype=float,
            )
            # Build RoundContext (includes effect scaling pool from labels)
            setpoint = list(obj_params.get("setpoint_vector", [0, 0, 0, 1]))
            beta = float(obj_params.get("logic_exponent_beta", 1.0))
            gamma = float(obj_params.get("intensity_exponent_gamma", 1.0))
            delta = float(obj_params.get("intensity_log2_offset_delta", 0.0))

            # compute effect pool from labels (vec8) -> y_lin -> w·y_lin
            effect_pool: list[float] = []
            for v in y_vals:
                if isinstance(v, list) and len(v) == 8:
                    ystar = np.asarray(v[4:8], dtype=float)
                    ylin = np.maximum(0.0, np.power(2.0, ystar) - delta)
                    s = np.asarray(setpoint, dtype=float)
                    w = (
                        np.zeros_like(s)
                        if float(np.sum(s)) <= 0
                        else (s / float(np.sum(s)))
                    )
                    effect_pool.append(float(np.sum(ylin * w)))
            percentile_cfg = obj_params.get("scaling", {})
            run_id = f"r{round}-{now_iso()}"
            rctx = RoundContext.build(
                slug=cfg.campaign.slug,
                round_index=int(round),
                setpoint=setpoint,
                label_ids=[str(_id) for _id in tr_ids],
                effect_pool_for_scaling=effect_pool,
                percentile_cfg={
                    "p": int(percentile_cfg.get("percentile", 95)),
                    "fallback_p": int(percentile_cfg.get("fallback_p", 75)),
                    "min_n": int(percentile_cfg.get("min_n", 5)),
                    "eps": float(percentile_cfg.get("eps", 1e-8)),
                },
                model_name=cfg.training.models.name,
                model_params=cfg.training.models.params,
                x_transform={
                    "name": cfg.data.transforms_x.name,
                    "params": cfg.data.transforms_x.params,
                },
                y_ingest_transform={
                    "name": cfg.data.transforms_y.name,
                    "params": cfg.data.transforms_y.params,
                },
                y_expected_length=cfg.data.y_expected_length,
                run_id=run_id,
            )

            _echo_and_capture(
                log_lines,
                f"[objective] {obj_name} setpoint={setpoint} "
                f"β={beta:.3g} γ={gamma:.3g} δ={delta:.3g}",
            )
            _echo_and_capture(log_lines, "[objective] computing scores…")
            obj_res = obj_fn(
                y_pred=ypred_mat,
                params=obj_params,
                round_ctx=rctx,
            )
            scored["selection_score"] = obj_res.score
            diag = obj_res.diagnostics or {}
            scored["logic_fidelity_l2_norm01"] = diag.get("logic_fidelity", np.nan)
            scored["effect_scaled"] = diag.get("effect_scaled", np.nan)

            n_clipped = int(np.sum(np.asarray(scored["effect_scaled"]) >= 1.0))
            _echo_and_capture(
                log_lines,
                f"[objective] denom p={rctx.percentile_cfg.get('p')} "
                f"from n={len(effect_pool)} labeled → denom={diag.get('denom_used', float('nan'))}",
            )
            _echo_and_capture(
                log_lines,
                f"[objective] clip: {n_clipped} / {len(scored)} (effect_scaled==1.0)",
            )

            # flags: simple QC
            def _flags_row(vpred: list[float], e_scaled: float) -> str:
                flags: list[str] = []
                arr = np.asarray(vpred, dtype=float)
                if not np.all(np.isfinite(arr)):
                    flags.append("nan_pred")
                if float(e_scaled) >= 1.0:
                    flags.append("clip_effect")
                return "|".join(flags)

            scored["flags"] = [
                _flags_row(v, e)
                for v, e in zip(scored["y_pred_vec"], scored["effect_scaled"])
            ]

            # deterministic order and selection via registry
            ids_arr = scored["id"].astype(str).to_numpy()
            sel_fn = get_selection(
                cfg.selection.selection.name,
                cfg.selection.selection.params,
            )
            tie_handling = cfg.selection.tie_handling or "competition_rank"
            objective_mode = (
                cfg.selection.selection.params.get("objective", "maximize")
                if cfg.selection and cfg.selection.selection
                else "maximize"
            )

            _echo_and_capture(
                log_lines,
                f"[selection] strategy={cfg.selection.selection.name} "
                f"objective={objective_mode} tie_handling={tie_handling}",
            )
            raw_sel_res = sel_fn(
                ids=ids_arr,
                scores=scored["selection_score"].to_numpy(dtype=float),
                top_k=int(k),
                tie_handling=tie_handling,
            )
            sel_res = normalize_selection_result(
                raw_sel_res,
                ids=ids_arr,
                scores=scored["selection_score"].to_numpy(dtype=float),
                top_k=int(k),
                tie_handling=tie_handling,
                objective=objective_mode,
            )
            # Deterministic global order:
            #   maximize  -> (-score, id)
            #   minimize  -> (score, id)
            # Keep the registry-provided ranks/selection booleans but enforce
            # a stable full ordering for downstream CSVs/artifacts.
            order = sel_res["order_idx"]
            scored_ranked = scored.iloc[order].copy().reset_index(drop=True)
            asc_score = objective_mode == "minimize"
            scored_ranked = scored_ranked.sort_values(
                by=["selection_score", "id"],
                ascending=[asc_score, True],
                kind="mergesort",  # stable
            ).reset_index(drop=True)
            _echo_and_capture(
                log_lines,
                "[selection] applied stable ordering: "
                + ("(score asc, id asc)" if asc_score else "(score desc, id asc)"),
            )

            scored_ranked["rank_competition"] = sel_res["ranks"].astype(int)
            scored_ranked["selected_top_k_bool"] = sel_res["selected_bool"].astype(bool)
            top_k_effective = int(scored_ranked["selected_top_k_bool"].sum())
            _echo_and_capture(
                log_lines,
                f"[selection] top_k=requested={int(k)} → effective={top_k_effective} (ties)",
            )

            # resuggested_unlabeled count
            allow_resuggest = bool(
                pol.get("allow_resuggesting_candidates_until_labeled", True)
            )
            prev_sel_cols = [
                c
                for c in df.columns
                if c.startswith(f"opal__{cfg.campaign.slug}__r")
                and c.endswith("__selected_top_k_bool")
            ]
            labeled_id_leq_round = store.labeled_id_set_leq_round(df, round)
            sel_current_ids = set(
                scored_ranked.loc[scored_ranked["selected_top_k_bool"], "id"].astype(
                    str
                )
            )
            resuggested_unlabeled = 0
            if prev_sel_cols:
                prev_selected_any = (
                    df[["id", *prev_sel_cols]]
                    .assign(_any=lambda _d: _d[prev_sel_cols].fillna(False).any(axis=1))
                    .loc[lambda _d: _d["_any"], "id"]
                    .astype(str)
                    .tolist()
                )
                prev_selected_set = set(prev_selected_any)
                for sid in sel_current_ids:
                    if (sid in prev_selected_set) and (sid not in labeled_id_leq_round):
                        resuggested_unlabeled += 1
            _echo_and_capture(
                log_lines,
                "[selection] resuggested_unlabeled="
                f"{resuggested_unlabeled} (policy allow_resuggest="
                f"{str(allow_resuggest).lower()})",
            )

            # artifacts
            apaths = ArtifactPaths(
                model=rdir / "model.joblib",
                selection_csv=rdir / "selection_top_k.csv",
                round_log_jsonl=rdir / "round.log.jsonl",
                round_ctx_json=rdir / "round_ctx.json",
                objective_meta_json=rdir / "objective_meta.json",
            )
            _echo_and_capture(log_lines, "[artifacts] writing model + CSVs…")
            mdl.save(str(apaths.model))
            model_sha = file_sha256(apaths.model)

            sel_sha = write_selection_csv(
                apaths.selection_csv,
                scored_ranked[scored_ranked["selected_top_k_bool"]][
                    ["id", "sequence", "selection_score"]
                ],
            )

            # write round_ctx and objective_meta
            ctx_sha = write_round_ctx(apaths.round_ctx_json, rctx.to_dict())
            obj_meta = {
                "denom_percentile": rctx.percentile_cfg.get("p"),
                "denom_value": float(diag.get("denom_used", float("nan"))),
                "beta": float(beta),
                "gamma": float(gamma),
                "delta": float(delta),
            }
            objm_sha = write_objective_meta(apaths.objective_meta_json, obj_meta)

            # ---- write canonical events.parquet ----
            ev_path = events_path(workdir)
            # fingerprint: short token derived from round_ctx.json checksum
            short_fp = ctx_sha[:12]
            events_df = pd.DataFrame(
                {
                    "round": int(round),
                    "run_id": rctx.run_id,
                    "objective": obj_name,
                    "id": scored_ranked["id"].astype(str),
                    "sequence": scored_ranked["sequence"].astype(str),
                    "score": scored_ranked["selection_score"].astype(float),
                    "rank_competition": scored_ranked["rank_competition"].astype(int),
                    "selected_top_k_bool": scored_ranked["selected_top_k_bool"].astype(
                        bool
                    ),
                    "uncertainty_mean_all_std": scored_ranked[
                        "uncertainty_mean_all_std"
                    ].astype(float),
                    "flags": scored_ranked["flags"].astype(str),
                    "fingerprint": short_fp,
                    "ts": pd.Timestamp.utcnow(),
                }
            )
            ev_sha = append_events(ev_path, events_df)

            df2 = store.update_latest_cache(
                df,
                slug=cfg.campaign.slug,
                latest_round=int(round),
                latest_score=events_df.set_index("id")["score"].to_dict(),
            )
            store.save_atomic(df2)
            _echo_and_capture(log_lines, "[events] appended to outputs/events.parquet")
            _echo_and_capture(
                log_lines, "[records] caches updated: latest_round/latest_score"
            )

            # update state
            st_path = Path(cfg.campaign.workdir) / "state.json"
            st = CampaignState.load(st_path) if st_path.exists() else CampaignState()
            st.campaign_slug = cfg.campaign.slug
            st.campaign_name = cfg.campaign.name
            st.workdir = str(workdir.resolve())
            st.x_column_name = cfg.data.x_column_name
            st.y_column_name = cfg.data.y_column_name
            st.representation_vector_dimension = rep.x_dim
            st.representation_transform = {
                "name": cfg.data.transforms_x.name,
                "params": cfg.data.transforms_x.params,
            }
            st.training_policy = cfg.training.policy
            st.performance = {
                "score_batch_size": score_batch_size or cfg.scoring.score_batch_size,
                "objective": obj_name,
            }
            st.data_location = {
                "kind": store.kind,
                "records_path": str(store.records_path.resolve()),
                "records_sha256": file_sha256(store.records_path),
            }
            st.add_round(
                RoundEntry(
                    round_index=round,
                    round_name=f"round_{round}",
                    round_dir=str(rdir.resolve()),
                    labels_used_rounds=list(range(0, round + 1)),
                    number_of_training_examples_used_in_round=int(len(tr_ids)),
                    number_of_candidates_scored_in_round=int(len(scored_ranked)),
                    selection_top_k_requested=int(k),
                    selection_top_k_effective_after_ties=int(top_k_effective),
                    model={
                        "type": cfg.training.models.name,
                        "params": cfg.training.models.params,
                        "artifact_path": str(apaths.model.resolve()),
                        "artifact_sha256": model_sha,
                    },
                    metrics={},  # metrics file removed (storage v2)
                    durations_sec={"fit": fit_dt, "score": score_dt},
                    seeds={
                        "global": cfg.training.models.params.get("random_state", None),
                        "model": cfg.training.models.params.get("random_state", None),
                    },
                    artifacts={
                        "selection_top_k_csv": str(apaths.selection_csv.resolve()),
                        "events_parquet": str(ev_path.resolve()),
                        "round_ctx_json": str(apaths.round_ctx_json.resolve()),
                        "objective_meta_json": str(
                            apaths.objective_meta_json.resolve()
                        ),
                        "checksums": {
                            "selection_top_k_csv": sel_sha,
                            "events_parquet": ev_sha,
                            "round_ctx_json": ctx_sha,
                            "objective_meta_json": objm_sha,
                        },
                    },
                    writebacks={
                        "records_caches_updated": [
                            "opal__<slug>__latest_round",
                            "opal__<slug>__latest_score",
                        ]
                    },
                    warnings=[],
                )
            )
            st.save(st_path)

            append_round_log_event(
                apaths.round_log_jsonl,
                {
                    "ts": now_iso(),
                    "round": int(round),
                    "stage": "fit",
                    "labels": int(len(tr_ids)),
                    "oob_r2": metrics.oob_r2,
                },
            )
            append_round_log_event(
                apaths.round_log_jsonl,
                {
                    "ts": now_iso(),
                    "round": int(round),
                    "stage": "predict",
                    "universe": int(len(cand_ids)),
                    "uncertainty_p10_p50_p90": [unc_p10, unc_p50, unc_p90],
                },
            )
            append_round_log_event(
                apaths.round_log_jsonl,
                {
                    "ts": now_iso(),
                    "round": int(round),
                    "stage": "objective",
                    "denom_p": rctx.percentile_cfg.get("p"),
                    "denom": float(diag.get("denom_used", float("nan"))),
                    "n_clipped": int(n_clipped),
                },
            )
            append_round_log_event(
                apaths.round_log_jsonl,
                {
                    "ts": now_iso(),
                    "round": int(round),
                    "stage": "selection",
                    "top_k": int(k),
                    "effective": int(top_k_effective),
                    "resuggested_unlabeled": int(resuggested_unlabeled),
                },
            )

            out_summary: Dict[str, Any] = {
                "trained_on": len(tr_ids),
                "scored": len(scored_ranked),
                "top_k_requested": k,
                "top_k_source": "cli_override" if _k_from_cli else "yaml_default",
                "selection_top_k_effective_after_ties": top_k_effective,
                "uncertainty_p10_p50_p90": [unc_p10, unc_p50, unc_p90],
                "objective": {
                    "name": obj_name,
                    "setpoint": setpoint,
                    "beta": beta,
                    "gamma": gamma,
                    "delta": delta,
                    "denom_p": rctx.percentile_cfg.get("p"),
                    "denom_used": diag.get("denom_used", None),
                    "n_clipped": n_clipped,
                },
                "selection": {
                    "strategy": cfg.selection.selection.name,
                    "tie_handling": tie_handling,
                    "objective_mode": objective_mode,
                    "resuggested_unlabeled": resuggested_unlabeled,
                    "policy_allow_resuggest": allow_resuggest,
                },
                "labels": {
                    "y_column": cfg.data.y_column_name,
                    "y_dim": y_dim,
                    "n_effective_training_examples": len(labels_eff),
                },
                "storage": {
                    "events_parquet": str(ev_path.resolve()),
                    "events_rows_appended": int(len(events_df)),
                    "fingerprint": short_fp,
                },
                "history": {
                    "column": lh_col,
                    "entries_used_leq_round": entries_used,
                    "rounds_seen": rounds_seen,
                    "policy": {
                        "cumulative_training": bool(
                            pol.get("cumulative_training", True)
                        ),
                        "dedup": pol.get(
                            "label_cross_round_deduplication_policy", "latest_only"
                        ),
                    },
                },
                "log_lines": log_lines,
            }
            if getattr(rep, "backfill", {}).get("checked", None) is not None:
                out_summary["preflight_backfill"] = rep.backfill

            json_out(out_summary)

    except OpalError as e:
        typer.echo(str(e), err=True)
        raise typer.Exit(code=e.exit_code)
    except Exception as e:
        internal_error("run", e)
        raise typer.Exit(code=ExitCodes.INTERNAL_ERROR)
