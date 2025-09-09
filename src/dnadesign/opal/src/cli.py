"""
--------------------------------------------------------------------------------
<dnadesign project>
src/dnadesign/opal/src/cli.py

This is the user-facing command surface.

Module Author(s): Eric J. South
Dunlop Lab
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import json
import os
import traceback
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import typer

from .artifacts import (
    ArtifactPaths,
    round_dir,
    write_feature_importance,
    write_round_metrics,
    write_selection_csv,
)
from .config import LocationLocal, LocationUSR, RootConfig, load_config
from .data_access import RecordsStore
from .explain import explain_round
from .locks import CampaignLock
from .models.registry import get_model
from .predict import run_predict_ephemeral
from .preflight import preflight_run
from .record_show import build_record_report
from .selection.top_n import select_top_n
from .state import CampaignState, RoundEntry
from .status import build_status
from .utils import (
    ExitCodes,
    OpalError,
    ensure_dir,
    file_sha256,
    print_stderr,
    print_stdout,
)
from .writebacks import write_round_columns

app = typer.Typer(add_completion=True, no_args_is_help=True, help="OPAL")


def _debug_enabled() -> bool:
    v = str(os.getenv("OPAL_DEBUG", "")).strip().lower()
    return v in ("1", "true", "yes", "on")


def _internal_error(ctx: str, e: Exception) -> None:
    """Print a helpful internal error. With OPAL_DEBUG set, include traceback."""
    if _debug_enabled():
        tb = traceback.format_exc()
        print_stderr(f"Internal error during {ctx}: {e}\n{tb}")
    else:
        print_stderr(
            f"Internal error during {ctx}: {e}\n(Hint: set OPAL_DEBUG=1 for a full traceback)"
        )


def _resolve_config_path(opt: Optional[Path]) -> Path:
    """Resolve campaign config by precedence:
    1) explicit --config,
    2) $OPAL_CONFIG,
    3) nearest ./campaign.yaml walking up,
    4) single campaign.yaml under src/dnadesign/opal/campaigns.
    """
    if opt:
        return opt.resolve()

    env = os.getenv("OPAL_CONFIG")
    if env:
        p = Path(env).expanduser()
        if p.exists():
            return p.resolve()

    # walk up from CWD
    cur = Path.cwd()
    for base in (cur, *cur.parents):
        cand = base / "campaign.yaml"
        if cand.exists():
            return cand.resolve()

    # single fallback in campaigns/
    root = Path("src/dnadesign/opal/campaigns")
    if root.exists():
        found = list(root.glob("*/campaign.yaml"))
        if len(found) == 1:
            return found[0].resolve()

    raise OpalError(
        "No campaign.yaml found. Use --config, set $OPAL_CONFIG, "
        "or run from within a campaign directory.",
        ExitCodes.BAD_ARGS,
    )


def _store_from_cfg(cfg: RootConfig) -> RecordsStore:
    loc = cfg.data.location
    if isinstance(loc, LocationUSR):
        records = Path(loc.usr_root) / loc.dataset / "records.parquet"
        data_location = {
            "kind": "usr",
            "dataset": loc.dataset,
            "usr_root": str(Path(loc.usr_root).resolve()),
            "records_path": str(records.resolve()),
        }
    elif isinstance(loc, LocationLocal):
        records = Path(loc.path)
        data_location = {"kind": "local", "records_path": str(records.resolve())}
    else:
        # Pydantic has already validated kinds; this is defensive
        raise OpalError("Unknown data location kind.", ExitCodes.BAD_ARGS)

    return RecordsStore(
        kind=data_location["kind"],
        records_path=records,
        campaign_slug=cfg.campaign.slug,
        x_col=cfg.data.representation_column_name,
        y_col=cfg.data.label_source_column_name,
        transform_name=cfg.data.transform.name,
        transform_params=cfg.data.transform.params,
    )


def _state_path(cfg: RootConfig) -> Path:
    return Path(cfg.campaign.workdir) / "state.json"


def _campaign_log_path(cfg: RootConfig) -> Path:
    return Path(cfg.campaign.workdir) / "campaign.log.jsonl"


@app.command("init")
def cmd_init(
    config: Optional[Path] = typer.Option(
        None, "--config", "-c", envvar="OPAL_CONFIG", help="Path to campaign.yaml"
    )
):
    try:
        config = _resolve_config_path(config)
        cfg = load_config(config)
        workdir = Path(cfg.campaign.workdir)
        ensure_dir(workdir / "outputs")

        # write initial state.json
        store = _store_from_cfg(cfg)
        st = CampaignState(
            campaign_slug=cfg.campaign.slug,
            campaign_name=cfg.campaign.name,
            workdir=str(workdir.resolve()),
            data_location={
                **({"kind": store.kind}),
                "records_path": str(store.records_path.resolve()),
                "records_sha256": (
                    file_sha256(store.records_path)
                    if store.records_path.exists()
                    else ""
                ),
            },
            representation_column_name=cfg.data.representation_column_name,
            label_source_column_name=cfg.data.label_source_column_name,
            transform=cfg.data.transform.dict(),
            training_policy=cfg.training["policy"].dict(),
            performance={
                "score_batch_size": cfg.scoring.score_batch_size,
                "objective": cfg.metadata.objective,
            },
            representation_vector_dimension=0,
            backlog={"number_of_selected_but_not_yet_labeled_candidates_total": 0},
        )
        st.save(_state_path(cfg))
        print_stdout(f"Initialized campaign at {workdir}")
        return
    except OpalError as e:
        print_stderr(str(e))
        raise typer.Exit(code=e.exit_code)
    except Exception as e:
        _internal_error("init", e)
        raise typer.Exit(code=ExitCodes.INTERNAL_ERROR)


@app.command("explain")
def cmd_explain(
    config: Optional[Path] = typer.Option(
        None, "--config", "-c", envvar="OPAL_CONFIG", help="Path to campaign.yaml"
    ),
    round: int = typer.Option(..., "--round", "-r"),
):
    try:
        config = _resolve_config_path(config)
        cfg = load_config(config)
        store = _store_from_cfg(cfg)
        df = store.load()
        info = explain_round(store, df, cfg, round)
        print_stdout(json.dumps(info, indent=2))
        return
    except OpalError as e:
        print_stderr(str(e))
        raise typer.Exit(code=e.exit_code)
    except Exception as e:
        _internal_error("explain", e)
        raise typer.Exit(code=ExitCodes.INTERNAL_ERROR)


@app.command("labels-import")
def cmd_labels_import(
    config: Optional[Path] = typer.Option(
        None, "--config", "-c", envvar="OPAL_CONFIG", help="Path to campaign.yaml"
    ),
    round: int = typer.Option(..., "--round", "-r"),
    path: Path = typer.Option(..., "--path", "-p"),
    allow_overwrite_meta: bool = typer.Option(False, "--allow-overwrite-meta"),
):
    try:
        config = _resolve_config_path(config)
        cfg = load_config(config)
        store = _store_from_cfg(cfg)
        df = store.load()
        # load labels file
        lab = (
            pd.read_parquet(path)
            if path.suffix.lower() in (".parquet", ".pq")
            else pd.read_csv(path)
        )
        df2 = store.append_labels(
            df, lab, r=round, allow_overwrite_meta=allow_overwrite_meta
        )
        store.save_atomic(df2)
        print_stdout(f"Imported labels for round {round} from {path}.")
        return
    except OpalError as e:
        print_stderr(str(e))
        raise typer.Exit(code=e.exit_code)
    except Exception as e:
        _internal_error("labels-import", e)
        raise typer.Exit(code=ExitCodes.INTERNAL_ERROR)


@app.command("run")
@app.command("fit")  # alias
def cmd_run(
    config: Optional[Path] = typer.Option(
        None, "--config", "-c", envvar="OPAL_CONFIG", help="Path to campaign.yaml"
    ),
    round: int = typer.Option(..., "--round", "-r"),
    k: Optional[int] = typer.Option(None, "--k"),
    resume: bool = typer.Option(False, "--resume"),
    force: bool = typer.Option(False, "--force"),
    score_batch_size: Optional[int] = typer.Option(None, "--score-batch-size"),
    accept_x_mismatch: bool = typer.Option(False, "--accept-x-mismatch"),
):
    try:
        config = _resolve_config_path(config)
        cfg = load_config(config)
        if k is None:
            k = cfg.selection.top_k_default

        workdir = Path(cfg.campaign.workdir)
        rdir = round_dir(workdir, round)
        ensure_dir(rdir)

        lock = CampaignLock(workdir)
        with lock:
            store = _store_from_cfg(cfg)
            df = store.load()

            # preflight
            rep = preflight_run(
                store, df, round, cfg.safety.fail_on_mixed_biotype_or_alphabet
            )

            # record x dim to state
            st_path = _state_path(cfg)
            st = CampaignState.load(st_path) if st_path.exists() else CampaignState()
            st.campaign_slug = cfg.campaign.slug
            st.campaign_name = cfg.campaign.name
            st.workdir = str(workdir.resolve())
            st.representation_column_name = cfg.data.representation_column_name
            st.label_source_column_name = cfg.data.label_source_column_name
            st.representation_vector_dimension = rep.x_dim
            st.transform = cfg.data.transform.dict()
            st.training_policy = cfg.training["policy"].dict()
            st.performance = {
                "score_batch_size": score_batch_size or cfg.scoring.score_batch_size,
                "objective": cfg.metadata.objective,
            }
            st.data_location = {
                "kind": store.kind,
                "records_path": str(store.records_path.resolve()),
                "records_sha256": file_sha256(store.records_path),
            }

            # prepare training data
            labels_eff = store.effective_labels_latest_only(df, round)
            tr_ids = labels_eff["id"].tolist()
            Xtr, _ = store.transform_matrix(df, tr_ids)
            ytr = labels_eff["y"].to_numpy(dtype=float)

            # candidate universe
            cand = store.candidate_universe(df, round)
            cand_ids = cand["id"].tolist()

            # train
            model_cfg = cfg.training["model"]
            mdl = get_model(model_cfg.name, model_cfg.params.dict())
            with_time = {}
            import time

            t0 = time.perf_counter()
            metrics = mdl.fit(Xtr, ytr)
            with_time["fit"] = time.perf_counter() - t0

            # score in batches
            sbatch = score_batch_size or cfg.scoring.score_batch_size
            preds = []
            t1 = time.perf_counter()
            for i in range(0, len(cand_ids), sbatch):
                chunk_ids = cand_ids[i : i + sbatch]
                Xc, _ = store.transform_matrix(df, chunk_ids)
                yh = mdl.predict(Xc)
                preds.append(pd.DataFrame({"id": chunk_ids, "y_pred": yh}))
            score_dt = time.perf_counter() - t1
            with_time["score"] = score_dt
            preds_df = (
                pd.concat(preds, ignore_index=True)
                if preds
                else pd.DataFrame(columns=["id", "y_pred"])
            )

            # assemble scored universe (sorted deterministically)
            seq_map = df.set_index("id")["sequence"].to_dict()
            scored = preds_df.copy()
            scored["sequence"] = scored["id"].map(seq_map)
            scored = scored.sort_values(
                by=["y_pred", "id"], ascending=[False, True]
            ).reset_index(drop=True)

            # selection and ranking
            scored_ranked, top_k_effective = select_top_n(scored, "y_pred", k)

            # write artifacts
            apaths = ArtifactPaths(
                model=rdir / "model.joblib",
                selection_csv=rdir / "selection_top_k.csv",
                feature_importance_csv=rdir / "feature_importance.csv",
                metrics_json=rdir / "round_model_metrics.json",
                round_log_jsonl=rdir / "round.log.jsonl",
            )
            mdl.save(str(apaths.model))
            model_sha = file_sha256(apaths.model)
            sel_sha = write_selection_csv(
                apaths.selection_csv,
                scored_ranked[scored_ranked["selected_top_k_bool"]],
            )
            fi_sha = write_feature_importance(
                apaths.feature_importance_csv, mdl.feature_importances()
            )
            met = {
                "random_forest_oob_r2_score": metrics.oob_r2,
                "random_forest_oob_mse": metrics.oob_mse,
            }
            met_sha = write_round_metrics(apaths.metrics_json, met)

            # write-backs to records
            df2 = write_round_columns(
                df,
                cfg.campaign.slug,
                round,
                scored_ranked,
                allow_overwrite=resume or force,
            )
            store.save_atomic(df2)

            # update state
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
                        "type": model_cfg.name,
                        "params": model_cfg.params.dict(),
                        "artifact_path": str(apaths.model.resolve()),
                        "artifact_sha256": model_sha,
                    },
                    metrics=met,
                    durations_sec={
                        "fit": with_time.get("fit", 0.0),
                        "score": with_time.get("score", 0.0),
                    },
                    seeds={
                        "global": model_cfg.params.random_state,
                        "model": model_cfg.params.random_state,
                    },
                    artifacts={
                        "selection_top_k_csv": str(apaths.selection_csv.resolve()),
                        "feature_importance_csv": str(
                            apaths.feature_importance_csv.resolve()
                        ),
                        "round_model_metrics_json": str(apaths.metrics_json.resolve()),
                        "checksums": {
                            "selection_top_k_csv": sel_sha,
                            "feature_importance_csv": fi_sha,
                            "round_model_metrics_json": met_sha,
                        },
                    },
                    writebacks={
                        "records_columns_written": [
                            f"opal__{cfg.campaign.slug}__r{round}__pred",
                            f"opal__{cfg.campaign.slug}__r{round}__rank_competition",
                            (
                                f"opal__{cfg.campaign.slug}__r{round}"
                                f"__selected_top_k_bool"
                            ),
                        ]
                    },
                    warnings=[],
                )
            )
            st.save(_state_path(cfg))

            print_stdout(
                json.dumps(
                    {
                        "number_of_training_examples_used_in_round": len(tr_ids),
                        "number_of_candidates_scored_in_round": len(scored_ranked),
                        "selection_top_k_effective_after_ties": top_k_effective,
                    },
                    indent=2,
                )
            )
            return
    except OpalError as e:
        print_stderr(str(e))
        raise typer.Exit(code=e.exit_code)
    except Exception as e:
        _internal_error("run", e)
        raise typer.Exit(code=ExitCodes.INTERNAL_ERROR)


@app.command("predict")
def cmd_predict(
    config: Optional[Path] = typer.Option(
        None, "--config", "-c", envvar="OPAL_CONFIG", help="Path to campaign.yaml"
    ),
    model_path: Path = typer.Option(..., "--model-path"),
    input_path: Optional[Path] = typer.Option(None, "--in"),
    out_path: Optional[Path] = typer.Option(None, "--out"),
):
    try:
        config = _resolve_config_path(config)
        cfg = load_config(config)
        store = _store_from_cfg(cfg)
        df = (
            store.load()
            if input_path is None
            else (
                pd.read_parquet(input_path)
                if input_path.suffix.lower() in (".parquet", ".pq")
                else pd.read_csv(input_path)
            )
        )
        # require x_col presence for all rows
        if cfg.data.representation_column_name not in df.columns:
            raise OpalError(
                "Input missing representation column: "
                f"{cfg.data.representation_column_name}"
            )
        preds = run_predict_ephemeral(store, df, model_path, None)
        if out_path:
            (
                preds.to_csv(out_path, index=False)
                if out_path.suffix.lower() == ".csv"
                else preds.to_parquet(out_path, index=False)
            )
            print_stdout(f"Wrote predictions: {out_path}")
        else:
            print_stdout(preds.to_csv(index=False))
        return
    except OpalError as e:
        print_stderr(str(e))
        raise typer.Exit(code=e.exit_code)
    except Exception as e:
        _internal_error("predict", e)
        raise typer.Exit(code=ExitCodes.INTERNAL_ERROR)


@app.command("model-show")
def cmd_model_show(
    model_path: Path = typer.Option(..., "--model-path"),
    out_dir: Optional[Path] = typer.Option(None, "--out-dir"),
):
    try:
        from .models.random_forest import RandomForestModel

        mdl = RandomForestModel.load(str(model_path))
        info = {
            "model_type": "random_forest",
            "params": mdl.get_params(),
        }
        # no easy access to OOB stored; in state/metrics.json per round
        if out_dir:
            ensure_dir(out_dir)
            # write full feature importance
            importances = mdl.feature_importances()
            if importances is not None:
                fi = pd.DataFrame(
                    {
                        "feature_index": np.arange(len(importances)),
                        "feature_importance": importances,
                    }
                )
                fi["feature_rank"] = (
                    fi["feature_importance"]
                    .rank(ascending=False, method="min")
                    .astype(int)
                )
                fi.sort_values("feature_rank").to_csv(
                    out_dir / "feature_importance_full.csv", index=False
                )
                info["feature_importance_top20"] = fi.nlargest(
                    20, "feature_importance"
                ).to_dict(orient="records")
        print_stdout(json.dumps(info, indent=2))
        return
    except OpalError as e:
        print_stderr(str(e))
        raise typer.Exit(code=e.exit_code)
    except Exception as e:
        _internal_error("model-show", e)
        raise typer.Exit(code=ExitCodes.INTERNAL_ERROR)


@app.command("record-show")
def cmd_record_show(
    config: Optional[Path] = typer.Option(
        None, "--config", "-c", envvar="OPAL_CONFIG", help="Path to campaign.yaml"
    ),
    id: Optional[str] = typer.Option(None, "--id"),
    sequence: Optional[str] = typer.Option(None, "--sequence"),
    bio_type: Optional[str] = typer.Option(None, "--bio-type"),
    alphabet: Optional[str] = typer.Option(None, "--alphabet"),
    with_sequence: bool = typer.Option(False, "--with-sequence"),
    json_out: bool = typer.Option(False, "--json"),
):
    try:
        config = _resolve_config_path(config)
        cfg = load_config(config)
        store = _store_from_cfg(cfg)
        df = store.load()

        if not id and not sequence:
            raise OpalError(
                "Provide --id or --sequence (with --bio-type/--alphabet).",
                ExitCodes.BAD_ARGS,
            )

        report = build_record_report(
            df,
            cfg.campaign.slug,
            id_=id,
            sequence=sequence,
            with_sequence=with_sequence,
        )
        if json_out:
            print_stdout(json.dumps(report, indent=2))
        else:
            # compact text table
            lines = [f"id: {report['id']}"]
            if with_sequence and "sequence" in report:
                lines.append(f"sequence: {report['sequence']}")
            lines.append(f"length: {report['length']}")
            if "ground_truth_label" in report:
                gt = report["ground_truth_label"]
                gr = report["ground_truth_src_round"]
                lines.append(f"ground_truth: y={gt} (src_round={gr})")
            if "per_round" in report:
                for r, d in report["per_round"].items():
                    lines.append(
                        f"r={r}: y_pred={d['y_pred']}, "
                        f"rank={d['rank']}, selected={d['selected']}"
                    )
            print_stdout("\n".join(lines))
        return
    except OpalError as e:
        print_stderr(str(e))
        raise typer.Exit(code=e.exit_code)
    except Exception as e:
        _internal_error("record-show", e)
        raise typer.Exit(code=ExitCodes.INTERNAL_ERROR)


@app.command("status")
def cmd_status(
    config: Optional[Path] = typer.Option(
        None, "--config", "-c", envvar="OPAL_CONFIG", help="Path to campaign.yaml"
    ),
    round: Optional[int] = typer.Option(None, "--round"),
    all: bool = typer.Option(False, "--all"),
    json_out: bool = typer.Option(False, "--json"),
):
    try:
        config = _resolve_config_path(config)
        cfg = load_config(config)
        st = build_status(_state_path(cfg), round_k=round, show_all=all)
        if json_out:
            print_stdout(json.dumps(st, indent=2))
        else:
            if all:
                print_stdout(json.dumps(st, indent=2))
            else:
                # compact
                if "latest_round" in st and st["latest_round"]:
                    lr = st["latest_round"]
                    msg = (
                        f"latest round: r={lr['round_index']}, "
                        f"n_train={lr['number_of_training_examples_used_in_round']}, "
                        f"n_scored={lr['number_of_candidates_scored_in_round']}"
                    )
                    print_stdout(msg)
                elif "round" in st and st["round"]:
                    r = st["round"]
                    msg = (
                        f"round r={r['round_index']}, "
                        f"n_train={r['number_of_training_examples_used_in_round']}, "
                        f"n_scored={r['number_of_candidates_scored_in_round']}"
                    )
                    print_stdout(msg)
                else:
                    print_stdout("No completed rounds.")
        return
    except OpalError as e:
        print_stderr(str(e))
        raise typer.Exit(code=e.exit_code)
    except Exception as e:
        _internal_error("status", e)
        raise typer.Exit(code=ExitCodes.INTERNAL_ERROR)


@app.command("validate")
def cmd_validate(
    config: Optional[Path] = typer.Option(
        None, "--config", "-c", envvar="OPAL_CONFIG", help="Path to campaign.yaml"
    )
):
    try:
        config = _resolve_config_path(config)
        cfg = load_config(config)
        store = _store_from_cfg(cfg)
        df = store.load()
        # basic preflight with current round=0 if unsure; here just check essentials + x column presence
        missing_ess = [
            c
            for c in (
                "id",
                "bio_type",
                "sequence",
                "alphabet",
                "length",
                "source",
                "created_at",
            )
            if c not in df.columns
        ]
        if missing_ess:
            raise OpalError(f"Missing essential columns: {missing_ess}")
        if cfg.data.representation_column_name not in df.columns:
            raise OpalError(
                f"Missing representation column: {cfg.data.representation_column_name}"
            )
        print_stdout("OK: validation passed.")
        return
    except OpalError as e:
        print_stderr(str(e))
        raise typer.Exit(code=e.exit_code)
    except Exception as e:
        _internal_error("validate", e)
        raise typer.Exit(code=ExitCodes.INTERNAL_ERROR)


def main():
    app()


if __name__ == "__main__":
    main()
