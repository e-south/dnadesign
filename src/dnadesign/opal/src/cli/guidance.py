"""
--------------------------------------------------------------------------------
<dnadesign project>
src/dnadesign/opal/src/cli/guidance.py

Builds guided workflow runbooks and state-aware next-step recommendations for
OPAL campaigns.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ..config.types import LabelSourceUSRSidecar, LocationLocal, LocationUSR, RootConfig
from ..core.utils import OpalError
from ..storage.data_access import RecordsStore
from ..storage.label_sources import label_source_status
from ..storage.state import CampaignState
from ..storage.workspace import CampaignWorkspace


@dataclass(frozen=True)
class GuidanceStep:
    title: str
    why: str
    command: str
    reads: list[str] = field(default_factory=list)
    writes: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class GuidanceReport:
    workflow_key: str
    campaign: dict[str, Any]
    plugins: dict[str, Any]
    round_semantics: dict[str, str]
    steps: list[GuidanceStep]
    common_errors: list[str]
    learn_more: dict[str, list[str]]


@dataclass(frozen=True)
class NextGuidance:
    stage: str
    reason: str
    labels_as_of: int
    observed_round: int
    labels_in_observed_round: int
    state_exists: bool
    run_exists_for_labels_as_of: bool
    next_commands: list[str]
    learn_more: list[str]
    records_path: str | None = None
    records_exists: bool | None = None
    label_source: dict[str, Any] = field(default_factory=dict)


def detect_workflow_key(cfg: RootConfig) -> str:
    model_name = str(cfg.model.name)
    selection_name = str(cfg.selection.selection.name)
    objective_names = [str(o.name) for o in cfg.objectives.objectives]
    has_sfxi = "sfxi_v1" in objective_names
    if model_name == "random_forest" and selection_name == "top_n" and has_sfxi:
        return "rf_sfxi_topn"
    if model_name == "gaussian_process" and selection_name == "top_n" and has_sfxi:
        return "gp_sfxi_topn"
    if model_name == "gaussian_process" and selection_name == "expected_improvement" and has_sfxi:
        return "gp_sfxi_ei"
    return "custom"


def _build_doc_pointers(cfg: RootConfig, workflow_key: str) -> dict[str, list[str]]:
    src_root = "src/dnadesign/opal/src"
    docs: list[str] = [
        "docs/workflows",
        "docs/reference/configuration.md",
        "docs/reference/cli.md",
        "docs/concepts/architecture.md",
        "docs/concepts/roundctx.md",
    ]
    source: list[str] = [
        f"{src_root}/runtime/run_round.py",
        f"{src_root}/runtime/round/stages/training.py",
        f"{src_root}/runtime/round/stages/x_matrices.py",
        f"{src_root}/runtime/round/stages/scoring.py",
        f"{src_root}/registries/models.py",
        f"{src_root}/registries/objectives.py",
        f"{src_root}/registries/selection.py",
    ]

    model_name = str(cfg.model.name)
    selection_name = str(cfg.selection.selection.name)
    objective_names = [str(o.name) for o in cfg.objectives.objectives]
    if model_name == "gaussian_process":
        docs.append("docs/plugins/models/gaussian-process.md")
        source.append(f"{src_root}/models/gaussian_process.py")
    else:
        docs.append("docs/plugins/models/README.md")
        source.append(f"{src_root}/models/random_forest.py")
    if "sfxi_v1" in objective_names:
        docs.append("docs/plugins/objectives/sfxi.md")
        source.append(f"{src_root}/objectives/sfxi_v1.py")
    if selection_name == "expected_improvement":
        docs.append("docs/plugins/selection/expected-improvement.md")
        source.append(f"{src_root}/selection/expected_improvement.py")
    docs.append("docs/plugins/selection/README.md")
    if workflow_key == "rf_sfxi_topn":
        docs.append("docs/workflows/rf-sfxi-topn.md")
    elif workflow_key == "gp_sfxi_topn":
        docs.append("docs/workflows/gp-sfxi-topn.md")
    elif workflow_key == "gp_sfxi_ei":
        docs.append("docs/workflows/gp-sfxi-ei.md")
    return {
        "docs": sorted(set(docs)),
        "source": sorted(set(source)),
    }


def _default_labels_file(cfg_path: Path) -> str:
    return str((cfg_path.parent.parent / "inputs" / "r0" / "vec8-b0.xlsx").resolve())


def _records_path_from_cfg(cfg: RootConfig) -> Path:
    location = cfg.data.location
    if isinstance(location, LocationUSR):
        return (Path(location.path) / location.dataset / "records.parquet").resolve()
    if isinstance(location, LocationLocal):
        return Path(location.path).resolve()
    raise OpalError(f"Unsupported data.location.kind: {getattr(location, 'kind', None)!r}")


def _label_source_summary(cfg: RootConfig) -> dict[str, Any]:
    source = cfg.labels.source
    if isinstance(source, LabelSourceUSRSidecar):
        location = cfg.data.location
        if not isinstance(location, LocationUSR):
            raise OpalError("labels.source.kind=usr_sidecar requires data.location.kind=usr.")
        return {
            "kind": "usr_sidecar",
            "path": str((Path(location.path) / source.dataset / source.path).resolve()),
            "y_space": cfg.labels.y_space,
            "prediction_records": cfg.writeback.prediction_records,
        }
    return {
        "kind": "campaign_history",
        "column": f"opal__{cfg.campaign.slug}__label_hist",
        "prediction_records": cfg.writeback.prediction_records,
    }


def _ingest_command(*, cfg_path: Path, cfg: RootConfig, labels_as_of: int, labels_file: str) -> str:
    command = f"opal ingest-y -c {cfg_path.resolve()} --observed-round {labels_as_of} --in {labels_file}"
    if isinstance(cfg.labels.source, LabelSourceUSRSidecar):
        command += " --unknown-sequences error"
    return f"{command} --apply"


def _build_steps(cfg_path: Path, cfg: RootConfig, labels_as_of: int) -> list[GuidanceStep]:
    c = str(cfg_path.resolve())
    labels_file = _default_labels_file(cfg_path)
    records_path = str(_records_path_from_cfg(cfg))
    label_source = _label_source_summary(cfg)
    if label_source["kind"] == "usr_sidecar":
        ingest_why = "Append observed round labels to the shared USR observed-label sidecar."
        ingest_writes = [str(label_source["path"]), "outputs/ledger/labels.parquet"]
    else:
        ingest_why = "Append observed round labels to campaign-local label history."
        ingest_writes = ["records.parquet", "outputs/ledger/labels.parquet"]
    return [
        GuidanceStep(
            title="Validate schema and plugin wiring",
            why="Fail fast on config, plugin names, records schema, and X-vector issues before writing campaign state.",
            command=f"opal validate -c {c}",
            reads=[c, records_path],
            writes=[],
        ),
        GuidanceStep(
            title="Initialize campaign workspace",
            why="Create OPAL output sinks and state.json for this campaign after validation passes.",
            command=f"opal init -c {c}",
            reads=[c, records_path],
            writes=["state.json", "outputs/"],
        ),
        GuidanceStep(
            title="Ingest observed labels",
            why=ingest_why,
            command=_ingest_command(cfg_path=cfg_path, cfg=cfg, labels_as_of=labels_as_of, labels_file=labels_file),
            reads=[labels_file, records_path],
            writes=ingest_writes,
        ),
        GuidanceStep(
            title="Run train/score/select round",
            why="Train surrogate, evaluate objectives, and produce selected candidates.",
            command=f"opal run -c {c} --labels-as-of {labels_as_of}",
            reads=[records_path, "state.json"],
            writes=[
                f"outputs/rounds/round_{labels_as_of}/selection/selection_top_k.csv",
                "outputs/ledger/runs.parquet",
                "outputs/ledger/predictions/",
            ],
        ),
        GuidanceStep(
            title="Verify selection/ledger agreement",
            why="Confirm persisted selection artifacts match ledger prediction rows.",
            command=f"opal verify-outputs -c {c} --round latest",
            reads=["outputs/rounds/round_*/selection/selection_top_k.csv", "outputs/ledger/predictions/"],
            writes=[],
        ),
        GuidanceStep(
            title="Inspect runtime carriers",
            why="Review RoundCtx contracts emitted by the latest run.",
            command=f"opal ctx audit -c {c} --round latest",
            reads=["outputs/rounds/round_*/metadata/round_ctx.json", records_path],
            writes=[],
        ),
        GuidanceStep(
            title="Dry-run next-round preflight",
            why="Explain the next labels-as-of cut before running it.",
            command=f"opal explain -c {c} --labels-as-of {labels_as_of + 1}",
            reads=[records_path, "state.json"],
            writes=[],
        ),
    ]


def build_guidance_report(cfg_path: Path, cfg: RootConfig, *, labels_as_of: int = 0) -> GuidanceReport:
    ws = CampaignWorkspace.from_config(cfg, cfg_path)
    workflow_key = detect_workflow_key(cfg)
    objective_rows = [{"name": str(o.name), "params": dict(o.params or {})} for o in cfg.objectives.objectives]
    report = GuidanceReport(
        workflow_key=workflow_key,
        campaign={
            "name": cfg.campaign.name,
            "slug": cfg.campaign.slug,
            "config_path": str(ws.config_path),
            "workdir": str(ws.workdir),
            "records_path": str(_records_path_from_cfg(cfg)),
            "label_source": _label_source_summary(cfg),
        },
        plugins={
            "model": {"name": cfg.model.name, "params": dict(cfg.model.params or {})},
            "objectives": objective_rows,
            "selection": {
                "name": cfg.selection.selection.name,
                "params": dict(cfg.selection.selection.params or {}),
            },
        },
        round_semantics={
            "observed_round": "Round index stamped on ingested labels (wet-lab event time).",
            "labels_as_of": "Training cutoff; model sees labels with observed_round <= labels_as_of.",
        },
        steps=_build_steps(cfg_path, cfg, int(labels_as_of)),
        common_errors=[
            "SFXI min_n failures occur when current-round observed labels are missing for labels-as-of round.",
            "EI requires uncertainty_ref resolving to a finite, strictly positive standard-deviation channel.",
            "score_ref and uncertainty_ref must be '<objective>/<channel>' and resolve against configured objectives.",
        ],
        learn_more=_build_doc_pointers(cfg, workflow_key),
    )
    return report


def _label_counts_by_round(cfg: RootConfig, store: RecordsStore, df) -> dict[int, int]:
    status = label_source_status(cfg, store, df, strict=False)
    counts = status.get("counts_by_round") or {}
    return {int(round_index): int(count) for round_index, count in dict(counts).items()}


def _state_round_set(state_path: Path) -> set[int]:
    if not state_path.exists():
        return set()
    try:
        st = CampaignState.load(state_path)
    except Exception as e:
        raise OpalError(f"Failed to load state.json at {state_path}: {e}") from e
    return {int(r.round_index) for r in st.rounds}


def _sfxi_min_n(cfg: RootConfig) -> int | None:
    for obj in cfg.objectives.objectives:
        if str(obj.name) != "sfxi_v1":
            continue
        scaling = dict((obj.params or {}).get("scaling") or {})
        try:
            return int(scaling.get("min_n", 5))
        except Exception:
            return 5
    return None


def build_next_guidance(
    cfg_path: Path,
    cfg: RootConfig,
    store: RecordsStore,
    df,
    *,
    labels_as_of: int | None = None,
    observed_round: int | None = None,
) -> NextGuidance:
    ws = CampaignWorkspace.from_config(cfg, cfg_path)
    state_exists = ws.state_path.exists()
    target_as_of = int(labels_as_of) if labels_as_of is not None else 0
    target_observed = int(observed_round) if observed_round is not None else target_as_of

    if not state_exists:
        return NextGuidance(
            stage="validate",
            reason="Campaign state is missing. Validate the candidate table before writing state.json.",
            labels_as_of=target_as_of,
            observed_round=target_observed,
            labels_in_observed_round=0,
            state_exists=False,
            run_exists_for_labels_as_of=False,
            next_commands=[
                f"opal validate -c {ws.config_path}",
                f"opal init -c {ws.config_path}",
            ],
            learn_more=["docs/reference/cli.md", "docs/workflows"],
            records_path=str(store.records_path),
            records_exists=store.records_path.exists(),
            label_source=_label_source_summary(cfg),
        )

    labels_by_round = _label_counts_by_round(cfg, store, df)
    labels_in_round = int(labels_by_round.get(target_observed, 0))
    rounds_with_runs = _state_round_set(ws.state_path)
    run_exists = target_as_of in rounds_with_runs

    min_n = _sfxi_min_n(cfg)
    min_n_note = ""
    if min_n is not None:
        min_n_note = f" SFXI scaling requires current-round labels >= {min_n}."

    if labels_in_round <= 0:
        return NextGuidance(
            stage="ingest",
            reason=(f"No observed labels found for round {target_observed}.{min_n_note}").strip(),
            labels_as_of=target_as_of,
            observed_round=target_observed,
            labels_in_observed_round=labels_in_round,
            state_exists=True,
            run_exists_for_labels_as_of=run_exists,
            next_commands=[
                _ingest_command(
                    cfg_path=ws.config_path,
                    cfg=cfg,
                    labels_as_of=target_observed,
                    labels_file="<labels.xlsx>",
                ),
            ],
            learn_more=["docs/reference/cli.md", "docs/plugins/objectives/sfxi.md"],
        )

    if not run_exists:
        return NextGuidance(
            stage="run",
            reason=f"Labels exist for observed round {target_observed}; run selection for labels-as-of {target_as_of}.",
            labels_as_of=target_as_of,
            observed_round=target_observed,
            labels_in_observed_round=labels_in_round,
            state_exists=True,
            run_exists_for_labels_as_of=False,
            next_commands=[
                f"opal run -c {ws.config_path} --labels-as-of {target_as_of}",
            ],
            learn_more=["docs/reference/cli.md", "docs/concepts/architecture.md"],
        )

    return NextGuidance(
        stage="post_run",
        reason=f"Round {target_as_of} has run artifacts. Verify outputs and inspect selected records.",
        labels_as_of=target_as_of,
        observed_round=target_observed,
        labels_in_observed_round=labels_in_round,
        state_exists=True,
        run_exists_for_labels_as_of=True,
        next_commands=[
            f"opal verify-outputs -c {ws.config_path} --round latest",
            f"opal ctx audit -c {ws.config_path} --round latest",
            f"opal status -c {ws.config_path}",
            f"opal explain -c {ws.config_path} --labels-as-of {target_as_of + 1}",
        ],
        learn_more=["docs/reference/cli.md", "docs/concepts/roundctx.md"],
    )


def build_missing_records_guidance(
    cfg_path: Path,
    cfg: RootConfig,
    records_path: Path,
    *,
    labels_as_of: int | None = None,
    observed_round: int | None = None,
) -> NextGuidance:
    target_as_of = int(labels_as_of) if labels_as_of is not None else 0
    target_observed = int(observed_round) if observed_round is not None else target_as_of
    return NextGuidance(
        stage="candidate_table",
        reason=f"Candidate records.parquet is missing: {records_path}",
        labels_as_of=target_as_of,
        observed_round=target_observed,
        labels_in_observed_round=0,
        state_exists=CampaignWorkspace.from_config(cfg, cfg_path).state_path.exists(),
        run_exists_for_labels_as_of=False,
        next_commands=[
            f"opal guide -c {cfg_path}",
            f"opal validate -c {cfg_path}",
        ],
        learn_more=["docs/reference/data-contracts.md", "docs/workflows/usr-infer-x-active-learning.md"],
        records_path=str(records_path),
        records_exists=False,
        label_source=_label_source_summary(cfg),
    )
