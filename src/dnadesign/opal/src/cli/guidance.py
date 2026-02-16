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

from ..config.types import RootConfig
from ..core.utils import OpalError
from ..storage.data_access import RecordsStore
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
    docs: list[str] = [
        "docs/workflows",
        "docs/reference/configuration.md",
        "docs/reference/cli.md",
        "docs/concepts/architecture.md",
        "docs/concepts/roundctx.md",
    ]
    source: list[str] = [
        "src/runtime/run_round.py",
        "src/runtime/round/stages.py",
        "src/registries/models.py",
        "src/registries/objectives.py",
        "src/registries/selection.py",
    ]

    model_name = str(cfg.model.name)
    selection_name = str(cfg.selection.selection.name)
    objective_names = [str(o.name) for o in cfg.objectives.objectives]
    if model_name == "gaussian_process":
        docs.append("docs/plugins/model-gaussian-process.md")
        source.append("src/models/gaussian_process.py")
    else:
        docs.append("docs/plugins/models.md")
        source.append("src/models/random_forest.py")
    if "sfxi_v1" in objective_names:
        docs.append("docs/plugins/objective-sfxi.md")
        source.append("src/objectives/sfxi_v1.py")
    if selection_name == "expected_improvement":
        docs.append("docs/plugins/selection-expected-improvement.md")
        source.append("src/selection/expected_improvement.py")
    docs.append("docs/plugins/selection.md")
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


def _build_steps(cfg_path: Path, labels_as_of: int) -> list[GuidanceStep]:
    c = str(cfg_path.resolve())
    labels_file = _default_labels_file(cfg_path)
    return [
        GuidanceStep(
            title="Initialize campaign workspace",
            why="Create OPAL output sinks and state.json for this campaign.",
            command=f"opal init -c {c}",
            reads=[c],
            writes=["state.json", "outputs/"],
        ),
        GuidanceStep(
            title="Validate schema and plugin wiring",
            why="Fail fast on config, plugin names, and records schema issues before runtime.",
            command=f"opal validate -c {c}",
            reads=[c, "records.parquet"],
            writes=[],
        ),
        GuidanceStep(
            title="Ingest observed labels",
            why="Append observed round labels to canonical label history.",
            command=f"opal ingest-y -c {c} --observed-round {labels_as_of} --in {labels_file} --apply",
            reads=[labels_file, "records.parquet"],
            writes=["records.parquet", "outputs/ledger/labels.parquet"],
        ),
        GuidanceStep(
            title="Run train/score/select round",
            why="Train surrogate, evaluate objectives, and produce selected candidates.",
            command=f"opal run -c {c} --labels-as-of {labels_as_of}",
            reads=["records.parquet", "state.json"],
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
            title="Inspect runtime carriers and next-round preflight",
            why="Review RoundCtx contracts and dry-run the next labels-as-of cut.",
            command=f"opal ctx audit -c {c} --round latest && opal explain -c {c} --labels-as-of {labels_as_of + 1}",
            reads=["outputs/rounds/round_*/metadata/round_ctx.json", "records.parquet"],
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
            "records_path": str(Path(cfg.data.location.path).resolve()),
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
        steps=_build_steps(cfg_path, int(labels_as_of)),
        common_errors=[
            "SFXI min_n failures occur when current-round observed labels are missing for labels-as-of round.",
            "EI requires uncertainty_ref resolving to a finite, non-negative standard-deviation channel.",
            "score_ref and uncertainty_ref must be '<objective>/<channel>' and resolve against configured objectives.",
        ],
        learn_more=_build_doc_pointers(cfg, workflow_key),
    )
    return report


def _label_counts_by_round(store: RecordsStore, df) -> dict[int, int]:
    counts: dict[int, int] = {}
    lh_col = store.label_hist_col()
    if lh_col not in df.columns:
        return counts
    for cell in df[lh_col].tolist():
        for entry in store._normalize_hist_cell(cell):
            if entry.get("kind") != "label":
                continue
            try:
                rr = int(entry.get("observed_round"))
            except Exception:
                continue
            counts[rr] = counts.get(rr, 0) + 1
    return counts


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
            stage="init",
            reason="Campaign state is missing. Initialize before ingest/run operations.",
            labels_as_of=target_as_of,
            observed_round=target_observed,
            labels_in_observed_round=0,
            state_exists=False,
            run_exists_for_labels_as_of=False,
            next_commands=[
                f"opal init -c {ws.config_path}",
                f"opal validate -c {ws.config_path}",
            ],
            learn_more=["docs/reference/cli.md", "docs/workflows"],
        )

    labels_by_round = _label_counts_by_round(store, df)
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
                f"opal ingest-y -c {ws.config_path} --observed-round {target_observed} --in <labels.xlsx> --apply",
            ],
            learn_more=["docs/reference/cli.md", "docs/plugins/objective-sfxi.md"],
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
