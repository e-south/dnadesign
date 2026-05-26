"""Scratch execution helpers for the study-owned DenseGen axis OPAL probe."""

from __future__ import annotations

from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence

from .artifacts import ProbeArtifactLayout, ProbePlan
from .constants import CANDIDATE_RECORDS, NULL_ORACLE_ID, ORACLE_ID, RUN_STAGES
from .paths import _resolve_repo_path


def materialize_probe_inputs(
    *,
    repo_root: Path,
    plan: ProbePlan,
    labels: Any,
    null_labels: Any,
    split_metadata: Mapping[str, Mapping[str, Any]],
    label_family_manifest: Mapping[str, Any] | None = None,
    null_provenance: Mapping[str, Any] | None = None,
    suite_manifest: Mapping[str, Any] | None = None,
) -> None:
    import pandas as pd

    from .decision_inputs import _persisted_split_metadata
    from .plan import validate_scratch_paths
    from .scratch import (
        _make_training_input_for_run,
        _write_campaign_config,
        _write_candidate_scope,
        _write_json,
        _write_parquet,
        _write_records_reference,
    )

    layout = ProbeArtifactLayout(plan.run_root)
    for path in (layout.labels_dir, layout.splits_dir, layout.reports_dir, layout.scratch_campaigns_dir):
        path.mkdir(parents=True, exist_ok=True)
    _write_parquet(layout.densegen_labels_path, labels)
    _write_parquet(layout.null_labels_path, null_labels)
    if label_family_manifest is not None:
        _write_json(layout.label_family_manifest_path, label_family_manifest)
    if null_provenance is not None:
        _write_json(layout.null_provenance_path, null_provenance)
    if suite_manifest is not None:
        _write_json(layout.suite_manifest_path, suite_manifest)
    _write_json(layout.split_metadata_path, _persisted_split_metadata(split_metadata))
    for split_id, metadata in split_metadata.items():
        _write_parquet(layout.train_ids_path(split_id), pd.DataFrame({"id": list(metadata["train_ids"])}))
        _write_parquet(layout.eval_ids_path(split_id), pd.DataFrame({"id": list(metadata["eval_ids"])}))

    if plan.runs:
        source_records = _resolve_repo_path(repo_root, CANDIDATE_RECORDS)
        for split_id, metadata in split_metadata.items():
            split_ids = {
                *map(str, metadata.get("train_ids", [])),
                *map(str, metadata.get("eval_ids", [])),
            }
            _write_records_reference(source_records, layout.split_records_path(split_id))
            _write_candidate_scope(layout.split_candidate_scope_path(split_id), ids=sorted(split_ids))

    labels_by_oracle = {ORACLE_ID: labels, NULL_ORACLE_ID: null_labels}
    _write_campaign_collection_manifest(layout=layout, plan=plan)
    for run in plan.runs:
        validate_scratch_paths(run_root=plan.run_root, label_sidecar_path=run.sidecar_path)
        _write_campaign_config(repo_root, run, plan.run_root)
        train_ids = split_metadata[run.split_id]["train_ids"]
        training_input = _make_training_input_for_run(labels_by_oracle[run.oracle_id], train_ids, run)
        _write_parquet(layout.campaign_label_input_path(run.run_key, 0), training_input)


def _write_campaign_collection_manifest(*, layout: ProbeArtifactLayout, plan: ProbePlan) -> None:
    grouped_roles: dict[tuple[str, str, str, int], set[str]] = defaultdict(set)
    for run in plan.runs:
        grouped_roles[(run.campaign_key, run.label_family_id, run.split_id, int(run.seed))].add(
            "positive" if run.oracle_id == ORACLE_ID else "null"
        )
    has_pair = any({"positive", "null"}.issubset(roles) for roles in grouped_roles.values())
    relationships = []
    if has_pair:
        relationships.append(
            {
                "kind": "control_pair",
                "left_role": "positive",
                "right_role": "null",
                "match_on": ["target", "label_family_id", "label_split_id", "seed"],
                "replicate_on": ["seed"],
            }
        )
    from .scratch import _write_json

    _write_json(
        layout.campaign_collection_manifest_path,
        {
            "schema_version": "opal.campaign_collection.v1",
            "dimensions": ["target", "label_oracle_kind", "label_family_id", "label_split_id", "seed"],
            "relationships": relationships,
        },
    )


def selected_ids_from_round(
    run_key: str,
    workdir: Path,
    round_index: int,
    *,
    expected_k: int | None = None,
) -> list[str]:
    import pandas as pd

    selection_path = workdir / "outputs" / "rounds" / f"round_{int(round_index)}" / "selection" / "selection_top_k.csv"
    if not selection_path.exists():
        raise RuntimeError(f"selection artifact missing for {run_key} round {int(round_index)}: {selection_path}")
    try:
        frame = pd.read_csv(selection_path, usecols=["id"])
    except ValueError as exc:
        raise RuntimeError(
            f"selection artifact missing id column for {run_key} round {int(round_index)}: {selection_path}"
        ) from exc
    if frame["id"].isna().any():
        raise RuntimeError(f"selection artifact contains null id values for {run_key} round {int(round_index)}")
    ids = [str(value).strip() for value in frame["id"].tolist()]
    if any(not candidate_id for candidate_id in ids):
        raise RuntimeError(f"selection artifact contains blank id values for {run_key} round {int(round_index)}")
    if not ids:
        raise RuntimeError(f"selection artifact contains no selected ids for {run_key} round {int(round_index)}")
    id_series = pd.Series(ids, dtype="string")
    duplicated = id_series.loc[id_series.duplicated()].drop_duplicates().sort_values()
    if not duplicated.empty:
        preview = ", ".join(duplicated.head(5).tolist())
        suffix = "" if len(duplicated) <= 5 else f", ... ({len(duplicated)} total)"
        raise RuntimeError(
            f"selection artifact contains duplicate selected id(s) for {run_key} round {int(round_index)}: "
            f"{preview}{suffix}"
        )
    if expected_k is not None and len(ids) != int(expected_k):
        raise RuntimeError(
            f"selection artifact for {run_key} round {int(round_index)} expected {int(expected_k)} "
            f"selected id(s), got {len(ids)}"
        )
    return ids


def _observed_label_ids_for_round(sidecar_path: Path, round_index: int) -> set[str]:
    import pandas as pd

    if not sidecar_path.exists():
        return set()
    try:
        frame = pd.read_parquet(sidecar_path, columns=["id", "observed_round"])
    except (KeyError, ValueError) as exc:
        raise RuntimeError(f"observed-label sidecar missing id/observed_round columns: {sidecar_path}") from exc
    round_frame = frame.loc[pd.to_numeric(frame["observed_round"], errors="coerce") == int(round_index)]
    if round_frame.empty:
        return set()
    if round_frame["id"].isna().any():
        raise RuntimeError(f"observed-label sidecar has null id for round {int(round_index)}: {sidecar_path}")
    ids = {str(value).strip() for value in round_frame["id"].tolist()}
    if any(not candidate_id for candidate_id in ids):
        raise RuntimeError(f"observed-label sidecar has blank id for round {int(round_index)}: {sidecar_path}")
    return ids


def _round_selection_exists(workdir: Path, round_index: int) -> bool:
    return (workdir / "outputs" / "rounds" / f"round_{int(round_index)}" / "selection" / "selection_top_k.csv").exists()


def _campaign_has_mutable_state(run: Any) -> bool:
    return run.sidecar_path.exists() or (run.workdir / "outputs" / "rounds").exists()


def write_followup_label_input(
    *,
    layout: ProbeArtifactLayout,
    run,
    labels: Any,
    selected_ids: Sequence[str],
    already_labeled: set[str],
    round_index: int,
) -> list[str]:
    from .scratch import _make_training_input_for_run, _write_parquet

    run_key = str(run.run_key)

    raw_selected = list(selected_ids)
    if any(candidate_id is None for candidate_id in raw_selected):
        raise RuntimeError(f"round {int(round_index)} follow-up labels for {run_key} had null ids")
    selected = [str(candidate_id).strip() for candidate_id in raw_selected]
    if any(not candidate_id for candidate_id in selected):
        raise RuntimeError(f"round {int(round_index)} follow-up labels for {run_key} had blank ids")
    duplicate_ids = sorted(candidate_id for candidate_id, count in Counter(selected).items() if count > 1)
    if duplicate_ids:
        preview = ", ".join(duplicate_ids[:5])
        suffix = "" if len(duplicate_ids) <= 5 else f", ... ({len(duplicate_ids)} total)"
        raise RuntimeError(
            f"round {int(round_index)} follow-up labels for {run_key} had duplicate ids: {preview}{suffix}"
        )
    new_ids = [candidate_id for candidate_id in selected if candidate_id not in already_labeled]
    if not new_ids:
        raise RuntimeError(f"round {int(round_index)} follow-up labels for {run_key} had no newly selected ids")
    label_input = _make_training_input_for_run(labels, new_ids, run)
    _write_parquet(layout.campaign_label_input_path(run_key, int(round_index)), label_input)
    already_labeled.update(new_ids)
    return new_ids


def run_opal_rounds_for_probe(
    *,
    repo_root: Path,
    plan: ProbePlan,
    labels_by_oracle: Mapping[str, Any],
    split_metadata: Mapping[str, Mapping[str, Any]],
    machine_readable: bool,
) -> dict[str, set[str]]:
    from .plan import (
        _opal_ingest_command,
        _opal_init_command,
        _opal_run_command,
        _opal_status_command,
        _opal_validate_command,
    )
    from .scratch import _run_command

    labeled_ids_by_run: dict[str, set[str]] = {}
    if RUN_STAGES.index(plan.stop_after) < RUN_STAGES.index("validate"):
        return labeled_ids_by_run
    layout = ProbeArtifactLayout(plan.run_root)
    for run in plan.runs:
        labeled_ids = set(map(str, split_metadata[run.split_id]["train_ids"]))
        labeled_ids_by_run[run.run_key] = labeled_ids
        _run_command(_opal_validate_command(run.config_path), cwd=repo_root, machine_readable=machine_readable)
        if RUN_STAGES.index(plan.stop_after) < RUN_STAGES.index("init"):
            continue
        if not _campaign_has_mutable_state(run):
            _run_command(_opal_init_command(run.config_path), cwd=repo_root, machine_readable=machine_readable)
        if RUN_STAGES.index(plan.stop_after) < RUN_STAGES.index("ingest"):
            continue

        run_labels = labels_by_oracle[run.oracle_id]
        round_label_ids = _observed_label_ids_for_round(run.sidecar_path, 0)
        if round_label_ids:
            labeled_ids.update(round_label_ids)
        else:
            _run_command(_opal_ingest_command(run.config_path, 0), cwd=repo_root, machine_readable=machine_readable)
            labeled_ids.update(map(str, split_metadata[run.split_id]["train_ids"]))
        if RUN_STAGES.index(plan.stop_after) < RUN_STAGES.index("run"):
            continue

        for round_index in range(plan.rounds):
            if round_index > 0:
                round_label_ids = _observed_label_ids_for_round(run.sidecar_path, round_index)
                if round_label_ids:
                    labeled_ids.update(round_label_ids)
                else:
                    selected_ids = selected_ids_from_round(
                        run.run_key,
                        run.workdir,
                        round_index - 1,
                        expected_k=run.selection_k,
                    )
                    write_followup_label_input(
                        layout=layout,
                        run=run,
                        labels=run_labels,
                        selected_ids=selected_ids,
                        already_labeled=labeled_ids,
                        round_index=round_index,
                    )
                    _run_command(
                        _opal_ingest_command(run.config_path, round_index),
                        cwd=repo_root,
                        machine_readable=machine_readable,
                    )
            if _round_selection_exists(run.workdir, round_index):
                selected_ids_from_round(run.run_key, run.workdir, round_index, expected_k=run.selection_k)
            else:
                _run_command(
                    _opal_run_command(run.config_path, round_index),
                    cwd=repo_root,
                    machine_readable=machine_readable,
                )

        if RUN_STAGES.index(plan.stop_after) >= RUN_STAGES.index("status"):
            _run_command(_opal_status_command(run.config_path), cwd=repo_root, machine_readable=machine_readable)
    return labeled_ids_by_run
