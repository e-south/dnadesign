"""
--------------------------------------------------------------------------------
<cruncher project>
src/dnadesign/cruncher/tests/portfolio/test_portfolio_run_smoke.py

Smoke tests for Portfolio run aggregation workflow.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import json
import threading
from pathlib import Path
from types import SimpleNamespace

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest
import yaml

import dnadesign.cruncher.app.portfolio_materialization as portfolio_materialization
import dnadesign.cruncher.app.portfolio_workflow as portfolio_workflow
from dnadesign.cruncher.analysis.layout import analysis_root, summary_path
from dnadesign.cruncher.analysis.parquet import read_parquet, write_parquet
from dnadesign.cruncher.app.portfolio_workflow import portfolio_show_payload, run_portfolio
from dnadesign.cruncher.artifacts.layout import (
    elites_path,
    manifest_path,
    run_export_sequences_manifest_path,
    run_export_table_path,
)
from dnadesign.cruncher.core.pwm import PWM
from dnadesign.cruncher.portfolio.layout import portfolio_table_path
from dnadesign.cruncher.portfolio.plots.elite_showcase import plot_portfolio_elite_showcase
from dnadesign.cruncher.portfolio.schema_models import PortfolioRoot
from dnadesign.cruncher.study.identity import resolve_deterministic_study_run_dir
from dnadesign.cruncher.study.layout import study_manifest_path, study_status_path, study_table_path
from dnadesign.cruncher.study.manifest import StudyManifestV1, StudyStatusV1, write_study_manifest, write_study_status


def _seed_source_run(
    *,
    workspace: Path,
    run_rel: str,
    source_tag: str,
    tf_names: list[str],
    n_elites: int,
    stage: str = "sample",
    manifest_top_k: int | None = None,
) -> Path:
    run_dir = workspace / run_rel
    run_dir.mkdir(parents=True, exist_ok=True)

    manifest_payload = {
        "stage": stage,
        "run_dir": str(run_dir.resolve()),
        "run_group": source_tag,
        "regulator_set": {"index": 1, "tfs": tf_names},
        "created_at": "2026-02-20T00:00:00+00:00",
        "top_k": int(manifest_top_k) if manifest_top_k is not None else int(n_elites),
    }
    manifest_file = manifest_path(run_dir)
    manifest_file.parent.mkdir(parents=True, exist_ok=True)
    manifest_file.write_text(json.dumps(manifest_payload))
    config_used_file = run_dir / "meta" / "config_used.yaml"
    config_used_file.parent.mkdir(parents=True, exist_ok=True)
    pwm_template = [
        [0.70, 0.10, 0.10, 0.10],
        [0.10, 0.70, 0.10, 0.10],
        [0.10, 0.10, 0.70, 0.10],
        [0.10, 0.10, 0.10, 0.70],
        [0.25, 0.25, 0.25, 0.25],
        [0.40, 0.20, 0.20, 0.20],
    ]
    config_used_file.write_text(
        yaml.safe_dump({"cruncher": {"pwms_info": {tf_name: {"pwm_matrix": pwm_template} for tf_name in tf_names}}})
    )

    elites_rows = []
    windows_rows = []
    for rank in range(1, n_elites + 1):
        elite_id = f"{source_tag}_elite_{rank}"
        seq = ("ACGT" * 12)[:20]
        elites_rows.append(
            {
                "id": elite_id,
                "rank": rank,
                "sequence": seq,
                "combined_score_final": 0.90 - (rank * 0.01),
            }
        )
        for tf_idx, tf_name in enumerate(tf_names):
            start = tf_idx + rank
            end = start + 6
            windows_rows.append(
                {
                    "elite_id": elite_id,
                    "elite_rank": rank,
                    "elite_sequence": seq,
                    "tf": tf_name,
                    "best_start": start,
                    "best_end": end,
                    "best_strand": "+",
                    "best_window_seq": seq[start:end],
                    "best_core_seq": seq[start : start + 4],
                    "best_score_raw": 1.0 + rank,
                    "best_score_scaled": 0.8 + rank * 0.01,
                    "best_score_norm": 0.85 - rank * 0.01,
                    "pwm_ref": f"regulondb:{tf_name}_m1",
                    "pwm_hash": f"hash_{tf_name}",
                    "pwm_width": 6,
                    "core_width": 4,
                }
            )

    write_parquet(pd.DataFrame(elites_rows), elites_path(run_dir))
    export_elites_csv = run_export_table_path(run_dir, table_name="elites", fmt="csv")
    export_elites_csv.parent.mkdir(parents=True, exist_ok=True)
    export_rows: list[dict[str, object]] = []
    for elite_row in elites_rows:
        elite_id = str(elite_row["id"])
        elite_rank = int(elite_row["rank"])
        elite_sequence = str(elite_row["sequence"])
        members = [
            {
                "regulator_id": str(window["tf"]),
                "offset_start": int(window["best_start"]),
                "offset_end": int(window["best_end"]),
                "strand": str(window["best_strand"]),
                "window_kmer": str(window["best_window_seq"]),
                "core_kmer": str(window["best_core_seq"]),
                "score_name": "best_score_norm",
                "score": float(window["best_score_norm"]),
                "score_scaled_name": "best_score_scaled",
                "score_scaled": float(window["best_score_scaled"]),
                "score_raw_name": "best_score_raw",
                "score_raw": float(window["best_score_raw"]),
                "pwm_ref": str(window["pwm_ref"]),
                "pwm_hash": str(window["pwm_hash"]),
                "pwm_width": int(window["pwm_width"]),
                "core_width": int(window["core_width"]),
            }
            for window in windows_rows
            if str(window["elite_id"]) == elite_id
        ]
        export_rows.append(
            {
                "elite_id": elite_id,
                "elite_rank": elite_rank,
                "elite_sequence": elite_sequence,
                "sequence_length": len(elite_sequence),
                "window_count": len(members),
                "regulator_ids_csv": ",".join(sorted({str(member["regulator_id"]) for member in members})),
                "window_members_json": json.dumps(members, sort_keys=True),
                "combined_score_final": float(elite_row["combined_score_final"]),
            }
        )
    pd.DataFrame(export_rows).to_csv(export_elites_csv, index=False)

    export_manifest_file = run_export_sequences_manifest_path(run_dir)
    export_manifest_file.parent.mkdir(parents=True, exist_ok=True)
    consensus_file = export_manifest_file.parent / "table__consensus_sites.parquet"
    consensus_rows = []
    for tf_name in tf_names:
        consensus_rows.append(
            {
                "tf": tf_name,
                "motif_source": "regulondb",
                "motif_id": f"{tf_name}_m1",
                "pwm_ref": f"regulondb:{tf_name}_m1",
                "pwm_hash": f"hash_{tf_name}",
                "pwm_width": 6,
                "consensus_sequence": "ACGTAC",
                "consensus_width": 6,
            }
        )
    write_parquet(pd.DataFrame(consensus_rows), consensus_file)
    export_manifest_file.write_text(
        json.dumps(
            {
                "schema_version": 3,
                "kind": "sequence_export_v3",
                "table_format": "parquet",
                "files": {
                    "consensus_sites": "export/table__consensus_sites.parquet",
                    "elites": "export/table__elites.csv",
                },
                "row_counts": {
                    "consensus_sites": len(consensus_rows),
                    "elites": len(export_rows),
                },
            }
        )
    )

    analysis_summary_file = summary_path(analysis_root(run_dir))
    analysis_summary_file.parent.mkdir(parents=True, exist_ok=True)
    analysis_summary_file.write_text(
        json.dumps(
            {
                "analysis_id": f"analysis_{source_tag}",
                "run": run_dir.name,
                "tf_names": tf_names,
                "objective_components": {"best_score_final": 0.95},
            }
        )
    )
    return run_dir


def _seed_study_spec(
    *,
    workspace: Path,
    study_name: str,
    trials: list[dict[str, object]],
    replays_enabled: bool,
) -> Path:
    config_file = workspace / "configs" / "config.yaml"
    config_file.parent.mkdir(parents=True, exist_ok=True)
    if not config_file.exists():
        config_file.write_text(
            "cruncher: {schema_version: 3, workspace: {out_dir: outputs, regulator_sets: [[cpxR, baeR]]}}\n"
        )

    spec = workspace / "configs" / "studies" / f"{study_name}.study.yaml"
    spec.parent.mkdir(parents=True, exist_ok=True)
    spec.write_text(
        yaml.safe_dump(
            {
                "study": {
                    "schema_version": 3,
                    "name": study_name,
                    "base_config": "config.yaml",
                    "target": {"kind": "regulator_set", "set_index": 1},
                    "execution": {
                        "parallelism": 1,
                        "on_trial_error": "continue",
                        "exit_code_policy": "nonzero_if_any_error",
                        "summarize_after_run": True,
                    },
                    "artifacts": {"trial_output_profile": "minimal"},
                    "replicates": {"seed_path": "sample.seed", "seeds": [31]},
                    "trials": trials,
                    "replays": {
                        "mmr_sweep": {
                            "enabled": replays_enabled,
                            "pool_size_values": ["auto"],
                            "diversity_values": [0.0, 0.5, 1.0],
                        }
                    },
                }
            }
        )
    )
    return spec


def _seed_study_outputs(
    *,
    study_spec: Path,
    agg_rows: list[dict[str, object]],
    length_rows: list[dict[str, object]],
) -> Path:
    run_dir = resolve_deterministic_study_run_dir(study_spec)
    manifest = StudyManifestV1(
        study_name=study_spec.stem.replace(".study", ""),
        study_id=run_dir.name,
        spec_path=str(study_spec.resolve()),
        spec_sha256="spec",
        base_config_path=str((study_spec.parent.parent / "config.yaml").resolve()),
        base_config_sha256="cfg",
        created_at="2026-02-20T00:00:00+00:00",
        trial_runs=[],
    )
    status = StudyStatusV1(
        study_name=manifest.study_name,
        study_id=run_dir.name,
        status="completed",
        total_runs=1,
        pending_runs=0,
        running_runs=0,
        success_runs=1,
        error_runs=0,
        skipped_runs=0,
    )
    study_manifest_path(run_dir).parent.mkdir(parents=True, exist_ok=True)
    write_study_manifest(study_manifest_path(run_dir), manifest)
    write_study_status(study_status_path(run_dir), status)
    write_parquet(pd.DataFrame(agg_rows), study_table_path(run_dir, "trial_metrics_agg", "parquet"))
    write_parquet(pd.DataFrame(length_rows), study_table_path(run_dir, "length_tradeoff_agg", "parquet"))
    return run_dir


def test_select_portfolio_showcase_elites_supports_multiple_per_source() -> None:
    payload = {
        "portfolio": {
            "schema_version": 3,
            "name": "pairwise_handoff",
            "execution": {"mode": "aggregate_only"},
            "plots": {
                "elite_showcase": {
                    "top_n_per_source": 1,
                    "source_selectors": {
                        "pairwise_cpxr_baer": {"elite_ranks": [1, 3]},
                        "pairwise_cpxr_lexa": {"elite_ids": ["lexa_2", "lexa_4"]},
                    },
                }
            },
            "sources": [
                {
                    "id": "pairwise_cpxr_baer",
                    "workspace": "../pairwise_cpxr_baer",
                    "run_dir": "outputs/set1_cpxr_baer",
                },
                {
                    "id": "pairwise_cpxr_lexa",
                    "workspace": "../pairwise_cpxr_lexa",
                    "run_dir": "outputs/set1_cpxr_lexa",
                },
            ],
        }
    }
    spec = PortfolioRoot.model_validate(payload).portfolio
    elite_summary_df = pd.DataFrame(
        [
            {
                "source_id": "pairwise_cpxr_baer",
                "source_label": "pairwise_cpxr_baer",
                "elite_id": "baer_1",
                "elite_rank": 1,
                "sequence": "ACGTACGT",
            },
            {
                "source_id": "pairwise_cpxr_baer",
                "source_label": "pairwise_cpxr_baer",
                "elite_id": "baer_2",
                "elite_rank": 2,
                "sequence": "ACGTACGA",
            },
            {
                "source_id": "pairwise_cpxr_baer",
                "source_label": "pairwise_cpxr_baer",
                "elite_id": "baer_3",
                "elite_rank": 3,
                "sequence": "ACGTACGC",
            },
            {
                "source_id": "pairwise_cpxr_lexa",
                "source_label": "pairwise_cpxr_lexa",
                "elite_id": "lexa_1",
                "elite_rank": 1,
                "sequence": "TTTTACGT",
            },
            {
                "source_id": "pairwise_cpxr_lexa",
                "source_label": "pairwise_cpxr_lexa",
                "elite_id": "lexa_2",
                "elite_rank": 2,
                "sequence": "TTTTACGA",
            },
            {
                "source_id": "pairwise_cpxr_lexa",
                "source_label": "pairwise_cpxr_lexa",
                "elite_id": "lexa_3",
                "elite_rank": 3,
                "sequence": "TTTTACGC",
            },
            {
                "source_id": "pairwise_cpxr_lexa",
                "source_label": "pairwise_cpxr_lexa",
                "elite_id": "lexa_4",
                "elite_rank": 4,
                "sequence": "TTTTACGG",
            },
        ]
    )

    selected = portfolio_workflow._select_portfolio_showcase_elites(spec, elite_summary_df)
    selected_keys = list(selected[["source_id", "elite_id"]].itertuples(index=False, name=None))
    assert selected_keys == [
        ("pairwise_cpxr_baer", "baer_1"),
        ("pairwise_cpxr_baer", "baer_3"),
        ("pairwise_cpxr_lexa", "lexa_2"),
        ("pairwise_cpxr_lexa", "lexa_4"),
    ]


def test_select_portfolio_showcase_elites_defaults_to_all_per_source() -> None:
    payload = {
        "portfolio": {
            "schema_version": 3,
            "name": "pairwise_handoff",
            "execution": {"mode": "aggregate_only"},
            "sources": [
                {
                    "id": "pairwise_cpxr_baer",
                    "workspace": "../pairwise_cpxr_baer",
                    "run_dir": "outputs/set1_cpxr_baer",
                },
                {
                    "id": "pairwise_cpxr_lexa",
                    "workspace": "../pairwise_cpxr_lexa",
                    "run_dir": "outputs/set1_cpxr_lexa",
                },
            ],
        }
    }
    spec = PortfolioRoot.model_validate(payload).portfolio
    elite_summary_df = pd.DataFrame(
        [
            {
                "source_id": "pairwise_cpxr_baer",
                "source_label": "pairwise_cpxr_baer",
                "elite_id": "baer_1",
                "elite_rank": 1,
                "sequence": "ACGTACGT",
            },
            {
                "source_id": "pairwise_cpxr_baer",
                "source_label": "pairwise_cpxr_baer",
                "elite_id": "baer_2",
                "elite_rank": 2,
                "sequence": "ACGTACGA",
            },
            {
                "source_id": "pairwise_cpxr_lexa",
                "source_label": "pairwise_cpxr_lexa",
                "elite_id": "lexa_1",
                "elite_rank": 1,
                "sequence": "TTTTACGT",
            },
            {
                "source_id": "pairwise_cpxr_lexa",
                "source_label": "pairwise_cpxr_lexa",
                "elite_id": "lexa_2",
                "elite_rank": 2,
                "sequence": "TTTTACGA",
            },
        ]
    )

    selected = portfolio_workflow._select_portfolio_showcase_elites(spec, elite_summary_df)
    selected_keys = list(selected[["source_id", "elite_id"]].itertuples(index=False, name=None))
    assert selected_keys == [
        ("pairwise_cpxr_baer", "baer_1"),
        ("pairwise_cpxr_baer", "baer_2"),
        ("pairwise_cpxr_lexa", "lexa_1"),
        ("pairwise_cpxr_lexa", "lexa_2"),
    ]


def test_mean_pairwise_hamming_bp_handles_variable_length_sequences() -> None:
    observed = portfolio_workflow._mean_pairwise_hamming_bp(["AAAA", "AAA", "AATA"])
    assert observed is not None
    assert observed == pytest.approx((1 + 1 + 2) / 3)


def test_select_portfolio_showcase_elites_rejects_missing_requested_rank() -> None:
    payload = {
        "portfolio": {
            "schema_version": 3,
            "name": "pairwise_handoff",
            "execution": {"mode": "aggregate_only"},
            "plots": {
                "elite_showcase": {
                    "source_selectors": {
                        "pairwise_cpxr_baer": {"elite_ranks": [4]},
                    },
                }
            },
            "sources": [
                {
                    "id": "pairwise_cpxr_baer",
                    "workspace": "../pairwise_cpxr_baer",
                    "run_dir": "outputs/set1_cpxr_baer",
                }
            ],
        }
    }
    spec = PortfolioRoot.model_validate(payload).portfolio
    elite_summary_df = pd.DataFrame(
        [
            {
                "source_id": "pairwise_cpxr_baer",
                "source_label": "pairwise_cpxr_baer",
                "elite_id": "baer_1",
                "elite_rank": 1,
                "sequence": "ACGTACGT",
            },
            {
                "source_id": "pairwise_cpxr_baer",
                "source_label": "pairwise_cpxr_baer",
                "elite_id": "baer_2",
                "elite_rank": 2,
                "sequence": "ACGTACGA",
            },
        ]
    )

    with pytest.raises(ValueError, match="unknown elite ranks"):
        portfolio_workflow._select_portfolio_showcase_elites(spec, elite_summary_df)


def test_portfolio_elite_showcase_plot_handles_reverse_strand_windows(tmp_path: Path) -> None:
    selected_elites_df = pd.DataFrame(
        [
            {
                "source_id": "demo_pairwise",
                "source_label": "demo_pairwise",
                "elite_id": "demo_pairwise_elite_1",
                "sequence": "AACCGGTTACGA",
            }
        ]
    )
    handoff_df = pd.DataFrame(
        [
            {
                "source_id": "demo_pairwise",
                "elite_id": "demo_pairwise_elite_1",
                "tf": "lexA",
                "best_start": 0,
                "best_end": 5,
                "best_strand": "-",
                "best_window_seq": "AACCG",
                "best_score_norm": 0.82,
            }
        ]
    )
    pwms_by_source = {
        "demo_pairwise": {
            "lexA": PWM(
                name="lexA",
                matrix=np.array(
                    [
                        [0.6, 0.1, 0.2, 0.1],
                        [0.1, 0.6, 0.2, 0.1],
                        [0.1, 0.1, 0.7, 0.1],
                        [0.1, 0.2, 0.1, 0.6],
                        [0.25, 0.25, 0.25, 0.25],
                    ],
                    dtype=float,
                ),
            )
        }
    }
    out_path = tmp_path / "portfolio_showcase.pdf"

    plot_portfolio_elite_showcase(
        selected_elites_df=selected_elites_df,
        handoff_df=handoff_df,
        pwms_by_source=pwms_by_source,
        out_path=out_path,
        ncols=5,
        dpi=250,
    )

    assert out_path.exists()


def test_portfolio_elite_showcase_uses_motif_logo_effects_and_multiline_scores(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    selected_elites_df = pd.DataFrame(
        [
            {
                "source_id": "demo_pairwise",
                "source_label": "Demo Pairwise",
                "elite_id": "demo_pairwise_elite_1",
                "sequence": "AACCGGTTACGA",
            }
        ]
    )
    handoff_df = pd.DataFrame(
        [
            {
                "source_id": "demo_pairwise",
                "elite_id": "demo_pairwise_elite_1",
                "tf": "lexA",
                "best_start": 1,
                "best_end": 5,
                "best_strand": "+",
                "best_window_seq": "ACCG",
                "best_score_norm": 0.88,
            }
        ]
    )
    pwms_by_source = {
        "demo_pairwise": {
            "lexA": PWM(
                name="lexA",
                matrix=np.array(
                    [
                        [0.6, 0.1, 0.2, 0.1],
                        [0.1, 0.6, 0.2, 0.1],
                        [0.1, 0.1, 0.7, 0.1],
                        [0.1, 0.2, 0.1, 0.6],
                    ],
                    dtype=float,
                ),
            )
        }
    }
    seen: dict[str, object] = {}

    def _fake_grid(records, *, ncols, style_overrides):
        rows = list(records)
        seen["records"] = rows
        seen["ncols"] = int(ncols)
        return plt.figure(figsize=(2, 2), dpi=100)

    monkeypatch.setattr(
        "dnadesign.cruncher.portfolio.plots.elite_showcase.render_record_grid_figure",
        _fake_grid,
    )

    out_path = tmp_path / "portfolio_showcase_effects.pdf"
    plot_portfolio_elite_showcase(
        selected_elites_df=selected_elites_df,
        handoff_df=handoff_df,
        pwms_by_source=pwms_by_source,
        out_path=out_path,
        ncols=5,
        dpi=250,
    )

    assert out_path.exists()
    assert seen["ncols"] == 5
    records = seen["records"]
    assert isinstance(records, list) and len(records) == 1
    record = records[0]
    assert any(effect.kind == "motif_logo" for effect in record.effects)
    assert record.display.overlay_text.count("\n") >= 2
    assert "lexA=" in str(record.display.overlay_text)


def test_portfolio_elite_showcase_fails_when_source_pwms_missing(tmp_path: Path) -> None:
    selected_elites_df = pd.DataFrame(
        [
            {
                "source_id": "demo_pairwise",
                "source_label": "Demo Pairwise",
                "elite_id": "demo_pairwise_elite_1",
                "sequence": "AACCGGTTACGA",
            }
        ]
    )
    handoff_df = pd.DataFrame(
        [
            {
                "source_id": "demo_pairwise",
                "elite_id": "demo_pairwise_elite_1",
                "tf": "lexA",
                "best_start": 1,
                "best_end": 5,
                "best_strand": "+",
                "best_window_seq": "ACCG",
                "best_score_norm": 0.88,
            }
        ]
    )
    out_path = tmp_path / "portfolio_showcase_missing_pwms.pdf"

    with pytest.raises(ValueError, match="Missing source PWM set"):
        plot_portfolio_elite_showcase(
            selected_elites_df=selected_elites_df,
            handoff_df=handoff_df,
            pwms_by_source={},
            out_path=out_path,
            ncols=5,
            dpi=250,
        )


def test_portfolio_elite_showcase_fails_when_best_score_norm_out_of_bounds(tmp_path: Path) -> None:
    selected_elites_df = pd.DataFrame(
        [
            {
                "source_id": "demo_pairwise",
                "source_label": "Demo Pairwise",
                "elite_id": "demo_pairwise_elite_1",
                "sequence": "AACCGGTTACGA",
            }
        ]
    )
    handoff_df = pd.DataFrame(
        [
            {
                "source_id": "demo_pairwise",
                "elite_id": "demo_pairwise_elite_1",
                "tf": "lexA",
                "best_start": 1,
                "best_end": 5,
                "best_strand": "+",
                "best_window_seq": "ACCG",
                "best_score_norm": 1.4,
            }
        ]
    )
    pwms_by_source = {
        "demo_pairwise": {
            "lexA": PWM(
                name="lexA",
                matrix=np.array(
                    [
                        [0.6, 0.1, 0.2, 0.1],
                        [0.1, 0.6, 0.2, 0.1],
                        [0.1, 0.1, 0.7, 0.1],
                        [0.1, 0.2, 0.1, 0.6],
                    ],
                    dtype=float,
                ),
            )
        }
    }
    out_path = tmp_path / "portfolio_showcase_invalid_norm.pdf"

    with pytest.raises(ValueError, match="best_score_norm"):
        plot_portfolio_elite_showcase(
            selected_elites_df=selected_elites_df,
            handoff_df=handoff_df,
            pwms_by_source=pwms_by_source,
            out_path=out_path,
            ncols=5,
            dpi=250,
        )


def test_run_portfolio_fails_when_source_config_used_is_missing(tmp_path: Path) -> None:
    roots = tmp_path / "workspaces"
    source = roots / "pairwise_cpxr_baer"
    source.mkdir(parents=True, exist_ok=True)
    run_dir = _seed_source_run(
        workspace=source,
        run_rel="outputs/set1_cpxr_baer",
        source_tag="cpxr_baer",
        tf_names=["cpxR", "baeR"],
        n_elites=2,
    )
    config_used = run_dir / "meta" / "config_used.yaml"
    if config_used.exists():
        config_used.unlink()

    portfolio_workspace = roots / "portfolio_pairwise_handoff"
    portfolio_workspace.mkdir(parents=True, exist_ok=True)
    spec_path = portfolio_workspace / "pairwise_handoff.portfolio.yaml"
    spec_path.write_text(
        yaml.safe_dump(
            {
                "portfolio": {
                    "schema_version": 3,
                    "name": "pairwise_handoff",
                    "execution": {"mode": "aggregate_only"},
                    "artifacts": {"table_format": "parquet"},
                    "sources": [
                        {
                            "id": "pairwise_cpxr_baer",
                            "workspace": "../pairwise_cpxr_baer",
                            "run_dir": "outputs/set1_cpxr_baer",
                        }
                    ],
                }
            }
        )
    )

    with pytest.raises(FileNotFoundError, match="config_used.yaml"):
        run_portfolio(spec_path)


def test_run_portfolio_writes_handoff_tables_and_payload(tmp_path: Path) -> None:
    roots = tmp_path / "workspaces"
    source_a = roots / "pairwise_cpxr_baer"
    source_b = roots / "pairwise_cpxr_lexa"
    source_a.mkdir(parents=True, exist_ok=True)
    source_b.mkdir(parents=True, exist_ok=True)

    _seed_source_run(
        workspace=source_a,
        run_rel="outputs/set1_cpxr_baer",
        source_tag="cpxr_baer",
        tf_names=["cpxR", "baeR"],
        n_elites=4,
    )
    _seed_source_run(
        workspace=source_b,
        run_rel="outputs/set1_cpxr_lexa",
        source_tag="cpxr_lexa",
        tf_names=["cpxR", "lexA"],
        n_elites=5,
    )

    portfolio_workspace = roots / "portfolio_pairwise_handoff"
    portfolio_workspace.mkdir(parents=True, exist_ok=True)
    spec_path = portfolio_workspace / "pairwise_handoff.portfolio.yaml"
    spec_path.write_text(
        yaml.safe_dump(
            {
                "portfolio": {
                    "schema_version": 3,
                    "name": "pairwise_handoff",
                    "execution": {"mode": "aggregate_only"},
                    "artifacts": {"table_format": "parquet", "write_csv": True},
                    "sources": [
                        {
                            "id": "pairwise_cpxr_baer",
                            "workspace": "../pairwise_cpxr_baer",
                            "run_dir": "outputs/set1_cpxr_baer",
                        },
                        {
                            "id": "pairwise_cpxr_lexa",
                            "workspace": "../pairwise_cpxr_lexa",
                            "run_dir": "outputs/set1_cpxr_lexa",
                        },
                    ],
                }
            }
        )
    )

    run_dir = run_portfolio(spec_path)
    payload = portfolio_show_payload(run_dir)

    handoff_path = portfolio_table_path(run_dir, "handoff_windows_long", "parquet")
    elite_summary_path = portfolio_table_path(run_dir, "handoff_elites_summary", "parquet")
    source_summary_path = portfolio_table_path(run_dir, "source_summary", "parquet")
    assert handoff_path.exists()
    assert elite_summary_path.exists()
    assert source_summary_path.exists()

    handoff_df = read_parquet(handoff_path)
    elite_summary_df = read_parquet(elite_summary_path)
    source_summary_df = read_parquet(source_summary_path)
    assert len(handoff_df) == (4 * 2) + (5 * 2)
    assert len(elite_summary_df) == 9
    assert handoff_df["elite_hash_id"].astype(str).str.len().eq(16).all()
    assert handoff_df["window_hash_id"].astype(str).str.len().eq(16).all()
    assert elite_summary_df["elite_hash_id"].astype(str).str.len().eq(16).all()
    for required_column in [
        "source_id",
        "run_dir",
        "elite_hash_id",
        "window_hash_id",
        "sequence",
        "sequence_length",
        "tf",
        "best_start",
        "best_end",
        "best_strand",
        "best_score_norm",
    ]:
        assert required_column in handoff_df.columns
    for required_column in [
        "source_id",
        "run_dir",
        "elite_hash_id",
        "sequence",
        "sequence_length",
        "min_best_score_norm",
        "mean_best_score_norm",
    ]:
        assert required_column in elite_summary_df.columns
    assert sorted(source_summary_df["source_id"].astype(str).tolist()) == [
        "pairwise_cpxr_baer",
        "pairwise_cpxr_lexa",
    ]
    manifest_payload = json.loads((run_dir / "portfolio" / "portfolio_manifest.json").read_text())
    assert manifest_payload["source_runs"][0]["source_top_k"] == 4
    assert manifest_payload["source_runs"][1]["source_top_k"] == 5
    showcase_plot = portfolio_workflow.portfolio_plot_path(run_dir, "elite_showcase_cross_workspace", "pdf")
    assert showcase_plot.exists()
    assert str(showcase_plot) in payload["plot_paths"]
    assert payload["status"] == "completed"
    assert payload["n_sources"] == 2
    assert payload["n_selected_elites"] == 9


def test_run_portfolio_reads_elites_from_export_contract_only(tmp_path: Path) -> None:
    roots = tmp_path / "workspaces"
    source = roots / "pairwise_cpxr_baer"
    source.mkdir(parents=True, exist_ok=True)
    run_dir = _seed_source_run(
        workspace=source,
        run_rel="outputs/set1_cpxr_baer",
        source_tag="cpxr_baer",
        tf_names=["cpxR", "baeR"],
        n_elites=3,
    )
    elites_file = elites_path(run_dir)
    if elites_file.exists():
        elites_file.unlink()

    portfolio_workspace = roots / "portfolio_pairwise_handoff"
    portfolio_workspace.mkdir(parents=True, exist_ok=True)
    spec_path = portfolio_workspace / "pairwise_handoff.portfolio.yaml"
    spec_path.write_text(
        yaml.safe_dump(
            {
                "portfolio": {
                    "schema_version": 3,
                    "name": "pairwise_handoff",
                    "execution": {"mode": "aggregate_only"},
                    "artifacts": {"table_format": "parquet"},
                    "sources": [
                        {
                            "id": "pairwise_cpxr_baer",
                            "workspace": "../pairwise_cpxr_baer",
                            "run_dir": "outputs/set1_cpxr_baer",
                        }
                    ],
                }
            }
        )
    )

    portfolio_run_dir = run_portfolio(spec_path)
    handoff_path = portfolio_table_path(portfolio_run_dir, "handoff_windows_long", "parquet")
    elite_summary_path = portfolio_table_path(portfolio_run_dir, "handoff_elites_summary", "parquet")

    assert handoff_path.exists()
    assert elite_summary_path.exists()
    handoff_df = read_parquet(handoff_path)
    elite_summary_df = read_parquet(elite_summary_path)
    assert len(handoff_df) == 6
    assert len(elite_summary_df) == 3


def test_run_portfolio_accepts_nondefault_export_elites_path_from_manifest(tmp_path: Path) -> None:
    roots = tmp_path / "workspaces"
    source = roots / "pairwise_cpxr_baer"
    source.mkdir(parents=True, exist_ok=True)
    run_dir = _seed_source_run(
        workspace=source,
        run_rel="outputs/set1_cpxr_baer",
        source_tag="cpxr_baer",
        tf_names=["cpxR", "baeR"],
        n_elites=3,
    )

    default_elites_export = run_export_table_path(run_dir, table_name="elites", fmt="csv")
    custom_elites_export = default_elites_export.with_name("table__elites_contract.csv")
    default_elites_export.replace(custom_elites_export)
    export_manifest_file = run_export_sequences_manifest_path(run_dir)
    export_manifest = json.loads(export_manifest_file.read_text())
    export_manifest["files"]["elites"] = "export/table__elites_contract.csv"
    export_manifest_file.write_text(json.dumps(export_manifest))

    portfolio_workspace = roots / "portfolio_pairwise_handoff"
    portfolio_workspace.mkdir(parents=True, exist_ok=True)
    spec_path = portfolio_workspace / "pairwise_handoff.portfolio.yaml"
    spec_path.write_text(
        yaml.safe_dump(
            {
                "portfolio": {
                    "schema_version": 3,
                    "name": "pairwise_handoff",
                    "execution": {"mode": "aggregate_only"},
                    "artifacts": {"table_format": "parquet"},
                    "sources": [
                        {
                            "id": "pairwise_cpxr_baer",
                            "workspace": "../pairwise_cpxr_baer",
                            "run_dir": "outputs/set1_cpxr_baer",
                        }
                    ],
                }
            }
        )
    )

    run_dir_out = run_portfolio(spec_path)
    handoff_path = portfolio_table_path(run_dir_out, "handoff_windows_long", "parquet")
    assert handoff_path.exists()


def test_run_portfolio_default_artifacts_write_single_table_format_only(tmp_path: Path) -> None:
    roots = tmp_path / "workspaces"
    source = roots / "pairwise_cpxr_baer"
    source.mkdir(parents=True, exist_ok=True)
    _seed_source_run(
        workspace=source,
        run_rel="outputs/set1_cpxr_baer",
        source_tag="cpxr_baer",
        tf_names=["cpxR", "baeR"],
        n_elites=2,
    )

    portfolio_workspace = roots / "portfolio_pairwise_handoff"
    portfolio_workspace.mkdir(parents=True, exist_ok=True)
    spec_path = portfolio_workspace / "pairwise_handoff.portfolio.yaml"
    spec_path.write_text(
        yaml.safe_dump(
            {
                "portfolio": {
                    "schema_version": 3,
                    "name": "pairwise_handoff",
                    "execution": {"mode": "aggregate_only"},
                    "sources": [
                        {
                            "id": "pairwise_cpxr_baer",
                            "workspace": "../pairwise_cpxr_baer",
                            "run_dir": "outputs/set1_cpxr_baer",
                        }
                    ],
                }
            }
        )
    )

    run_dir = run_portfolio(spec_path)
    assert portfolio_table_path(run_dir, "handoff_windows_long", "parquet").exists()
    assert portfolio_table_path(run_dir, "handoff_elites_summary", "parquet").exists()
    assert portfolio_table_path(run_dir, "source_summary", "parquet").exists()
    assert not portfolio_table_path(run_dir, "handoff_windows_long", "csv").exists()
    assert not portfolio_table_path(run_dir, "handoff_elites_summary", "csv").exists()
    assert not portfolio_table_path(run_dir, "source_summary", "csv").exists()


def test_run_portfolio_flat_configs_spec_writes_outputs_under_workspace_root(tmp_path: Path) -> None:
    roots = tmp_path / "workspaces"
    source = roots / "pairwise_cpxr_baer"
    source.mkdir(parents=True, exist_ok=True)
    _seed_source_run(
        workspace=source,
        run_rel="outputs/set1_cpxr_baer",
        source_tag="cpxr_baer",
        tf_names=["cpxR", "baeR"],
        n_elites=3,
    )

    portfolio_workspace = roots / "portfolio_pairwise_handoff"
    portfolio_workspace.mkdir(parents=True, exist_ok=True)
    spec_path = portfolio_workspace / "configs" / "pairwise_handoff.portfolio.yaml"
    spec_path.parent.mkdir(parents=True, exist_ok=True)
    spec_path.write_text(
        yaml.safe_dump(
            {
                "portfolio": {
                    "schema_version": 3,
                    "name": "pairwise_handoff",
                    "execution": {"mode": "aggregate_only"},
                    "sources": [
                        {
                            "id": "pairwise_cpxr_baer",
                            "workspace": "../pairwise_cpxr_baer",
                            "run_dir": "outputs/set1_cpxr_baer",
                        }
                    ],
                }
            }
        )
    )

    run_dir = run_portfolio(spec_path)
    expected_root = (portfolio_workspace / "outputs" / "portfolios" / "pairwise_handoff").resolve()
    assert run_dir.parent == expected_root


def test_run_portfolio_writes_study_summary_table_when_study_specs_are_declared(tmp_path: Path) -> None:
    roots = tmp_path / "workspaces"
    source = roots / "pairwise_cpxr_baer"
    source.mkdir(parents=True, exist_ok=True)
    _seed_source_run(
        workspace=source,
        run_rel="outputs/set1_cpxr_baer",
        source_tag="cpxr_baer",
        tf_names=["cpxR", "baeR"],
        n_elites=4,
    )

    source_config = source / "configs" / "config.yaml"
    source_config.parent.mkdir(parents=True, exist_ok=True)
    source_config.write_text(
        "cruncher: {schema_version: 3, workspace: {out_dir: outputs, regulator_sets: [[cpxR, baeR]]}}\n"
    )

    study_spec = source / "configs" / "studies" / "diversity_vs_score.study.yaml"
    study_spec.parent.mkdir(parents=True, exist_ok=True)
    study_spec.write_text(
        yaml.safe_dump(
            {
                "study": {
                    "schema_version": 3,
                    "name": "diversity_vs_score",
                    "base_config": "config.yaml",
                    "target": {"kind": "regulator_set", "set_index": 1},
                    "execution": {
                        "parallelism": 1,
                        "on_trial_error": "continue",
                        "exit_code_policy": "nonzero_if_any_error",
                        "summarize_after_run": True,
                    },
                    "artifacts": {"trial_output_profile": "minimal"},
                    "replicates": {"seed_path": "sample.seed", "seeds": [7]},
                    "trials": [{"id": "BASE", "factors": {}}],
                    "replays": {
                        "mmr_sweep": {
                            "enabled": True,
                            "pool_size_values": ["auto"],
                            "diversity_values": [0.0, 0.5, 1.0],
                        }
                    },
                }
            }
        )
    )

    study_run_dir = resolve_deterministic_study_run_dir(study_spec)
    study_manifest = StudyManifestV1(
        study_name="diversity_vs_score",
        study_id=study_run_dir.name,
        spec_path=str(study_spec.resolve()),
        spec_sha256="spec",
        base_config_path=str(source_config.resolve()),
        base_config_sha256="cfg",
        created_at="2026-02-20T00:00:00+00:00",
        trial_runs=[],
    )
    study_status = StudyStatusV1(
        study_name="diversity_vs_score",
        study_id=study_run_dir.name,
        status="completed",
        total_runs=1,
        pending_runs=0,
        running_runs=0,
        success_runs=1,
        error_runs=0,
        skipped_runs=0,
    )
    study_manifest_path(study_run_dir).parent.mkdir(parents=True, exist_ok=True)
    write_study_manifest(study_manifest_path(study_run_dir), study_manifest)
    write_study_status(study_status_path(study_run_dir), study_status)
    write_parquet(
        pd.DataFrame(
            [
                {
                    "trial_id": "BASE",
                    "target_set_index": 1,
                    "target_tfs": "cpxR,baeR",
                    "series_label": "BASE",
                    "median_score_mean": 0.77,
                    "median_nn_full_bp_mean": 6.0,
                    "best_score_mean": 0.85,
                }
            ]
        ),
        study_table_path(study_run_dir, "trial_metrics_agg", "parquet"),
    )

    portfolio_workspace = roots / "portfolio_pairwise_handoff"
    portfolio_workspace.mkdir(parents=True, exist_ok=True)
    spec_path = portfolio_workspace / "configs" / "master.portfolio.yaml"
    spec_path.parent.mkdir(parents=True, exist_ok=True)
    spec_path.write_text(
        yaml.safe_dump(
            {
                "portfolio": {
                    "schema_version": 3,
                    "name": "master",
                    "execution": {"mode": "aggregate_only"},
                    "artifacts": {"table_format": "parquet", "write_csv": True},
                    "sources": [
                        {
                            "id": "pairwise_cpxr_baer",
                            "workspace": "../pairwise_cpxr_baer",
                            "run_dir": "outputs/set1_cpxr_baer",
                            "study_spec": "configs/studies/diversity_vs_score.study.yaml",
                        }
                    ],
                }
            }
        )
    )

    run_dir = run_portfolio(spec_path)
    study_summary_path = portfolio_table_path(run_dir, "study_summary", "parquet")
    assert study_summary_path.exists()
    study_summary_df = read_parquet(study_summary_path)
    assert len(study_summary_df) == 1
    assert study_summary_df.iloc[0]["source_id"] == "pairwise_cpxr_baer"
    assert study_summary_df.iloc[0]["study_name"] == "diversity_vs_score"


def test_run_portfolio_schema_v3_prepare_mode_runs_source_runbooks(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    roots = tmp_path / "workspaces"
    source = roots / "pairwise_cpxr_baer"
    source.mkdir(parents=True, exist_ok=True)
    _seed_source_run(
        workspace=source,
        run_rel="outputs/set1_cpxr_baer",
        source_tag="cpxr_baer",
        tf_names=["cpxR", "baeR"],
        n_elites=4,
    )

    runbook_file = source / "configs" / "runbook.yaml"
    runbook_file.parent.mkdir(parents=True, exist_ok=True)
    runbook_file.write_text(
        yaml.safe_dump(
            {
                "runbook": {
                    "schema_version": 1,
                    "name": "pairwise_cpxr_baer",
                    "steps": [
                        {"id": "analyze_summary", "run": ["analyze", "--summary", "-c", "configs/config.yaml"]},
                        {
                            "id": "export_sequences",
                            "run": ["export", "sequences", "--run", "outputs", "-c", "configs/config.yaml"],
                        },
                    ],
                }
            }
        )
    )

    portfolio_workspace = roots / "portfolio_pairwise_handoff"
    portfolio_workspace.mkdir(parents=True, exist_ok=True)
    spec_path = portfolio_workspace / "configs" / "pairwise_handoff.portfolio.yaml"
    spec_path.parent.mkdir(parents=True, exist_ok=True)
    spec_path.write_text(
        yaml.safe_dump(
            {
                "portfolio": {
                    "schema_version": 3,
                    "name": "pairwise_handoff",
                    "execution": {"mode": "prepare_then_aggregate"},
                    "sources": [
                        {
                            "id": "pairwise_cpxr_baer",
                            "workspace": "../pairwise_cpxr_baer",
                            "run_dir": "outputs/set1_cpxr_baer",
                            "prepare": {
                                "runbook": "configs/runbook.yaml",
                                "step_ids": ["analyze_summary", "export_sequences"],
                            },
                        }
                    ],
                }
            }
        )
    )

    calls: list[tuple[Path, list[str] | None, Path | None]] = []

    def _fake_run_workspace_runbook(
        path: Path,
        *,
        step_ids=None,
        dry_run: bool = False,
        output_log_path: Path | None = None,
    ):
        assert dry_run is False
        calls.append((path, list(step_ids or []), output_log_path))
        return SimpleNamespace(
            runbook_path=path,
            workspace_root=path.parent.parent,
            executed_step_ids=list(step_ids or []),
        )

    monkeypatch.setattr(portfolio_workflow, "run_workspace_runbook", _fake_run_workspace_runbook)

    run_dir = run_portfolio(spec_path)
    assert len(calls) == 1
    called_path, called_steps, output_log_path = calls[0]
    assert called_path == runbook_file.resolve()
    assert called_steps == ["analyze_summary", "export_sequences"]
    assert output_log_path == (run_dir / "portfolio" / "logs" / "prepare__pairwise_cpxr_baer.log")


def test_run_portfolio_prepare_mode_submits_multiple_sources_before_waiting(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    roots = tmp_path / "workspaces"
    source_a = roots / "pairwise_cpxr_baer"
    source_b = roots / "pairwise_cpxr_lexa"
    source_a.mkdir(parents=True, exist_ok=True)
    source_b.mkdir(parents=True, exist_ok=True)

    for workspace in (source_a, source_b):
        runbook_file = workspace / "configs" / "runbook.yaml"
        runbook_file.parent.mkdir(parents=True, exist_ok=True)
        runbook_file.write_text(
            yaml.safe_dump(
                {
                    "runbook": {
                        "schema_version": 1,
                        "name": workspace.name,
                        "steps": [
                            {"id": "sample_run", "run": ["sample", "--force-overwrite", "-c", "configs/config.yaml"]}
                        ],
                    }
                }
            )
        )

    portfolio_workspace = roots / "portfolio_pairwise_handoff"
    portfolio_workspace.mkdir(parents=True, exist_ok=True)
    spec_path = portfolio_workspace / "configs" / "pairwise_handoff.portfolio.yaml"
    spec_path.parent.mkdir(parents=True, exist_ok=True)
    spec_path.write_text(
        yaml.safe_dump(
            {
                "portfolio": {
                    "schema_version": 3,
                    "name": "pairwise_handoff",
                    "execution": {"mode": "prepare_then_aggregate", "max_parallel_sources": 2},
                    "sources": [
                        {
                            "id": "pairwise_cpxr_baer",
                            "workspace": "../pairwise_cpxr_baer",
                            "run_dir": "outputs/set1_cpxr_baer",
                            "prepare": {"runbook": "configs/runbook.yaml", "step_ids": ["sample_run"]},
                        },
                        {
                            "id": "pairwise_cpxr_lexa",
                            "workspace": "../pairwise_cpxr_lexa",
                            "run_dir": "outputs/set1_cpxr_lexa",
                            "prepare": {"runbook": "configs/runbook.yaml", "step_ids": ["sample_run"]},
                        },
                    ],
                }
            }
        )
    )

    second_started = threading.Event()

    def _fake_run_workspace_runbook(
        path: Path,
        *,
        step_ids=None,
        dry_run: bool = False,
        output_log_path: Path | None = None,
    ):
        assert dry_run is False
        workspace_name = path.parent.parent.name
        if workspace_name == "pairwise_cpxr_baer":
            if not second_started.wait(timeout=1.0):
                raise RuntimeError("second source prepare did not start before first waited")
            _seed_source_run(
                workspace=source_a,
                run_rel="outputs/set1_cpxr_baer",
                source_tag="cpxr_baer",
                tf_names=["cpxR", "baeR"],
                n_elites=3,
            )
        else:
            second_started.set()
            _seed_source_run(
                workspace=source_b,
                run_rel="outputs/set1_cpxr_lexa",
                source_tag="cpxr_lexa",
                tf_names=["cpxR", "lexA"],
                n_elites=3,
            )
        _ = output_log_path
        return SimpleNamespace(
            runbook_path=path,
            workspace_root=path.parent.parent,
            executed_step_ids=list(step_ids or []),
        )

    monkeypatch.setattr(portfolio_workflow, "run_workspace_runbook", _fake_run_workspace_runbook)

    run_dir = run_portfolio(spec_path, prepare_ready_policy="rerun")
    assert run_dir.exists()


def test_run_portfolio_schema_v3_fails_if_run_dir_missing_after_prepare(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    roots = tmp_path / "workspaces"
    source = roots / "pairwise_cpxr_baer"
    source.mkdir(parents=True, exist_ok=True)
    runbook_file = source / "configs" / "runbook.yaml"
    runbook_file.parent.mkdir(parents=True, exist_ok=True)
    runbook_file.write_text(
        yaml.safe_dump(
            {
                "runbook": {
                    "schema_version": 1,
                    "name": "pairwise_cpxr_baer",
                    "steps": [
                        {
                            "id": "sample_run",
                            "run": ["sample", "--force-overwrite", "-c", "configs/config.yaml"],
                        }
                    ],
                }
            }
        )
    )

    portfolio_workspace = roots / "portfolio_pairwise_handoff"
    portfolio_workspace.mkdir(parents=True, exist_ok=True)
    spec_path = portfolio_workspace / "configs" / "pairwise_handoff.portfolio.yaml"
    spec_path.parent.mkdir(parents=True, exist_ok=True)
    spec_path.write_text(
        yaml.safe_dump(
            {
                "portfolio": {
                    "schema_version": 3,
                    "name": "pairwise_handoff",
                    "execution": {"mode": "prepare_then_aggregate"},
                    "sources": [
                        {
                            "id": "pairwise_cpxr_baer",
                            "workspace": "../pairwise_cpxr_baer",
                            "run_dir": "outputs/set1_cpxr_baer",
                            "prepare": {
                                "runbook": "configs/runbook.yaml",
                                "step_ids": ["sample_run"],
                            },
                        }
                    ],
                }
            }
        )
    )

    def _fake_run_workspace_runbook(
        path: Path,
        *,
        step_ids=None,
        dry_run: bool = False,
        output_log_path: Path | None = None,
    ):
        return SimpleNamespace(
            runbook_path=path,
            workspace_root=path.parent.parent,
            executed_step_ids=list(step_ids or []),
        )

    monkeypatch.setattr(portfolio_workflow, "run_workspace_runbook", _fake_run_workspace_runbook)

    with pytest.raises(FileNotFoundError, match="run_dir does not exist after preparation"):
        run_portfolio(spec_path)


def test_run_portfolio_fails_when_export_manifest_is_missing(tmp_path: Path) -> None:
    source_workspace = tmp_path / "workspaces" / "pairwise_cpxr_baer"
    source_workspace.mkdir(parents=True, exist_ok=True)
    run_dir = _seed_source_run(
        workspace=source_workspace,
        run_rel="outputs/set1_cpxr_baer",
        source_tag="cpxr_baer",
        tf_names=["cpxR", "baeR"],
        n_elites=3,
    )
    run_export_sequences_manifest_path(run_dir).unlink()

    portfolio_workspace = tmp_path / "workspaces" / "portfolio_pairwise_handoff"
    portfolio_workspace.mkdir(parents=True, exist_ok=True)
    spec_path = portfolio_workspace / "pairwise_handoff.portfolio.yaml"
    spec_path.write_text(
        yaml.safe_dump(
            {
                "portfolio": {
                    "schema_version": 3,
                    "name": "pairwise_handoff",
                    "execution": {"mode": "aggregate_only"},
                    "sources": [
                        {
                            "id": "pairwise_cpxr_baer",
                            "workspace": "../pairwise_cpxr_baer",
                            "run_dir": "outputs/set1_cpxr_baer",
                        }
                    ],
                }
            }
        )
    )

    with pytest.raises(ValueError, match="aggregate_only preflight failed") as exc_info:
        run_portfolio(spec_path)
    assert "export_manifest.json" in str(exc_info.value)


def test_run_portfolio_accepts_manifest_top_k_above_export_elites_rows(tmp_path: Path) -> None:
    source_workspace = tmp_path / "workspaces" / "pairwise_cpxr_baer"
    source_workspace.mkdir(parents=True, exist_ok=True)
    _seed_source_run(
        workspace=source_workspace,
        run_rel="outputs/set1_cpxr_baer",
        source_tag="cpxr_baer",
        tf_names=["cpxR", "baeR"],
        n_elites=3,
        manifest_top_k=8,
    )

    portfolio_workspace = tmp_path / "workspaces" / "portfolio_pairwise_handoff"
    portfolio_workspace.mkdir(parents=True, exist_ok=True)
    spec_path = portfolio_workspace / "pairwise_handoff.portfolio.yaml"
    spec_path.write_text(
        yaml.safe_dump(
            {
                "portfolio": {
                    "schema_version": 3,
                    "name": "pairwise_handoff",
                    "execution": {"mode": "aggregate_only"},
                    "sources": [
                        {
                            "id": "pairwise_cpxr_baer",
                            "workspace": "../pairwise_cpxr_baer",
                            "run_dir": "outputs/set1_cpxr_baer",
                        }
                    ],
                }
            }
        )
    )

    run_dir = run_portfolio(spec_path)
    handoff_path = portfolio_table_path(run_dir, "handoff_windows_long", "parquet")
    elite_summary_path = portfolio_table_path(run_dir, "handoff_elites_summary", "parquet")
    handoff_df = read_parquet(handoff_path)
    elite_summary_df = read_parquet(elite_summary_path)
    assert len(elite_summary_df) == 3
    assert len(handoff_df) == 6


def test_run_portfolio_fails_when_manifest_top_k_below_export_elites_rows(tmp_path: Path) -> None:
    source_workspace = tmp_path / "workspaces" / "pairwise_cpxr_baer"
    source_workspace.mkdir(parents=True, exist_ok=True)
    _seed_source_run(
        workspace=source_workspace,
        run_rel="outputs/set1_cpxr_baer",
        source_tag="cpxr_baer",
        tf_names=["cpxR", "baeR"],
        n_elites=3,
        manifest_top_k=2,
    )

    portfolio_workspace = tmp_path / "workspaces" / "portfolio_pairwise_handoff"
    portfolio_workspace.mkdir(parents=True, exist_ok=True)
    spec_path = portfolio_workspace / "pairwise_handoff.portfolio.yaml"
    spec_path.write_text(
        yaml.safe_dump(
            {
                "portfolio": {
                    "schema_version": 3,
                    "name": "pairwise_handoff",
                    "execution": {"mode": "aggregate_only"},
                    "sources": [
                        {
                            "id": "pairwise_cpxr_baer",
                            "workspace": "../pairwise_cpxr_baer",
                            "run_dir": "outputs/set1_cpxr_baer",
                        }
                    ],
                }
            }
        )
    )

    with pytest.raises(ValueError, match="Manifest top_k must be >= export elites row count"):
        run_portfolio(spec_path)


def test_run_portfolio_fails_when_source_run_stage_is_not_sample(tmp_path: Path) -> None:
    source_workspace = tmp_path / "workspaces" / "pairwise_cpxr_baer"
    source_workspace.mkdir(parents=True, exist_ok=True)
    _seed_source_run(
        workspace=source_workspace,
        run_rel="outputs/set1_cpxr_baer",
        source_tag="cpxr_baer",
        tf_names=["cpxR", "baeR"],
        n_elites=3,
        stage="analyze",
    )

    portfolio_workspace = tmp_path / "workspaces" / "portfolio_pairwise_handoff"
    portfolio_workspace.mkdir(parents=True, exist_ok=True)
    spec_path = portfolio_workspace / "pairwise_handoff.portfolio.yaml"
    spec_path.write_text(
        yaml.safe_dump(
            {
                "portfolio": {
                    "schema_version": 3,
                    "name": "pairwise_handoff",
                    "execution": {"mode": "aggregate_only"},
                    "sources": [
                        {
                            "id": "pairwise_cpxr_baer",
                            "workspace": "../pairwise_cpxr_baer",
                            "run_dir": "outputs/set1_cpxr_baer",
                        }
                    ],
                }
            }
        )
    )

    with pytest.raises(ValueError, match="aggregate_only preflight failed") as exc_info:
        run_portfolio(spec_path)
    assert "stage must be 'sample'" in str(exc_info.value)


def test_run_portfolio_fails_when_windows_have_duplicate_elite_tf_rows(tmp_path: Path) -> None:
    source_workspace = tmp_path / "workspaces" / "pairwise_cpxr_baer"
    source_workspace.mkdir(parents=True, exist_ok=True)
    run_dir = _seed_source_run(
        workspace=source_workspace,
        run_rel="outputs/set1_cpxr_baer",
        source_tag="cpxr_baer",
        tf_names=["cpxR", "baeR"],
        n_elites=3,
    )

    elites_export_path = run_dir / "export" / "table__elites.csv"
    elites_export_df = pd.read_csv(elites_export_path)
    members = json.loads(str(elites_export_df.loc[0, "window_members_json"]))
    members.append(dict(members[0]))
    elites_export_df.loc[0, "window_members_json"] = json.dumps(members, sort_keys=True)
    elites_export_df.to_csv(elites_export_path, index=False)

    portfolio_workspace = tmp_path / "workspaces" / "portfolio_pairwise_handoff"
    portfolio_workspace.mkdir(parents=True, exist_ok=True)
    spec_path = portfolio_workspace / "pairwise_handoff.portfolio.yaml"
    spec_path.write_text(
        yaml.safe_dump(
            {
                "portfolio": {
                    "schema_version": 3,
                    "name": "pairwise_handoff",
                    "execution": {"mode": "aggregate_only"},
                    "sources": [
                        {
                            "id": "pairwise_cpxr_baer",
                            "workspace": "../pairwise_cpxr_baer",
                            "run_dir": "outputs/set1_cpxr_baer",
                        }
                    ],
                }
            }
        )
    )

    with pytest.raises(ValueError, match="duplicate"):
        run_portfolio(spec_path)


def test_run_portfolio_aggregate_only_reports_all_unready_sources_with_nudges(tmp_path: Path) -> None:
    roots = tmp_path / "workspaces"
    source_a = roots / "pairwise_cpxr_baer"
    source_b = roots / "pairwise_cpxr_lexa"
    source_a.mkdir(parents=True, exist_ok=True)
    source_b.mkdir(parents=True, exist_ok=True)
    run_a = _seed_source_run(
        workspace=source_a,
        run_rel="outputs/set1_cpxr_baer",
        source_tag="cpxr_baer",
        tf_names=["cpxR", "baeR"],
        n_elites=3,
    )
    run_b = _seed_source_run(
        workspace=source_b,
        run_rel="outputs/set1_cpxr_lexa",
        source_tag="cpxr_lexa",
        tf_names=["cpxR", "lexA"],
        n_elites=3,
    )
    run_export_sequences_manifest_path(run_a).unlink()
    summary_path(analysis_root(run_b)).unlink()

    portfolio_workspace = roots / "portfolio_pairwise_handoff"
    portfolio_workspace.mkdir(parents=True, exist_ok=True)
    spec_path = portfolio_workspace / "pairwise_handoff.portfolio.yaml"
    spec_path.write_text(
        yaml.safe_dump(
            {
                "portfolio": {
                    "schema_version": 3,
                    "name": "pairwise_handoff",
                    "execution": {"mode": "aggregate_only"},
                    "sources": [
                        {
                            "id": "pairwise_cpxr_baer",
                            "workspace": "../pairwise_cpxr_baer",
                            "run_dir": "outputs/set1_cpxr_baer",
                        },
                        {
                            "id": "pairwise_cpxr_lexa",
                            "workspace": "../pairwise_cpxr_lexa",
                            "run_dir": "outputs/set1_cpxr_lexa",
                        },
                    ],
                }
            }
        )
    )

    with pytest.raises(ValueError, match="aggregate_only preflight failed") as exc_info:
        run_portfolio(spec_path)
    message = str(exc_info.value)
    assert "pairwise_cpxr_baer" in message
    assert "pairwise_cpxr_lexa" in message
    assert "export/export_manifest.json" in message
    assert "analysis/reports/summary.json" in message
    assert "prepare_then_aggregate" in message


def test_run_portfolio_aggregate_only_prepare_nudge_uses_repeatable_step_flags(tmp_path: Path) -> None:
    roots = tmp_path / "workspaces"
    source_a = roots / "pairwise_cpxr_baer"
    source_a.mkdir(parents=True, exist_ok=True)
    run_a = _seed_source_run(
        workspace=source_a,
        run_rel="outputs/set1_cpxr_baer",
        source_tag="cpxr_baer",
        tf_names=["cpxR", "baeR"],
        n_elites=3,
    )
    run_export_sequences_manifest_path(run_a).unlink()

    runbook_a = source_a / "configs" / "runbook.yaml"
    runbook_a.parent.mkdir(parents=True, exist_ok=True)
    runbook_a.write_text(
        yaml.safe_dump(
            {
                "runbook": {
                    "schema_version": 1,
                    "name": "pairwise_cpxr_baer",
                    "steps": [
                        {"id": "analyze_summary", "run": ["analyze", "--summary", "-c", "configs/config.yaml"]},
                        {
                            "id": "export_sequences_latest",
                            "run": ["export", "sequences", "--latest", "-c", "configs/config.yaml"],
                        },
                    ],
                }
            }
        )
    )

    portfolio_workspace = roots / "portfolio_pairwise_handoff"
    portfolio_workspace.mkdir(parents=True, exist_ok=True)
    spec_path = portfolio_workspace / "pairwise_handoff.portfolio.yaml"
    spec_path.write_text(
        yaml.safe_dump(
            {
                "portfolio": {
                    "schema_version": 3,
                    "name": "pairwise_handoff",
                    "execution": {"mode": "aggregate_only"},
                    "sources": [
                        {
                            "id": "pairwise_cpxr_baer",
                            "workspace": "../pairwise_cpxr_baer",
                            "run_dir": "outputs/set1_cpxr_baer",
                            "prepare": {
                                "runbook": "configs/runbook.yaml",
                                "step_ids": ["analyze_summary", "export_sequences_latest"],
                            },
                        }
                    ],
                }
            }
        )
    )

    with pytest.raises(ValueError, match="aggregate_only preflight failed") as exc_info:
        run_portfolio(spec_path)
    message = str(exc_info.value)
    assert f"cruncher workspaces run --workspace {source_a.resolve()} --runbook configs/runbook.yaml" in message
    assert "--step analyze_summary" in message
    assert "--step export_sequences_latest" in message
    assert "--steps" not in message


def test_run_portfolio_aggregate_only_missing_run_dir_adds_full_runbook_nudge(tmp_path: Path) -> None:
    roots = tmp_path / "workspaces"
    source = roots / "pairwise_cpxr_baer"
    source.mkdir(parents=True, exist_ok=True)
    (source / "outputs" / "set1_cpxr_baer").mkdir(parents=True, exist_ok=True)

    runbook = source / "configs" / "runbook.yaml"
    runbook.parent.mkdir(parents=True, exist_ok=True)
    runbook.write_text(
        yaml.safe_dump(
            {
                "runbook": {
                    "schema_version": 1,
                    "name": "pairwise_cpxr_baer",
                    "steps": [
                        {"id": "analyze_summary", "run": ["analyze", "--summary", "-c", "configs/config.yaml"]},
                        {
                            "id": "export_sequences_latest",
                            "run": ["export", "sequences", "--latest", "-c", "configs/config.yaml"],
                        },
                    ],
                }
            }
        )
    )

    portfolio_workspace = roots / "portfolio_pairwise_handoff"
    portfolio_workspace.mkdir(parents=True, exist_ok=True)
    spec_path = portfolio_workspace / "pairwise_handoff.portfolio.yaml"
    spec_path.write_text(
        yaml.safe_dump(
            {
                "portfolio": {
                    "schema_version": 3,
                    "name": "pairwise_handoff",
                    "execution": {"mode": "aggregate_only"},
                    "sources": [
                        {
                            "id": "pairwise_cpxr_baer",
                            "workspace": "../pairwise_cpxr_baer",
                            "run_dir": "outputs/set1_cpxr_baer",
                            "prepare": {
                                "runbook": "configs/runbook.yaml",
                                "step_ids": ["analyze_summary", "export_sequences_latest"],
                            },
                        }
                    ],
                }
            }
        )
    )

    with pytest.raises(ValueError, match="aggregate_only preflight failed") as exc_info:
        run_portfolio(spec_path)
    message = str(exc_info.value)
    assert "missing metadata manifest:" in message
    assert "nudge: missing source run artifacts require a full runbook execution:" in message
    assert f"cruncher workspaces run --workspace {source.resolve()} --runbook configs/runbook.yaml" in message


def test_run_portfolio_aggregate_only_missing_analyze_prereqs_adds_full_runbook_nudge(tmp_path: Path) -> None:
    roots = tmp_path / "workspaces"
    source = roots / "pairwise_cpxr_baer"
    source.mkdir(parents=True, exist_ok=True)
    run_dir = _seed_source_run(
        workspace=source,
        run_rel="outputs/set1_cpxr_baer",
        source_tag="cpxr_baer",
        tf_names=["cpxR", "baeR"],
        n_elites=3,
    )
    summary_path(analysis_root(run_dir)).unlink()
    run_export_sequences_manifest_path(run_dir).unlink()

    runbook = source / "configs" / "runbook.yaml"
    runbook.parent.mkdir(parents=True, exist_ok=True)
    runbook.write_text(
        yaml.safe_dump(
            {
                "runbook": {
                    "schema_version": 1,
                    "name": "pairwise_cpxr_baer",
                    "steps": [
                        {"id": "analyze_summary", "run": ["analyze", "--summary", "-c", "configs/config.yaml"]},
                        {
                            "id": "export_sequences_latest",
                            "run": ["export", "sequences", "--latest", "-c", "configs/config.yaml"],
                        },
                    ],
                }
            }
        )
    )

    portfolio_workspace = roots / "portfolio_pairwise_handoff"
    portfolio_workspace.mkdir(parents=True, exist_ok=True)
    spec_path = portfolio_workspace / "pairwise_handoff.portfolio.yaml"
    spec_path.write_text(
        yaml.safe_dump(
            {
                "portfolio": {
                    "schema_version": 3,
                    "name": "pairwise_handoff",
                    "execution": {"mode": "aggregate_only"},
                    "sources": [
                        {
                            "id": "pairwise_cpxr_baer",
                            "workspace": "../pairwise_cpxr_baer",
                            "run_dir": "outputs/set1_cpxr_baer",
                            "prepare": {
                                "runbook": "configs/runbook.yaml",
                                "step_ids": ["analyze_summary", "export_sequences_latest"],
                            },
                        }
                    ],
                }
            }
        )
    )

    with pytest.raises(ValueError, match="aggregate_only preflight failed") as exc_info:
        run_portfolio(spec_path)
    message = str(exc_info.value)
    assert "missing analysis summary:" in message
    assert "missing sample artifact required to recompute analysis summary:" in message
    assert "nudge: missing source run artifacts require a full runbook execution:" in message
    assert f"cruncher workspaces run --workspace {source.resolve()} --runbook configs/runbook.yaml" in message


def test_run_portfolio_prepare_skip_ready_only_runs_missing_sources(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    roots = tmp_path / "workspaces"
    source_a = roots / "pairwise_cpxr_baer"
    source_b = roots / "pairwise_cpxr_lexa"
    source_a.mkdir(parents=True, exist_ok=True)
    source_b.mkdir(parents=True, exist_ok=True)
    _seed_source_run(
        workspace=source_a,
        run_rel="outputs/set1_cpxr_baer",
        source_tag="cpxr_baer",
        tf_names=["cpxR", "baeR"],
        n_elites=3,
    )

    runbook_a = source_a / "configs" / "runbook.yaml"
    runbook_b = source_b / "configs" / "runbook.yaml"
    runbook_a.parent.mkdir(parents=True, exist_ok=True)
    runbook_b.parent.mkdir(parents=True, exist_ok=True)
    runbook_a.write_text(
        yaml.safe_dump(
            {
                "runbook": {
                    "schema_version": 1,
                    "name": "pairwise_cpxr_baer",
                    "steps": [{"id": "sample_run", "run": ["sample", "-c", "configs/config.yaml"]}],
                }
            }
        )
    )
    runbook_b.write_text(
        yaml.safe_dump(
            {
                "runbook": {
                    "schema_version": 1,
                    "name": "pairwise_cpxr_lexa",
                    "steps": [{"id": "sample_run", "run": ["sample", "-c", "configs/config.yaml"]}],
                }
            }
        )
    )

    portfolio_workspace = roots / "portfolio_pairwise_handoff"
    portfolio_workspace.mkdir(parents=True, exist_ok=True)
    spec_path = portfolio_workspace / "pairwise_handoff.portfolio.yaml"
    spec_path.write_text(
        yaml.safe_dump(
            {
                "portfolio": {
                    "schema_version": 3,
                    "name": "pairwise_handoff",
                    "execution": {"mode": "prepare_then_aggregate"},
                    "sources": [
                        {
                            "id": "pairwise_cpxr_baer",
                            "workspace": "../pairwise_cpxr_baer",
                            "run_dir": "outputs/set1_cpxr_baer",
                            "prepare": {
                                "runbook": "configs/runbook.yaml",
                                "step_ids": ["sample_run"],
                            },
                        },
                        {
                            "id": "pairwise_cpxr_lexa",
                            "workspace": "../pairwise_cpxr_lexa",
                            "run_dir": "outputs/set1_cpxr_lexa",
                            "prepare": {
                                "runbook": "configs/runbook.yaml",
                                "step_ids": ["sample_run"],
                            },
                        },
                    ],
                }
            }
        )
    )

    runbook_calls: list[Path] = []
    events: list[str] = []

    def _fake_run_workspace_runbook(
        path: Path,
        *,
        step_ids=None,
        dry_run: bool = False,
        output_log_path: Path | None = None,
    ):
        runbook_calls.append(path.resolve())
        if path.resolve() == runbook_b.resolve():
            _seed_source_run(
                workspace=source_b,
                run_rel="outputs/set1_cpxr_lexa",
                source_tag="cpxr_lexa",
                tf_names=["cpxR", "lexA"],
                n_elites=3,
            )
        return SimpleNamespace(
            runbook_path=path,
            workspace_root=path.parent.parent,
            executed_step_ids=list(step_ids or []),
        )

    def _on_event(name: str, payload: dict[str, object]) -> None:
        _ = payload
        events.append(name)

    monkeypatch.setattr(portfolio_workflow, "run_workspace_runbook", _fake_run_workspace_runbook)

    run_portfolio(spec_path, prepare_ready_policy="skip", on_event=_on_event)
    assert runbook_calls == [runbook_b.resolve()]
    assert "prepare_source_skipped" in events


def test_run_portfolio_prepare_failure_reports_actionable_nudge(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    roots = tmp_path / "workspaces"
    source = roots / "pairwise_cpxr_baer"
    source.mkdir(parents=True, exist_ok=True)
    runbook_file = source / "configs" / "runbook.yaml"
    runbook_file.parent.mkdir(parents=True, exist_ok=True)
    runbook_file.write_text(
        yaml.safe_dump(
            {
                "runbook": {
                    "schema_version": 1,
                    "name": "pairwise_cpxr_baer",
                    "steps": [
                        {"id": "analyze_summary", "run": ["analyze", "--summary", "-c", "configs/config.yaml"]},
                        {
                            "id": "export_sequences_latest",
                            "run": ["export", "sequences", "--latest", "-c", "configs/config.yaml"],
                        },
                    ],
                }
            }
        )
    )

    portfolio_workspace = roots / "portfolio_pairwise_handoff"
    portfolio_workspace.mkdir(parents=True, exist_ok=True)
    spec_path = portfolio_workspace / "pairwise_handoff.portfolio.yaml"
    spec_path.write_text(
        yaml.safe_dump(
            {
                "portfolio": {
                    "schema_version": 3,
                    "name": "pairwise_handoff",
                    "execution": {"mode": "prepare_then_aggregate"},
                    "sources": [
                        {
                            "id": "pairwise_cpxr_baer",
                            "workspace": "../pairwise_cpxr_baer",
                            "run_dir": "outputs/set1_cpxr_baer",
                            "prepare": {
                                "runbook": "configs/runbook.yaml",
                                "step_ids": ["analyze_summary", "export_sequences_latest"],
                            },
                        }
                    ],
                }
            }
        )
    )

    def _fake_run_workspace_runbook(
        path: Path,
        *,
        step_ids=None,
        dry_run: bool = False,
        output_log_path: Path | None = None,
    ):
        _ = (path, step_ids, dry_run)
        raise RuntimeError("Runbook step failed: step='analyze_summary' returncode=1")

    monkeypatch.setattr(portfolio_workflow, "run_workspace_runbook", _fake_run_workspace_runbook)

    with pytest.raises(ValueError, match="Portfolio source preparation failed") as exc_info:
        run_portfolio(spec_path)
    message = str(exc_info.value)
    assert "pairwise_cpxr_baer" in message
    assert "sample_run" in message
    assert "analyze_summary" in message
    assert "export_sequences_latest" in message
    assert "nudge: full runbook required:" in message
    assert f"cruncher workspaces run --workspace {source.resolve()} --runbook configs/runbook.yaml" in message


def test_run_portfolio_ensures_missing_studies_and_writes_sequence_length_table(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    roots = tmp_path / "workspaces"
    source = roots / "pairwise_cpxr_baer"
    source.mkdir(parents=True, exist_ok=True)
    _seed_source_run(
        workspace=source,
        run_rel="outputs/set1_cpxr_baer",
        source_tag="cpxr_baer",
        tf_names=["cpxR", "baeR"],
        n_elites=4,
    )

    length_spec = _seed_study_spec(
        workspace=source,
        study_name="length_vs_score",
        trials=[{"id": "BASE", "factors": {}}],
        replays_enabled=False,
    )
    diversity_spec = _seed_study_spec(
        workspace=source,
        study_name="diversity_vs_score",
        trials=[{"id": "BASE", "factors": {}}],
        replays_enabled=True,
    )
    _seed_study_outputs(
        study_spec=diversity_spec,
        agg_rows=[
            {
                "trial_id": "BASE",
                "median_score_mean": 0.75,
                "best_score_mean": 0.84,
                "median_nn_full_bp_mean": 6.0,
            }
        ],
        length_rows=[
            {
                "trial_id": "BASE",
                "sequence_length": 11,
                "median_score_mean": 0.75,
                "best_score_mean": 0.84,
                "median_nn_full_bp_mean": 6.0,
            }
        ],
    )

    runbook_file = source / "configs" / "runbook.yaml"
    runbook_file.parent.mkdir(parents=True, exist_ok=True)
    runbook_file.write_text(
        yaml.safe_dump(
            {
                "runbook": {
                    "schema_version": 1,
                    "name": "pairwise_cpxr_baer",
                    "steps": [
                        {"id": "analyze_summary", "run": ["analyze", "--summary", "-c", "configs/config.yaml"]},
                        {
                            "id": "export_sequences_latest",
                            "run": ["export", "sequences", "--latest", "-c", "configs/config.yaml"],
                        },
                    ],
                }
            }
        )
    )

    portfolio_workspace = roots / "portfolio_pairwise_handoff"
    portfolio_workspace.mkdir(parents=True, exist_ok=True)
    spec_path = portfolio_workspace / "configs" / "master.portfolio.yaml"
    spec_path.parent.mkdir(parents=True, exist_ok=True)
    spec_path.write_text(
        yaml.safe_dump(
            {
                "portfolio": {
                    "schema_version": 3,
                    "name": "master",
                    "execution": {"mode": "prepare_then_aggregate"},
                    "artifacts": {"table_format": "parquet", "write_csv": True},
                    "studies": {
                        "ensure_specs": [
                            "configs/studies/length_vs_score.study.yaml",
                            "configs/studies/diversity_vs_score.study.yaml",
                        ],
                        "sequence_length_table": {
                            "enabled": True,
                            "study_spec": "configs/studies/length_vs_score.study.yaml",
                            "top_n_lengths": 2,
                        },
                    },
                    "sources": [
                        {
                            "id": "pairwise_cpxr_baer",
                            "workspace": "../pairwise_cpxr_baer",
                            "run_dir": "outputs/set1_cpxr_baer",
                            "study_spec": "configs/studies/diversity_vs_score.study.yaml",
                            "prepare": {
                                "runbook": "configs/runbook.yaml",
                                "step_ids": ["analyze_summary", "export_sequences_latest"],
                            },
                        }
                    ],
                }
            }
        )
    )

    runbook_calls: list[tuple[Path, list[str] | None]] = []
    study_calls: list[Path] = []

    def _fake_run_workspace_runbook(
        path: Path,
        *,
        step_ids=None,
        dry_run: bool = False,
        output_log_path: Path | None = None,
    ):
        runbook_calls.append((path, list(step_ids or [])))
        return SimpleNamespace(
            runbook_path=path,
            workspace_root=path.parent.parent,
            executed_step_ids=list(step_ids or []),
        )

    def _fake_run_study(
        path: Path,
        *,
        resume: bool = False,
        force_overwrite: bool = False,
        progress_bar: bool = True,
        quiet_logs: bool = False,
    ):
        study_calls.append(path.resolve())
        assert resume is False
        assert force_overwrite is False
        assert progress_bar is False
        assert quiet_logs is True
        _seed_study_outputs(
            study_spec=path,
            agg_rows=[
                {
                    "trial_id": "L11",
                    "median_score_mean": 0.71,
                    "best_score_mean": 0.80,
                    "median_nn_full_bp_mean": 6.4,
                },
                {
                    "trial_id": "L12",
                    "median_score_mean": 0.72,
                    "best_score_mean": 0.81,
                    "median_nn_full_bp_mean": 6.1,
                },
            ],
            length_rows=[
                {
                    "trial_id": "L11",
                    "sequence_length": 11,
                    "median_score_mean": 0.71,
                    "best_score_mean": 0.80,
                    "median_nn_full_bp_mean": 6.4,
                },
                {
                    "trial_id": "L12",
                    "sequence_length": 12,
                    "median_score_mean": 0.72,
                    "best_score_mean": 0.81,
                    "median_nn_full_bp_mean": 6.1,
                },
            ],
        )
        return resolve_deterministic_study_run_dir(path)

    monkeypatch.setattr(portfolio_workflow, "run_workspace_runbook", _fake_run_workspace_runbook)
    monkeypatch.setattr(portfolio_workflow, "run_study", _fake_run_study)

    run_dir = run_portfolio(spec_path)
    assert runbook_calls == [(runbook_file.resolve(), ["analyze_summary", "export_sequences_latest"])]
    assert study_calls == [length_spec.resolve()]

    length_table = portfolio_table_path(run_dir, "handoff_sequence_length", "parquet")
    assert length_table.exists()
    length_df = read_parquet(length_table)
    assert length_df["source_id"].astype(str).tolist() == ["pairwise_cpxr_baer", "pairwise_cpxr_baer"]
    assert length_df["sequence_length"].astype(int).tolist() == [11, 12]


def test_run_portfolio_materializes_outputs_before_later_source_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    roots = tmp_path / "workspaces"
    source_a = roots / "pairwise_cpxr_baer"
    source_b = roots / "pairwise_cpxr_lexa"
    source_a.mkdir(parents=True, exist_ok=True)
    source_b.mkdir(parents=True, exist_ok=True)

    _seed_source_run(
        workspace=source_a,
        run_rel="outputs/set1_cpxr_baer",
        source_tag="cpxr_baer",
        tf_names=["cpxR", "baeR"],
        n_elites=4,
    )
    _seed_source_run(
        workspace=source_b,
        run_rel="outputs/set1_cpxr_lexa",
        source_tag="cpxr_lexa",
        tf_names=["cpxR", "lexA"],
        n_elites=4,
    )

    portfolio_workspace = roots / "portfolio_pairwise_handoff"
    portfolio_workspace.mkdir(parents=True, exist_ok=True)
    spec_path = portfolio_workspace / "pairwise_handoff.portfolio.yaml"
    spec_path.write_text(
        yaml.safe_dump(
            {
                "portfolio": {
                    "schema_version": 3,
                    "name": "pairwise_handoff",
                    "execution": {"mode": "aggregate_only"},
                    "artifacts": {"table_format": "parquet", "write_csv": False},
                    "sources": [
                        {
                            "id": "pairwise_cpxr_baer",
                            "workspace": "../pairwise_cpxr_baer",
                            "run_dir": "outputs/set1_cpxr_baer",
                        },
                        {
                            "id": "pairwise_cpxr_lexa",
                            "workspace": "../pairwise_cpxr_lexa",
                            "run_dir": "outputs/set1_cpxr_lexa",
                        },
                    ],
                }
            }
        )
    )

    load_source_rows = portfolio_workflow._load_source_rows

    def _fail_second_source(source):
        if str(source.id) == "pairwise_cpxr_lexa":
            raise RuntimeError("synthetic second source failure")
        return load_source_rows(source)

    monkeypatch.setattr(portfolio_workflow, "_load_source_rows", _fail_second_source)

    with pytest.raises(RuntimeError, match="synthetic second source failure"):
        run_portfolio(spec_path)

    run_root = portfolio_workspace / "outputs" / "portfolios" / "pairwise_handoff"
    run_dirs = sorted(item for item in run_root.iterdir() if item.is_dir())
    assert len(run_dirs) == 1
    run_dir = run_dirs[0]

    source_summary_table = portfolio_table_path(run_dir, "source_summary", "parquet")
    assert source_summary_table.exists()
    summary_df = read_parquet(source_summary_table)
    assert summary_df["source_id"].astype(str).tolist() == ["pairwise_cpxr_baer"]

    tradeoff_plot = portfolio_workflow.portfolio_plot_path(run_dir, "source_tradeoff_score_vs_diversity", "pdf")
    assert tradeoff_plot.exists()


def test_write_tradeoff_plot_hides_top_and_right_spines(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source_summary_df = pd.DataFrame(
        [
            {
                "source_id": "source_a",
                "median_min_best_score_norm": 0.75,
                "mean_pairwise_hamming_bp": 12.0,
            },
            {
                "source_id": "source_b",
                "median_min_best_score_norm": 0.68,
                "mean_pairwise_hamming_bp": 15.0,
            },
        ]
    )
    out_path = tmp_path / "source_tradeoff_score_vs_diversity.pdf"
    captured: dict[str, object] = {}
    real_plt = plt

    class _FakePyplot:
        @staticmethod
        def subplots(*args, **kwargs):
            fig, ax = real_plt.subplots(*args, **kwargs)
            captured["fig"] = fig
            captured["ax"] = ax
            return fig, ax

        @staticmethod
        def close(fig):
            _ = fig

    monkeypatch.setattr(portfolio_materialization, "_pyplot", lambda: _FakePyplot)
    written = portfolio_workflow._write_tradeoff_plot(source_summary_df, out_path)

    assert written == out_path
    assert out_path.exists()
    ax = captured["ax"]
    assert ax.spines["top"].get_visible() is False
    assert ax.spines["right"].get_visible() is False
    plt.close(captured["fig"])


def test_run_portfolio_prepare_mode_materializes_before_later_prepare_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    roots = tmp_path / "workspaces"
    source_a = roots / "pairwise_cpxr_baer"
    source_b = roots / "pairwise_cpxr_lexa"
    source_a.mkdir(parents=True, exist_ok=True)
    source_b.mkdir(parents=True, exist_ok=True)

    runbook_a = source_a / "configs" / "runbook.yaml"
    runbook_b = source_b / "configs" / "runbook.yaml"
    runbook_a.parent.mkdir(parents=True, exist_ok=True)
    runbook_b.parent.mkdir(parents=True, exist_ok=True)
    runbook_payload = {
        "runbook": {
            "schema_version": 1,
            "name": "prepare_source",
            "steps": [{"id": "sample_run", "run": ["sample", "--force-overwrite", "-c", "configs/config.yaml"]}],
        }
    }
    runbook_a.write_text(yaml.safe_dump(runbook_payload))
    runbook_b.write_text(yaml.safe_dump(runbook_payload))

    portfolio_workspace = roots / "portfolio_pairwise_handoff"
    portfolio_workspace.mkdir(parents=True, exist_ok=True)
    spec_path = portfolio_workspace / "pairwise_handoff.portfolio.yaml"
    spec_path.write_text(
        yaml.safe_dump(
            {
                "portfolio": {
                    "schema_version": 3,
                    "name": "pairwise_handoff",
                    "execution": {"mode": "prepare_then_aggregate"},
                    "artifacts": {"table_format": "parquet", "write_csv": False},
                    "sources": [
                        {
                            "id": "pairwise_cpxr_baer",
                            "workspace": "../pairwise_cpxr_baer",
                            "run_dir": "outputs/set1_cpxr_baer",
                            "prepare": {"runbook": "configs/runbook.yaml", "step_ids": ["sample_run"]},
                        },
                        {
                            "id": "pairwise_cpxr_lexa",
                            "workspace": "../pairwise_cpxr_lexa",
                            "run_dir": "outputs/set1_cpxr_lexa",
                            "prepare": {"runbook": "configs/runbook.yaml", "step_ids": ["sample_run"]},
                        },
                    ],
                }
            }
        )
    )

    def _fake_run_workspace_runbook(
        path: Path,
        *,
        step_ids=None,
        dry_run: bool = False,
        output_log_path: Path | None = None,
    ):
        assert dry_run is False
        if path.resolve() == runbook_a.resolve():
            _seed_source_run(
                workspace=source_a,
                run_rel="outputs/set1_cpxr_baer",
                source_tag="cpxr_baer",
                tf_names=["cpxR", "baeR"],
                n_elites=4,
            )
            return SimpleNamespace(
                runbook_path=path,
                workspace_root=path.parent.parent,
                executed_step_ids=list(step_ids or []),
            )
        raise RuntimeError("synthetic prepare failure on second source")

    monkeypatch.setattr(portfolio_workflow, "run_workspace_runbook", _fake_run_workspace_runbook)

    with pytest.raises(ValueError, match="synthetic prepare failure on second source"):
        run_portfolio(spec_path, prepare_ready_policy="rerun")

    run_root = portfolio_workspace / "outputs" / "portfolios" / "pairwise_handoff"
    run_dirs = sorted(item for item in run_root.iterdir() if item.is_dir())
    assert len(run_dirs) == 1
    run_dir = run_dirs[0]

    source_summary_table = portfolio_table_path(run_dir, "source_summary", "parquet")
    assert source_summary_table.exists()
    summary_df = read_parquet(source_summary_table)
    assert summary_df["source_id"].astype(str).tolist() == ["pairwise_cpxr_baer"]

    tradeoff_plot = portfolio_workflow.portfolio_plot_path(run_dir, "source_tradeoff_score_vs_diversity", "pdf")
    assert tradeoff_plot.exists()


def test_run_portfolio_prepare_mode_runs_prepare_before_ensured_studies(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    roots = tmp_path / "workspaces"
    source = roots / "pairwise_cpxr_baer"
    source.mkdir(parents=True, exist_ok=True)

    runbook_file = source / "configs" / "runbook.yaml"
    runbook_file.parent.mkdir(parents=True, exist_ok=True)
    runbook_file.write_text(
        yaml.safe_dump(
            {
                "runbook": {
                    "schema_version": 1,
                    "name": "pairwise_cpxr_baer",
                    "steps": [
                        {"id": "sample_run", "run": ["sample", "--force-overwrite", "-c", "configs/config.yaml"]},
                    ],
                }
            }
        )
    )

    length_spec = _seed_study_spec(
        workspace=source,
        study_name="length_vs_score",
        trials=[{"id": "BASE", "factors": {}}],
        replays_enabled=False,
    )

    portfolio_workspace = roots / "portfolio_pairwise_handoff"
    portfolio_workspace.mkdir(parents=True, exist_ok=True)
    spec_path = portfolio_workspace / "pairwise_handoff.portfolio.yaml"
    spec_path.write_text(
        yaml.safe_dump(
            {
                "portfolio": {
                    "schema_version": 3,
                    "name": "pairwise_handoff",
                    "execution": {"mode": "prepare_then_aggregate"},
                    "artifacts": {"table_format": "parquet", "write_csv": False},
                    "studies": {
                        "ensure_specs": [
                            "configs/studies/length_vs_score.study.yaml",
                        ]
                    },
                    "sources": [
                        {
                            "id": "pairwise_cpxr_baer",
                            "workspace": "../pairwise_cpxr_baer",
                            "run_dir": "outputs/set1_cpxr_baer",
                            "prepare": {"runbook": "configs/runbook.yaml", "step_ids": ["sample_run"]},
                        },
                    ],
                }
            }
        )
    )

    call_order: list[str] = []

    def _fake_run_workspace_runbook(
        path: Path,
        *,
        step_ids=None,
        dry_run: bool = False,
        output_log_path: Path | None = None,
    ):
        assert dry_run is False
        call_order.append("prepare")
        lock_file = source / ".cruncher" / "locks" / "config.lock.json"
        lock_file.parent.mkdir(parents=True, exist_ok=True)
        lock_file.write_text("{}\n")
        _seed_source_run(
            workspace=source,
            run_rel="outputs/set1_cpxr_baer",
            source_tag="cpxr_baer",
            tf_names=["cpxR", "baeR"],
            n_elites=4,
        )
        return SimpleNamespace(
            runbook_path=path,
            workspace_root=path.parent.parent,
            executed_step_ids=list(step_ids or []),
        )

    def _fake_run_study(
        path: Path,
        *,
        resume: bool = False,
        force_overwrite: bool = False,
        progress_bar: bool = True,
        quiet_logs: bool = False,
    ):
        assert path.resolve() == length_spec.resolve()
        call_order.append("study")
        lock_file = source / ".cruncher" / "locks" / "config.lock.json"
        if not lock_file.exists():
            raise FileNotFoundError(f"Missing lockfile for study run: {lock_file}")
        return resolve_deterministic_study_run_dir(path)

    monkeypatch.setattr(portfolio_workflow, "run_workspace_runbook", _fake_run_workspace_runbook)
    monkeypatch.setattr(portfolio_workflow, "run_study", _fake_run_study)

    run_dir = run_portfolio(spec_path, prepare_ready_policy="rerun")
    assert run_dir.exists()
    assert call_order == ["prepare", "study"]
