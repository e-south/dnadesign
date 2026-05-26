from __future__ import annotations

from .helpers import Path, json


def test_suite_review_accepts_complete_three_seed_roots(tmp_path: Path) -> None:
    from dnadesign.studies.units.stress_ethanol_cipro_growth.opal_densegen_axis_probe.suite_review import (
        build_probe_suite_review,
    )

    roots = [_write_complete_root(tmp_path, seed) for seed in (7, 17, 29)]

    payload = build_probe_suite_review(roots, out_dir=tmp_path / "suite")

    assert payload["status"] == "ok"
    assert payload["problems"] == []
    assert payload["trajectory_summary"]["pair_count"] == 36
    assert payload["null_attention"]["count"] == 3
    assert (tmp_path / "suite" / "suite_review.json").exists()
    assert (tmp_path / "suite" / "suite_review.md").exists()


def test_suite_review_rejects_missing_seed_and_partial_root(tmp_path: Path) -> None:
    from dnadesign.studies.units.stress_ethanol_cipro_growth.opal_densegen_axis_probe.suite_review import (
        build_probe_suite_review,
    )

    root7 = _write_complete_root(tmp_path, 7)
    root17 = _write_complete_root(tmp_path, 17, run_count=11, round_count=132)

    payload = build_probe_suite_review([root7, root17])

    assert payload["status"] == "attention"
    assert "expected_seed_missing:29" in payload["problems"]
    assert any(problem.endswith(":metrics_run_count:11") for problem in payload["problems"])
    assert any(problem.endswith(":round_metric_count:132") for problem in payload["problems"])


def _write_complete_root(tmp_path: Path, seed: int, *, run_count: int = 24, round_count: int = 288) -> Path:
    root = tmp_path / f"densegen_motif_qa_k12_s3_v1_seed{seed}_all_r12"
    reports = root / "reports"
    reports.mkdir(parents=True)
    run_keys = [f"run_{idx}" for idx in range(run_count)]
    runs = [{"run_key": run_key, "seed": seed, "as_of_round": 11} for run_key in run_keys]
    rounds = [
        {"run_key": run_keys[idx % max(1, len(run_keys))], "seed": seed, "as_of_round": idx % 12}
        for idx in range(round_count)
    ]
    metrics = {"decision": "PASS_FULL_MATRIX_GATE", "runs": runs, "rounds": rounds}
    status = {"status": "ok", "decision": "PASS_FULL_MATRIX_GATE", "problems": []}
    trajectory_pairs = [
        {
            "campaign": campaign,
            "label_family_id": label_family_id,
            "split_id": split_id,
            "seed": seed,
            "paired_auc_delta": 2.0,
            "final_positive_minus_null_lift": 1.0,
        }
        for label_family_id in ("densegen_plan_logic4", "tf_family_count")
        for campaign in ("cipro", "ethanol", "dual")
        for split_id in ("random_id", "leave_sigma35_variant")
    ]
    review = {
        "decision": "PASS_FULL_MATRIX_GATE",
        "problems": [],
        "gate_coverage": {
            "campaigns": ["cipro", "dual", "ethanol"],
            "splits": ["leave_sigma35_variant", "random_id"],
            "positive_null_pairs_complete": True,
            "omitted_scored_gates": [],
        },
        "plot_quality": {"status": "ok", "plot_count": 948, "problem_count": 0, "problems": []},
        "opal_campaign_reviews": [{"run_key": run_key, "warnings": [], "stale_artifacts": []} for run_key in run_keys],
        "trajectory_qa": {"pairs": trajectory_pairs},
        "round_dynamics": [
            {
                "run_key": "null_run",
                "campaign": "cipro",
                "split_id": "random_id",
                "null_transient_spike": True,
                "null_final_threshold_exceeded": False,
                "max_round": 3,
                "max_lift": 1.5,
                "final_lift": 0.5,
            }
        ],
    }
    (reports / "metrics.json").write_text(json.dumps(metrics), encoding="utf-8")
    (reports / "status.json").write_text(json.dumps(status), encoding="utf-8")
    (reports / "review_manifest.json").write_text(json.dumps(review), encoding="utf-8")
    (root / "probe_plan.json").write_text(json.dumps({"plan": {"seed": seed}}), encoding="utf-8")
    return root
