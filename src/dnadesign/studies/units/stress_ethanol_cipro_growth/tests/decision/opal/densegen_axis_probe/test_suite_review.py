from __future__ import annotations

from .helpers import Path, json, pytest
from .probe_modules import probe_module


def test_numeric_summary_uses_student_t_ci_for_small_seed_replicates() -> None:
    suite_replicates = probe_module("reporting.suite_replicates")

    summary = suite_replicates.numeric_mean_ci_summary([1.0, 2.0, 3.0])

    assert summary["mean"] == 2.0
    assert summary["sem"] == pytest.approx(1.0 / (3.0**0.5))
    assert summary["ci95_low"] == pytest.approx(-0.484137711719546)
    assert summary["ci95_high"] == pytest.approx(4.484137711719546)


def test_suite_review_accepts_complete_three_seed_roots(tmp_path: Path) -> None:
    build_probe_suite_review = probe_module("reporting.suite_review").build_probe_suite_review

    roots = [_write_complete_root(tmp_path, seed) for seed in (7, 17, 29)]

    payload = build_probe_suite_review(roots, out_dir=tmp_path / "suite")

    assert payload["status"] == "ok"
    assert payload["problems"] == []
    assert payload["trajectory_summary"]["pair_count"] == 36
    assert payload["trajectory_summary"]["paired_auc_delta"]["ci95_low"] == 2.0
    assert payload["replicate_summary"]["replicate_unit"] == "seed"
    assert payload["replicate_summary"]["interval_kind"] == "student_t_mean_ci"
    assert payload["replicate_summary"]["group_count"] == 12
    first_group = payload["replicate_summary"]["groups"][0]
    assert first_group["seed_count"] == 3
    assert first_group["paired_auc_delta"]["mean"] == 2.0
    assert first_group["paired_auc_delta"]["ci95_low"] == 2.0
    assert first_group["paired_auc_delta"]["ci95_high"] == 2.0
    assert payload["null_attention"]["count"] == 3
    assert (tmp_path / "suite" / "suite_review.json").exists()
    assert (tmp_path / "suite" / "suite_review.md").exists()
    assert (tmp_path / "suite" / "replicate_seed_mean_ci.csv").exists()
    assert (tmp_path / "suite" / "paired_auc_delta_mean_ci.png").exists()
    assert (tmp_path / "suite" / "final_positive_minus_null_lift_mean_ci.png").exists()


def test_suite_review_rejects_missing_seed_and_partial_root(tmp_path: Path) -> None:
    build_probe_suite_review = probe_module("reporting.suite_review").build_probe_suite_review

    root7 = _write_complete_root(tmp_path, 7)
    root17 = _write_complete_root(tmp_path, 17, run_count=11, round_count=132)

    payload = build_probe_suite_review([root7, root17])

    assert payload["status"] == "attention"
    assert "expected_seed_missing:29" in payload["problems"]
    assert any(problem.endswith(":metrics_run_count:11") for problem in payload["problems"])
    assert any(problem.endswith(":round_metric_count:132") for problem in payload["problems"])


def test_suite_opal_notebook_writes_combined_seed_scope(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    mod = probe_module("reporting.suite_notebook")

    roots = []
    for seed in (7, 17, 29):
        root = tmp_path / f"seed{seed}"
        cfg = root / "scratch_campaigns" / f"campaign_s{seed}" / "configs" / "campaign.yaml"
        cfg.parent.mkdir(parents=True)
        cfg.write_text("campaign: {}\n", encoding="utf-8")
        roots.append(root)

    monkeypatch.setattr(
        mod,
        "build_campaign_set_notebook_view_model",
        lambda paths, **kwargs: {
            "campaigns": [{"campaign": {"slug": str(path)}} for path in paths],
            "collection": {"collection_id": "fixture", "comparison_views": []},
        },
    )
    monkeypatch.setattr(
        mod,
        "materialize_campaign_set_collection_visuals",
        lambda campaigns, *, collection, output_dir: {
            "output_dir": str(output_dir),
            "visual_count": 1,
            "comparison_set_count": 1,
        },
    )
    monkeypatch.setattr(mod, "render_campaign_set_notebook", lambda paths, **kwargs: "import marimo\n")
    monkeypatch.setattr(mod, "smoke_check_notebook", lambda path, *, run_marimo_check: None)

    payload = mod.build_probe_suite_opal_notebook(roots, out_dir=tmp_path / "suite_notebook")

    assert payload["campaign_count"] == 3
    assert payload["collection_visual_count"] == 1
    collection = json.loads(Path(payload["collection_manifest"]).read_text(encoding="utf-8"))
    assert collection["collection_id"] == "densegen_motif_qa_k12_s3_v1_all_seed_replicates"
    assert collection["relationships"][0]["replicate_on"] == ["seed"]
    assert Path(payload["notebook"]).exists()


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
