from __future__ import annotations

from importlib import import_module

from .helpers import Path, pd

_CONFIGURED_PLOTS_MODULE = (
    "dnadesign.studies.units.stress_ethanol_cipro_growth.decision.opal."
    "densegen_axis_probe.reporting.review.configured_plots"
)


def _quality_for_configured_plot_entry(entry: dict[str, object]) -> dict[str, object]:
    module = import_module(_CONFIGURED_PLOTS_MODULE)
    return module._quality_for_configured_plot_entry(entry)


def test_configured_plot_quality_respects_artifact_round_scope_and_optional_tidy_csv(tmp_path: Path) -> None:
    def _png(path: Path) -> str:
        from PIL import Image, ImageDraw

        image = Image.new("RGB", (320, 240), "white")
        draw = ImageDraw.Draw(image)
        draw.line((20, 220, 300, 20), fill="black", width=3)
        image.save(path)
        return str(path)

    all_rounds_csv = tmp_path / "all_rounds.csv"
    pd.DataFrame(
        {
            "round": [0, 1, 2],
            "cohort": ["selected", "selected", "selected"],
            "metric": ["pred__score_selected"] * 3,
            "summary": ["mean", "mean", "mean"],
            "value": [0.1, 0.2, 0.3],
        }
    ).to_csv(all_rounds_csv, index=False)
    round_one_csv = tmp_path / "round_one.csv"
    pd.DataFrame(
        {
            "round": [1],
            "cohort": ["selected"],
            "metric": ["pred__score_selected"],
            "summary": ["mean"],
            "value": [0.2],
        }
    ).to_csv(round_one_csv, index=False)
    latest_csv = tmp_path / "latest.csv"
    pd.DataFrame(
        {
            "round": [2],
            "cohort": ["selected"],
            "metric": ["pred__score_selected"],
            "summary": ["mean"],
            "value": [0.3],
        }
    ).to_csv(latest_csv, index=False)

    tidy_metadata = {
        "capability": {"tidy_available": True},
        "tidy_schema": ["round", "cohort", "metric", "summary", "value"],
    }
    image_only_metadata = {"capability": {"tidy_available": False}, "tidy_schema": []}
    entry = {
        "expected_final_round": 2,
        "plots": [
            {
                "name": "score_selected_over_rounds",
                "kind": "metric_over_rounds",
                "status": "written",
                "rounds": "all",
                "media_paths": [_png(tmp_path / "all_rounds.png")],
                "tidy_csv_paths": [str(all_rounds_csv)],
                "metadata": tidy_metadata,
            },
            {
                "name": "score_selected_over_rounds",
                "kind": "metric_over_rounds",
                "status": "written",
                "rounds": [1],
                "media_paths": [_png(tmp_path / "round_one.png")],
                "tidy_csv_paths": [str(round_one_csv)],
                "metadata": tidy_metadata,
            },
            {
                "name": "score_selected_over_rounds",
                "kind": "metric_over_rounds",
                "status": "written",
                "rounds": "latest",
                "media_paths": [_png(tmp_path / "latest.png")],
                "tidy_csv_paths": [str(latest_csv)],
                "metadata": tidy_metadata,
            },
            {
                "name": "single_round_uncertainty_latest",
                "kind": "uncertainty_over_rounds",
                "status": "written",
                "rounds": [2],
                "media_paths": [_png(tmp_path / "uncertainty_over_rounds.png")],
                "tidy_csv_paths": [],
                "metadata": image_only_metadata,
            },
        ],
    }

    quality = _quality_for_configured_plot_entry(entry)

    assert quality == {"status": "ok", "problems": []}


def test_configured_plot_quality_rejects_stale_image_only_final_round(tmp_path: Path) -> None:
    from PIL import Image, ImageDraw

    media_path = tmp_path / "stale_uncertainty.png"
    image = Image.new("RGB", (320, 240), "white")
    draw = ImageDraw.Draw(image)
    draw.line((20, 220, 300, 20), fill="black", width=3)
    image.save(media_path)
    entry = {
        "expected_final_round": 2,
        "plots": [
            {
                "name": "single_round_uncertainty_latest",
                "kind": "uncertainty_over_rounds",
                "status": "written",
                "rounds": [1],
                "media_paths": [str(media_path)],
                "tidy_csv_paths": [],
                "metadata": {
                    "capability": {"round_scope": "single_round", "tidy_available": False},
                    "tidy_schema": [],
                },
            }
        ],
    }

    quality = _quality_for_configured_plot_entry(entry)

    assert quality["status"] == "attention"
    assert quality["problems"] == ["single_round_uncertainty_latest:round_scope_missing_final_round:2"]


def test_configured_plot_quality_rejects_missing_round_variant_artifacts(tmp_path: Path) -> None:
    from PIL import Image, ImageDraw

    media_path = tmp_path / "stale_uncertainty.png"
    image = Image.new("RGB", (320, 240), "white")
    draw = ImageDraw.Draw(image)
    draw.line((20, 220, 300, 20), fill="black", width=3)
    image.save(media_path)
    entry = {
        "expected_final_round": 2,
        "expected_configured_plot_specs": [
            {
                "name": "single_round_uncertainty_latest",
                "kind": "uncertainty_over_rounds",
                "enabled": True,
                "round_selector": "latest",
                "round_variants": "each",
            }
        ],
        "plots": [
            {
                "name": "single_round_uncertainty_latest",
                "kind": "uncertainty_over_rounds",
                "status": "written",
                "rounds": "latest",
                "media_paths": [str(media_path)],
                "tidy_csv_paths": [],
                "metadata": {
                    "capability": {"round_scope": "single_round", "tidy_available": False},
                    "tidy_schema": [],
                },
            }
        ],
    }

    quality = _quality_for_configured_plot_entry(entry)

    assert quality["status"] == "attention"
    assert quality["problems"] == [
        "single_round_uncertainty_latest:configured_plot_missing_scopes:r0,r1,r2",
        "single_round_uncertainty_latest:configured_plot_unexpected_scopes:latest",
    ]


def test_configured_plot_quality_requires_tidy_csv_only_when_declared(tmp_path: Path) -> None:
    from PIL import Image, ImageDraw

    media_path = tmp_path / "score.png"
    image = Image.new("RGB", (320, 240), "white")
    draw = ImageDraw.Draw(image)
    draw.line((20, 220, 300, 20), fill="black", width=3)
    image.save(media_path)
    entry = {
        "expected_final_round": 2,
        "plots": [
            {
                "name": "score_selected_over_rounds",
                "kind": "metric_over_rounds",
                "status": "written",
                "rounds": "all",
                "media_paths": [str(media_path)],
                "tidy_csv_paths": [],
                "metadata": {
                    "capability": {"tidy_available": True},
                    "tidy_schema": ["round", "cohort", "metric", "summary", "value"],
                },
            }
        ],
    }

    quality = _quality_for_configured_plot_entry(entry)

    assert quality["status"] == "attention"
    assert quality["problems"] == ["score_selected_over_rounds:tidy_csv_missing"]


def test_configured_plot_quality_requires_vector_reference_only_when_configured(tmp_path: Path) -> None:
    from PIL import Image, ImageDraw

    media_path = tmp_path / "vector_summary.png"
    image = Image.new("RGB", (320, 240), "white")
    draw = ImageDraw.Draw(image)
    draw.line((20, 220, 300, 20), fill="black", width=3)
    image.save(media_path)
    tidy_path = tmp_path / "vector_summary.csv"
    pd.DataFrame(
        {
            "row_type": ["cohort_mean", "cohort_mean"],
            "round": [0, 0],
            "cohort": ["selected", "selected"],
            "channel": ["tf_count__lexA", "tf_count__cpxR_plus_baeR"],
            "value": [0.25, 0.75],
            "n": [12, 12],
        }
    ).to_csv(tidy_path, index=False)
    plot = {
        "name": "selected_target_vector_summary",
        "kind": "vector_summary_heatmap",
        "status": "written",
        "rounds": "all",
        "media_paths": [str(media_path)],
        "tidy_csv_paths": [str(tidy_path)],
        "metadata": {
            "capability": {"tidy_available": True},
            "tidy_schema": ["row_type", "round", "cohort", "channel", "value", "n"],
        },
    }

    optional_quality = _quality_for_configured_plot_entry(
        {"expected_final_round": 0, "plots": [{**plot, "params": {"include_reference_vector": False}}]}
    )
    required_quality = _quality_for_configured_plot_entry(
        {"expected_final_round": 0, "plots": [{**plot, "params": {"include_reference_vector": True}}]}
    )

    assert optional_quality == {"status": "ok", "problems": []}
    assert required_quality["status"] == "attention"
    assert required_quality["problems"] == ["selected_target_vector_summary:tidy_csv_missing_reference_vector"]


def test_configured_plot_quality_rejects_stale_manifest_freshness(tmp_path: Path) -> None:
    from PIL import Image, ImageDraw

    media_path = tmp_path / "score.png"
    image = Image.new("RGB", (320, 240), "white")
    draw = ImageDraw.Draw(image)
    draw.line((20, 220, 300, 20), fill="black", width=3)
    image.save(media_path)
    entry = {
        "expected_final_round": 2,
        "plots": [
            {
                "name": "score_selected_over_rounds",
                "kind": "metric_over_rounds",
                "status": "written",
                "freshness": {"status": "stale"},
                "rounds": "all",
                "media_paths": [str(media_path)],
                "tidy_csv_paths": [],
                "metadata": {"capability": {"tidy_available": False}, "tidy_schema": []},
            }
        ],
    }

    quality = _quality_for_configured_plot_entry(entry)

    assert quality["status"] == "attention"
    assert quality["problems"] == ["score_selected_over_rounds:freshness_not_fresh:stale"]
