from __future__ import annotations

from .helpers import Path, pd


def test_configured_plot_quality_respects_artifact_round_scope_and_optional_tidy_csv(tmp_path: Path) -> None:
    from dnadesign.studies.units.stress_ethanol_cipro_growth.opal_densegen_axis_probe.review.configured_plots import (
        _quality_for_configured_plot_entry,
    )

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

    from dnadesign.studies.units.stress_ethanol_cipro_growth.opal_densegen_axis_probe.review.configured_plots import (
        _quality_for_configured_plot_entry,
    )

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

    from dnadesign.studies.units.stress_ethanol_cipro_growth.opal_densegen_axis_probe.review.configured_plots import (
        _quality_for_configured_plot_entry,
    )

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

    from dnadesign.studies.units.stress_ethanol_cipro_growth.opal_densegen_axis_probe.review.configured_plots import (
        _quality_for_configured_plot_entry,
    )

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


def test_configured_plot_quality_rejects_stale_manifest_freshness(tmp_path: Path) -> None:
    from PIL import Image, ImageDraw

    from dnadesign.studies.units.stress_ethanol_cipro_growth.opal_densegen_axis_probe.review.configured_plots import (
        _quality_for_configured_plot_entry,
    )

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
