"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/opal/tests/notebooks/test_layered_scatter_integrity.py

Test fail-closed integrity checks for manifest-backed notebook scatter data.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from dnadesign.opal.src.analysis.notebook_components import layered_scatter as layered_scatter_module
from dnadesign.opal.src.analysis.notebook_components.layered_scatter import (
    build_notebook_layered_scatter_contract,
)
from dnadesign.opal.src.core.utils import OpalError, file_sha256
from dnadesign.opal.src.registries.plots import describe_plot_kind


def _choice(workdir: Path, *, tidy_path: Path | None = None) -> dict[str, object]:
    plot_root = workdir / "outputs" / "plots"
    plot_root.mkdir(parents=True, exist_ok=True)
    path = tidy_path or plot_root / "frontier.csv"
    pd.DataFrame(
        {
            "id": ["pool", "observed"],
            "record_kind": ["prediction", "observed_label"],
            "selected": [False, False],
            "batch_key": [None, "batch_0"],
            "display_label": [None, "Control"],
            "response_separation": [0.1, 0.2],
            "on_magnitude_floor": [1.1, 1.2],
            "off_constraint_margin": [-0.1, 0.2],
        }
    ).to_csv(path, index=False)
    return {
        "workdir": str(workdir),
        "manifest": {
            "kind": "response_magnitude_feasibility_frontier",
            "run_id": "r0",
            "rounds": [0],
            "selection_view_id": "view-a",
            "tidy_csv": str(path),
            "outputs": [
                {
                    "role": "tidy_csv",
                    "path": str(path),
                    "sha256": file_sha256(path),
                }
            ],
            "metadata": describe_plot_kind("response_magnitude_feasibility_frontier"),
            "artifact_metadata": {
                "notebook_view": {
                    "title": "Candidate constraint landscape",
                    "context": "Configured view",
                    "x_label": "Response separation",
                    "y_label": "ON fluorescence",
                    "color_label": "OFF clearance",
                    "x_boundary": 0.0,
                    "y_boundary": 0.0,
                    "color_extent": 1.0,
                    "x_limits": [-0.5, 0.8],
                    "y_limits": [-0.5, 1.8],
                }
            },
        },
    }


def _rewrite_tidy_and_rebind(choice: dict[str, object], rows: pd.DataFrame) -> None:
    manifest = choice["manifest"]
    assert isinstance(manifest, dict)
    tidy_path = Path(str(manifest["tidy_csv"]))
    rows.to_csv(tidy_path, index=False)
    outputs = manifest["outputs"]
    assert isinstance(outputs, list)
    output = outputs[0]
    assert isinstance(output, dict)
    output["sha256"] = file_sha256(tidy_path)


def test_layered_scatter_rejects_mutated_tidy_table_before_csv_read(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    workdir = tmp_path / "campaign"
    choice = _choice(workdir)
    tidy_path = Path(str(choice["manifest"]["tidy_csv"]))  # type: ignore[index]
    tidy_path.write_text("id\nmutated\n", encoding="utf-8")

    def _unexpected_read(*args: object, **kwargs: object) -> None:
        raise AssertionError("CSV parsing must not precede digest verification")

    monkeypatch.setattr(layered_scatter_module.pd, "read_csv", _unexpected_read)

    with pytest.raises(OpalError, match="SHA-256 does not match"):
        build_notebook_layered_scatter_contract(choice)


def test_layered_scatter_rejects_tidy_path_escape_before_csv_read(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    workdir = tmp_path / "campaign"
    outside = tmp_path / "outside.csv"
    choice = _choice(workdir, tidy_path=outside)

    def _unexpected_read(*args: object, **kwargs: object) -> None:
        raise AssertionError("CSV parsing must not precede path verification")

    monkeypatch.setattr(layered_scatter_module.pd, "read_csv", _unexpected_read)

    with pytest.raises(OpalError, match="outside the campaign plot root"):
        build_notebook_layered_scatter_contract(choice)


@pytest.mark.parametrize(
    ("row_index", "column", "value", "message"),
    [
        (0, "record_kind", "other", "record_kind values"),
        (0, "response_separation", float("inf"), "finite numeric"),
        (0, "on_magnitude_floor", float("nan"), "finite numeric"),
        (0, "off_constraint_margin", float("-inf"), "finite numeric"),
        (0, "selected", "yes", "selected values must be boolean or null"),
        (0, "selected", None, "prediction rows require boolean selected values"),
        (1, "selected", True, "observed rows require selected to be false or null"),
        (0, "batch_key", "batch_prediction", "prediction rows require null batch IDs"),
        (1, "batch_key", None, "observed rows require non-empty batch IDs"),
    ],
)
def test_layered_scatter_rejects_digest_valid_semantic_corruption(
    tmp_path: Path,
    row_index: int,
    column: str,
    value: object,
    message: str,
) -> None:
    choice = _choice(tmp_path / "campaign")
    manifest = choice["manifest"]
    assert isinstance(manifest, dict)
    rows = pd.read_csv(Path(str(manifest["tidy_csv"])))
    if column == "selected":
        rows[column] = rows[column].astype(object)
    rows.loc[row_index, column] = value
    _rewrite_tidy_and_rebind(choice, rows)

    with pytest.raises(ValueError, match=message):
        build_notebook_layered_scatter_contract(choice)
