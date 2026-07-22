"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/studies/units/eco1_rt_repack/tests/materialization/review_deliverables/notebook_contract_text.py

Notebook contract source aggregation helpers.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from dnadesign.studies.units.eco1_rt_repack.tests._helpers import repo_root


def notebook_contract_text(notebook_text: str) -> tuple[str, str]:
    runtime_path = repo_root() / (
        "src/dnadesign/studies/units/eco1_rt_repack/operations/materialization/review_deliverables/notebook_runtime.py"
    )
    runtime_dir = runtime_path.parent
    runtime_text = "\n".join(
        path.read_text(encoding="utf-8")
        for path in (
            runtime_path,
            runtime_dir / "notebook_sae_features.py",
            runtime_dir / "notebook_sequences.py",
            runtime_dir / "notebook_selection_panel.py",
            runtime_dir / "notebook_selection_summary.py",
            runtime_dir / "notebook_structure_browser.py",
            runtime_dir / "notebook_visuals.py",
        )
    )
    return runtime_text, notebook_text + "\n" + runtime_text
