"""
--------------------------------------------------------------------------------
<dnadesign project>
src/dnadesign/opal/src/analysis/dashboard/artifacts.py

Resolves round artifact locations for dashboard notebooks. Translates campaign
workdir + round into artifact paths.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from pathlib import Path


def resolve_round_artifacts(
    workdir: Path | None,
    *,
    as_of_round: int | None,
) -> tuple[dict[str, str] | None, str | None]:
    if workdir is None:
        return None, "Campaign workdir unavailable."
    if as_of_round is None:
        return None, "As-of round is required to resolve artifacts."
    round_dir = Path(workdir) / "outputs" / "rounds" / f"round_{int(as_of_round)}"
    if not round_dir.exists():
        return None, f"Round directory not found: {round_dir}"
    artifacts: dict[str, str] = {"round_dir": str(round_dir)}
    model_path = round_dir / "model" / "model.joblib"
    if model_path.exists():
        artifacts["model/model.joblib"] = str(model_path)
    return artifacts, None
