"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/infer/tests/cli/test_run_command_config_inputs.py

Run command contracts for config-driven local ingest.path inputs.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from pathlib import Path

from typer.testing import CliRunner

from dnadesign.infer.cli import app

_RUNNER = CliRunner()


def test_run_config_sequences_ingest_path_resolves_relative_to_config(monkeypatch, tmp_path: Path) -> None:
    (tmp_path / "inputs").mkdir(parents=True, exist_ok=True)
    (tmp_path / "inputs" / "seqs.txt").write_text("ACGT\nTGCA\n", encoding="utf-8")
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        """
model:
  id: evo2_7b
  device: cpu
  precision: fp32
  alphabet: dna
jobs:
  - id: j1
    operation: extract
    ingest:
      source: sequences
      path: inputs/seqs.txt
    outputs:
      - id: ll
        fn: evo2.log_likelihood
        format: float
""".strip()
        + "\n",
        encoding="utf-8",
    )

    captured: dict[str, object] = {}

    def _fake_run_job(*, inputs, model, job, progress_factory=None):
        captured["inputs"] = inputs
        captured["job_id"] = job.id
        return {"ll": [0.1, 0.2]}

    monkeypatch.setattr("dnadesign.infer.src.cli.commands.run.run_job", _fake_run_job)
    monkeypatch.setattr(
        "dnadesign.infer.src.cli.commands.run.run_with_progress",
        lambda *, progress, runner: runner(None),
    )
    monkeypatch.setattr("dnadesign.infer.src.cli.commands.run.render_outputs_summary", lambda *_a, **_k: None)

    result = _RUNNER.invoke(app, ["run", "--config", config_path.as_posix()])

    assert result.exit_code == 0, result.stdout
    assert captured["job_id"] == "j1"
    assert captured["inputs"] == ["ACGT", "TGCA"]
