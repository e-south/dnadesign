"""
--------------------------------------------------------------------------------
<cruncher project>
src/dnadesign/cruncher/tests/cli/test_portfolio_cli.py

CLI contract tests for the Portfolio command group.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import os
import re
import tempfile
from pathlib import Path

from typer.testing import CliRunner

import dnadesign.cruncher.cli.commands.portfolio as portfolio_cli
from dnadesign.cruncher.cli.app import app

runner = CliRunner()
ANSI_RE = re.compile(r"\x1b\[[0-9;]*[mK]")


def combined_output(result) -> str:
    stderr = getattr(result, "stderr", "")
    return ANSI_RE.sub("", f"{result.output}{stderr}")


def test_root_help_includes_portfolio_group() -> None:
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    assert "portfolio" in result.output


def test_portfolio_run_requires_spec_option() -> None:
    result = runner.invoke(app, ["portfolio", "run"])
    assert result.exit_code != 0
    assert "--spec" in combined_output(result)


def test_portfolio_run_spec_path_must_be_file_not_directory(tmp_path: Path) -> None:
    portfolio_workspace = tmp_path / "workspaces" / "portfolio"
    spec_dir = portfolio_workspace / "configs"
    spec_dir.mkdir(parents=True, exist_ok=True)
    (spec_dir / "master_all_workspaces.portfolio.yaml").write_text("portfolio: {}\n")

    result = runner.invoke(app, ["portfolio", "run", "--spec", str(spec_dir)])
    assert result.exit_code == 1
    assert "Portfolio spec path must be a file" in result.output
    assert ".portfolio.yaml" in result.output


def test_portfolio_run_resolves_relative_spec_from_init_cwd(tmp_path: Path, monkeypatch) -> None:
    repo_root = tmp_path / "repo"
    workspace = repo_root / "workspaces" / "portfolio_workspace"
    workspace.mkdir(parents=True, exist_ok=True)
    expected_spec = workspace / "handoff.portfolio.yaml"
    expected_spec.write_text(
        "portfolio: {schema_version: 3, name: handoff, execution: {mode: aggregate_only}, sources: []}\n"
    )
    emitted_run_dir = workspace / "outputs" / "portfolios" / "handoff" / "abc123"

    captured: dict[str, object] = {}

    def _fake_run_portfolio(
        spec_path: Path,
        *,
        force_overwrite: bool = False,
        prepare_ready_policy: str = "rerun",
        on_event=None,
    ) -> Path:
        _ = on_event
        captured["spec_path"] = spec_path
        captured["force_overwrite"] = force_overwrite
        captured["prepare_ready_policy"] = prepare_ready_policy
        return emitted_run_dir

    monkeypatch.setattr(
        portfolio_cli,
        "portfolio_preflight_payload",
        lambda path: {
            "spec_path": str(path),
            "execution_mode": "aggregate_only",
            "ready_source_ids": [],
            "unready_source_ids": [],
            "source_count": 0,
            "sources": [],
        },
    )

    monkeypatch.setattr(portfolio_cli, "run_portfolio", _fake_run_portfolio)
    monkeypatch.setattr(
        portfolio_cli,
        "portfolio_show_payload",
        lambda path: {
            "portfolio_name": "handoff",
            "portfolio_id": "abc123",
            "status": "completed",
            "manifest_path": str(path / "portfolio" / "portfolio_manifest.json"),
            "status_path": str(path / "portfolio" / "portfolio_status.json"),
            "table_paths": [str(path / "tables" / "table__handoff_windows_long.parquet")],
            "plot_paths": [
                str(workspace / "outputs" / "plots" / "portfolio__handoff__abc123__plot__source_tradeoff.pdf")
            ],
            "n_sources": 2,
            "n_selected_elites": 24,
        },
    )
    monkeypatch.setattr(portfolio_cli.sys.stdin, "isatty", lambda: True)

    monkeypatch.chdir(repo_root)
    result = runner.invoke(
        app,
        ["portfolio", "run", "--spec", "handoff.portfolio.yaml"],
        env={"INIT_CWD": str(workspace)},
    )

    assert result.exit_code == 0
    assert captured["spec_path"] == expected_spec.resolve()
    assert captured["force_overwrite"] is False
    assert captured["prepare_ready_policy"] == "rerun"


def test_portfolio_run_prepare_ready_skip_passes_policy_to_workflow(tmp_path: Path, monkeypatch) -> None:
    repo_root = tmp_path / "repo"
    workspace = repo_root / "workspaces" / "portfolio_workspace"
    workspace.mkdir(parents=True, exist_ok=True)
    expected_spec = workspace / "handoff.portfolio.yaml"
    expected_spec.write_text(
        "portfolio: {schema_version: 3, name: handoff, execution: {mode: prepare_then_aggregate}, sources: []}\n"
    )
    emitted_run_dir = workspace / "outputs" / "portfolios" / "handoff" / "abc123"

    captured: dict[str, object] = {}

    def _fake_run_portfolio(
        spec_path: Path,
        *,
        force_overwrite: bool = False,
        prepare_ready_policy: str = "rerun",
        on_event=None,
    ) -> Path:
        _ = spec_path
        _ = force_overwrite
        _ = on_event
        captured["prepare_ready_policy"] = prepare_ready_policy
        return emitted_run_dir

    monkeypatch.setattr(portfolio_cli, "run_portfolio", _fake_run_portfolio)
    monkeypatch.setattr(
        portfolio_cli,
        "portfolio_preflight_payload",
        lambda path: {
            "spec_path": str(path),
            "execution_mode": "prepare_then_aggregate",
            "ready_source_ids": ["pairwise_cpxr_baer"],
            "unready_source_ids": ["pairwise_cpxr_lexa"],
            "source_count": 2,
            "sources": [
                {"source_id": "pairwise_cpxr_baer", "workspace_name": "pairwise_cpxr_baer", "ready": True},
                {"source_id": "pairwise_cpxr_lexa", "workspace_name": "pairwise_cpxr_lexa", "ready": False},
            ],
        },
    )
    monkeypatch.setattr(
        portfolio_cli,
        "portfolio_show_payload",
        lambda path: {
            "portfolio_name": "handoff",
            "portfolio_id": "abc123",
            "status": "completed",
            "manifest_path": str(path / "portfolio" / "portfolio_manifest.json"),
            "status_path": str(path / "portfolio" / "portfolio_status.json"),
            "table_paths": [str(path / "tables" / "table__handoff_windows_long.parquet")],
            "plot_paths": [],
            "n_sources": 2,
            "n_selected_elites": 24,
        },
    )
    monkeypatch.chdir(repo_root)
    result = runner.invoke(
        app,
        ["portfolio", "run", "--spec", "handoff.portfolio.yaml", "--prepare-ready", "skip"],
        env={"INIT_CWD": str(workspace)},
    )

    assert result.exit_code == 0
    assert captured["prepare_ready_policy"] == "skip"


def test_portfolio_run_prepare_ready_skip_progress_totals_match_events(tmp_path: Path, monkeypatch) -> None:
    repo_root = tmp_path / "repo"
    workspace = repo_root / "workspaces" / "portfolio_workspace"
    workspace.mkdir(parents=True, exist_ok=True)
    expected_spec = workspace / "handoff.portfolio.yaml"
    expected_spec.write_text(
        "portfolio: {schema_version: 3, name: handoff, execution: {mode: prepare_then_aggregate}, sources: []}\n"
    )
    emitted_run_dir = workspace / "outputs" / "portfolios" / "handoff" / "abc123"

    class _StrictProgress:
        def __init__(self, *args, **kwargs) -> None:
            _ = args
            _ = kwargs
            self._totals: dict[int, float] = {}
            self._values: dict[int, float] = {}
            self._next_task_id = 1

        def add_task(self, _description: str, *, total: float, **kwargs) -> int:
            task_id = self._next_task_id
            self._next_task_id += 1
            self._totals[task_id] = float(total)
            self._values[task_id] = 0.0
            _ = kwargs
            return task_id

        def advance(self, task_id: int, amount: float = 1.0) -> None:
            self._values[task_id] += float(amount)
            if self._values[task_id] > self._totals[task_id]:
                raise RuntimeError("progress overran task total")

        def update(self, task_id: int, **kwargs) -> None:
            _ = task_id
            _ = kwargs

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb) -> bool:
            _ = exc_type
            _ = exc
            _ = tb
            return False

    def _fake_run_portfolio(
        spec_path: Path,
        *,
        force_overwrite: bool = False,
        prepare_ready_policy: str = "rerun",
        on_event=None,
    ) -> Path:
        _ = spec_path
        _ = force_overwrite
        _ = prepare_ready_policy
        assert on_event is not None
        on_event("prepare_source_skipped", {"source_id": "pairwise_cpxr_baer"})
        on_event("prepare_source_completed", {"source_id": "pairwise_cpxr_lexa"})
        on_event("aggregate_source_completed", {"source_id": "pairwise_cpxr_baer"})
        on_event("aggregate_source_completed", {"source_id": "pairwise_cpxr_lexa"})
        return emitted_run_dir

    monkeypatch.setattr(portfolio_cli, "Progress", _StrictProgress)
    monkeypatch.setattr(portfolio_cli, "_progress_enabled", lambda: True)
    monkeypatch.setattr(portfolio_cli, "run_portfolio", _fake_run_portfolio)
    monkeypatch.setattr(
        portfolio_cli,
        "portfolio_preflight_payload",
        lambda path: {
            "spec_path": str(path),
            "execution_mode": "prepare_then_aggregate",
            "ready_source_ids": ["pairwise_cpxr_baer"],
            "unready_source_ids": ["pairwise_cpxr_lexa"],
            "source_count": 2,
            "sources": [
                {"source_id": "pairwise_cpxr_baer", "workspace_name": "pairwise_cpxr_baer", "ready": True},
                {"source_id": "pairwise_cpxr_lexa", "workspace_name": "pairwise_cpxr_lexa", "ready": False},
            ],
        },
    )
    monkeypatch.setattr(
        portfolio_cli,
        "portfolio_show_payload",
        lambda path: {
            "portfolio_name": "handoff",
            "portfolio_id": "abc123",
            "status": "completed",
            "manifest_path": str(path / "portfolio" / "portfolio_manifest.json"),
            "status_path": str(path / "portfolio" / "portfolio_status.json"),
            "table_paths": [str(path / "tables" / "table__handoff_windows_long.parquet")],
            "plot_paths": [],
            "n_sources": 2,
            "n_selected_elites": 24,
        },
    )
    monkeypatch.chdir(repo_root)
    result = runner.invoke(
        app,
        ["portfolio", "run", "--spec", "handoff.portfolio.yaml", "--prepare-ready", "skip"],
        env={"INIT_CWD": str(workspace)},
    )

    assert result.exit_code == 0


def test_portfolio_run_progress_updates_task_descriptions(tmp_path: Path, monkeypatch) -> None:
    repo_root = tmp_path / "repo"
    workspace = repo_root / "workspaces" / "portfolio_workspace"
    workspace.mkdir(parents=True, exist_ok=True)
    expected_spec = workspace / "handoff.portfolio.yaml"
    expected_spec.write_text(
        "portfolio: {schema_version: 3, name: handoff, execution: {mode: prepare_then_aggregate}, sources: []}\n"
    )
    emitted_run_dir = workspace / "outputs" / "portfolios" / "handoff" / "abc123"

    class _StrictProgress:
        descriptions: list[str] = []

        def __init__(self, *args, **kwargs) -> None:
            _ = args
            _ = kwargs

        def add_task(self, _description: str, *, total: float, **kwargs) -> int:
            _ = total
            _ = kwargs
            return 1

        def advance(self, task_id: int, amount: float = 1.0) -> None:
            _ = task_id
            _ = amount

        def update(self, task_id: int, **kwargs) -> None:
            _ = task_id
            description = kwargs.get("description")
            if isinstance(description, str):
                self.descriptions.append(description)

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb) -> bool:
            _ = exc_type
            _ = exc
            _ = tb
            return False

    def _fake_run_portfolio(
        spec_path: Path,
        *,
        force_overwrite: bool = False,
        prepare_ready_policy: str = "rerun",
        on_event=None,
    ) -> Path:
        _ = spec_path
        _ = force_overwrite
        _ = prepare_ready_policy
        assert on_event is not None
        on_event("prepare_source_started", {"source_id": "pairwise_cpxr_baer"})
        on_event("prepare_source_completed", {"source_id": "pairwise_cpxr_baer"})
        on_event("aggregate_source_started", {"source_id": "pairwise_cpxr_baer"})
        on_event("aggregate_source_completed", {"source_id": "pairwise_cpxr_baer"})
        return emitted_run_dir

    monkeypatch.setattr(portfolio_cli, "Progress", _StrictProgress)
    monkeypatch.setattr(portfolio_cli, "_progress_enabled", lambda: True)
    monkeypatch.setattr(portfolio_cli, "run_portfolio", _fake_run_portfolio)
    monkeypatch.setattr(
        portfolio_cli,
        "portfolio_preflight_payload",
        lambda path: {
            "spec_path": str(path),
            "execution_mode": "prepare_then_aggregate",
            "ready_source_ids": [],
            "unready_source_ids": ["pairwise_cpxr_baer"],
            "source_count": 1,
            "sources": [
                {"source_id": "pairwise_cpxr_baer", "workspace_name": "pairwise_cpxr_baer", "ready": False},
            ],
        },
    )
    monkeypatch.setattr(
        portfolio_cli,
        "portfolio_show_payload",
        lambda path: {
            "portfolio_name": "handoff",
            "portfolio_id": "abc123",
            "status": "completed",
            "manifest_path": str(path / "portfolio" / "portfolio_manifest.json"),
            "status_path": str(path / "portfolio" / "portfolio_status.json"),
            "table_paths": [str(path / "tables" / "table__handoff_windows_long.parquet")],
            "plot_paths": [],
            "n_sources": 1,
            "n_selected_elites": 8,
        },
    )
    monkeypatch.chdir(repo_root)

    result = runner.invoke(
        app,
        ["portfolio", "run", "--spec", "handoff.portfolio.yaml", "--prepare-ready", "rerun"],
        env={"INIT_CWD": str(workspace)},
    )
    assert result.exit_code == 0
    assert any("Prepare source 1/1: pairwise_cpxr_baer" in item for item in _StrictProgress.descriptions)
    assert any("Aggregate source 1/1: pairwise_cpxr_baer" in item for item in _StrictProgress.descriptions)


def test_portfolio_run_prompt_prepare_ready_policy_requires_tty(monkeypatch) -> None:
    with tempfile.TemporaryDirectory() as td:
        repo_root = Path(td) / "repo"
        workspace = repo_root / "workspaces" / "portfolio_workspace"
        workspace.mkdir(parents=True, exist_ok=True)
        expected_spec = workspace / "handoff.portfolio.yaml"
        expected_spec.write_text(
            "portfolio: {schema_version: 3, name: handoff, execution: {mode: prepare_then_aggregate}, sources: []}\n"
        )

        monkeypatch.setattr(
            portfolio_cli,
            "portfolio_preflight_payload",
            lambda path: {
                "spec_path": str(path),
                "execution_mode": "prepare_then_aggregate",
                "ready_source_ids": ["pairwise_cpxr_baer"],
                "unready_source_ids": [],
                "source_count": 1,
                "sources": [
                    {"source_id": "pairwise_cpxr_baer", "workspace_name": "pairwise_cpxr_baer", "ready": True},
                ],
            },
        )
        monkeypatch.setattr(portfolio_cli.sys.stdin, "isatty", lambda: False)
        monkeypatch.chdir(repo_root)
        result = runner.invoke(
            app,
            ["portfolio", "run", "--spec", "handoff.portfolio.yaml", "--prepare-ready", "prompt"],
            env={"INIT_CWD": str(workspace)},
        )
        assert result.exit_code == 1
        assert "--prepare-ready skip|rerun" in combined_output(result)


def test_portfolio_run_non_tty_does_not_construct_progress(monkeypatch, tmp_path: Path) -> None:
    repo_root = tmp_path / "repo"
    workspace = repo_root / "workspaces" / "portfolio_workspace"
    workspace.mkdir(parents=True, exist_ok=True)
    expected_spec = workspace / "handoff.portfolio.yaml"
    expected_spec.write_text(
        "portfolio: {schema_version: 3, name: handoff, execution: {mode: aggregate_only}, sources: []}\n"
    )

    monkeypatch.setattr(
        portfolio_cli,
        "portfolio_preflight_payload",
        lambda path: {
            "spec_path": str(path),
            "execution_mode": "aggregate_only",
            "ready_source_ids": [],
            "unready_source_ids": [],
            "source_count": 0,
            "sources": [],
        },
    )
    monkeypatch.setattr(
        portfolio_cli,
        "run_portfolio",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            ValueError("Portfolio run directory already exists: /tmp/demo. Use --force-overwrite.")
        ),
    )

    class _ProgressShouldNotConstruct:
        def __init__(self, *args, **kwargs) -> None:
            _ = (args, kwargs)
            raise AssertionError("Progress must not be constructed in non-interactive mode.")

    monkeypatch.setattr(portfolio_cli, "Progress", _ProgressShouldNotConstruct)
    monkeypatch.setattr(portfolio_cli.sys.stdin, "isatty", lambda: False)
    monkeypatch.setattr(portfolio_cli.sys.stdout, "isatty", lambda: False)
    monkeypatch.chdir(repo_root)

    result = runner.invoke(
        app,
        ["portfolio", "run", "--spec", "handoff.portfolio.yaml"],
        env={"INIT_CWD": str(workspace)},
    )
    assert result.exit_code == 1
    assert "--force-overwrite" in combined_output(result)


def test_run_with_noninteractive_env_sets_and_restores_value(monkeypatch) -> None:
    monkeypatch.setenv("CRUNCHER_NONINTERACTIVE", "0")
    seen: list[str | None] = []

    result = portfolio_cli._run_with_noninteractive_env(lambda: seen.append(os.environ.get("CRUNCHER_NONINTERACTIVE")))

    assert result is None
    assert seen == ["1"]
    assert os.environ.get("CRUNCHER_NONINTERACTIVE") == "0"


def test_run_with_noninteractive_env_unsets_when_not_preexisting(monkeypatch) -> None:
    monkeypatch.delenv("CRUNCHER_NONINTERACTIVE", raising=False)
    seen: list[str | None] = []

    result = portfolio_cli._run_with_noninteractive_env(lambda: seen.append(os.environ.get("CRUNCHER_NONINTERACTIVE")))

    assert result is None
    assert seen == ["1"]
    assert os.environ.get("CRUNCHER_NONINTERACTIVE") is None


def test_portfolio_show_requires_run_option() -> None:
    result = runner.invoke(app, ["portfolio", "show"])
    assert result.exit_code != 0
    assert "--run" in combined_output(result)


def test_portfolio_show_prints_source_selected_elite_counts(monkeypatch, tmp_path: Path) -> None:
    run_dir = tmp_path / "outputs" / "portfolios" / "handoff" / "abc123"
    run_dir.mkdir(parents=True, exist_ok=True)

    monkeypatch.setattr(
        portfolio_cli,
        "portfolio_show_payload",
        lambda path: {
            "portfolio_name": "handoff",
            "portfolio_id": "abc123",
            "status": "completed",
            "manifest_path": str(path / "portfolio" / "portfolio_manifest.json"),
            "status_path": str(path / "portfolio" / "portfolio_status.json"),
            "table_paths": [str(path / "tables" / "table__handoff_windows_long.parquet")],
            "plot_paths": [],
            "n_sources": 2,
            "n_selected_elites": 12,
            "source_runs": [
                {"source_id": "demo_pairwise", "source_top_k": 8, "selected_elites": 4},
                {"source_id": "demo_multitf", "source_top_k": 8, "selected_elites": 8},
            ],
        },
    )

    result = runner.invoke(app, ["portfolio", "show", "--run", str(run_dir)])
    assert result.exit_code == 0
    assert "source: demo_pairwise elites=4/8" in result.output
    assert "source: demo_multitf elites=8/8" in result.output
