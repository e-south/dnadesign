"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/seqledger/tests/docs_contract/test_layout.py

Structural layout and maintainer-boundary contracts for SeqLedger docs.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import re
import subprocess

from ..source_layout_inventory import TOP_LEVEL_SOURCE_PACKAGES
from .helpers import read_text, repo_root


def test_usr_layout_docs_cover_sanctioned_source_packages_and_new_support_surfaces() -> None:
    usr_agents = read_text("src/dnadesign/seqledger/AGENTS.md")
    code_map = read_text("src/dnadesign/seqledger/docs/reference/dataset-layout-and-code-map.md")
    api_doc = read_text("src/dnadesign/seqledger/docs/reference/python-api.md")
    introspection = read_text("src/dnadesign/seqledger/docs/architecture-introspection.md")

    for package_name in sorted(TOP_LEVEL_SOURCE_PACKAGES - {"storage"}):
        token = f"{package_name}/"
        assert token in usr_agents
        assert token in code_map

    for token in (
        "cli/support/resolution/dataset_targets.py",
        "cli/support/wiring/dependencies.py",
        "cli/support/wiring/registration.py",
        "cli/support/presentation/runtime.py",
        "src/dnadesign/seqledger/ops/sync_audit_drill.py",
        "src/dnadesign/devtools/tests/support/usr.py",
    ):
        assert token in code_map or token in introspection or token in usr_agents

    assert "Public import surface: `dnadesign.seqledger`" in api_doc
    assert "architecture-introspection.md" in api_doc


def test_usr_agents_describe_ops_scripts_assets_and_testsupport_boundaries() -> None:
    usr_agents = read_text("src/dnadesign/seqledger/AGENTS.md")

    assert "src/dnadesign/seqledger/ops/" in usr_agents
    assert "sync_audit_drill.py" in usr_agents
    assert "src/dnadesign/seqledger/scripts/" in usr_agents
    assert "not a public cross-tool API surface" in usr_agents
    assert "src/dnadesign/seqledger/assets/demo_material/" in usr_agents
    assert "src/dnadesign/devtools/tests/support/" in usr_agents
    assert (
        "Shared test fixtures consumed outside SeqLedger belong under `src/dnadesign/devtools/tests/support/`"
        in usr_agents
    )


def test_seqledger_quickstart_is_local_and_uses_the_installed_cli() -> None:
    quickstart = read_text("src/dnadesign/seqledger/docs/getting-started/cli-quickstart.md")
    shell_blocks = re.findall(r"```bash\n(.*?)```", quickstart, re.DOTALL)
    commands = "\n".join(shell_blocks[1:])

    assert "source .venv/bin/activate" in shell_blocks[0]
    assert "cat > sequences.csv" in commands
    for operation in ("namespace register", "init", "import", "attach", "validate", "export"):
        assert re.search(rf'seqledger --root "\$DATA_ROOT" {operation}\b', commands)
    assert "--strict" in commands
    assert "--out annotated_sequences.csv" in commands
    assert "materialize" not in commands
    assert "SEQLEDGER_SHOW_DEV_COMMANDS" not in commands
    assert "assets/demo_material" not in commands
    assert not (repo_root() / "src" / "dnadesign" / "usr" / "demo_material").exists()
    assert not (repo_root() / "src" / "dnadesign" / "usr" / "archived").exists()


def test_usr_surface_does_not_track_transient_cache_artifacts() -> None:
    result = subprocess.run(
        ["git", "ls-files", "src/dnadesign/seqledger"],
        cwd=repo_root(),
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    tracked = [
        path
        for path in result.stdout.splitlines()
        if path.endswith(".DS_Store") or "/__pycache__/" in path or path.endswith(".pyc")
    ]
    assert tracked == []
