"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/folding/tests/test_assessment_process_isolation.py

Process lifecycle contracts for digest-addressed structure assessments.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import os
import signal
import subprocess
import time
from pathlib import Path

import pytest

import dnadesign.folding.src.api as folding_api
import dnadesign.folding.src.assessment.worker as assessment_worker
from dnadesign.folding import FoldingExecutionError, publish_structure_assessment
from dnadesign.folding.src.assessment import execution as assessment_execution
from dnadesign.folding.src.assessment._limits import (
    ARTIFACT_AGGREGATE_SIZE_LIMIT_BYTES,
    ARTIFACT_ENTRY_COUNT_LIMIT,
)
from dnadesign.folding.tests._assessment_fixtures import assessment_request, cli_assessment_request


@pytest.mark.skipif(os.name != "posix", reason="process-group timeout contract is POSIX-specific")
def test_structure_assessment_timeout_terminates_cli_descendants(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    executable = tmp_path / "fake-rnafold"
    descendant_marker = tmp_path / "descendant-survived"
    executable.write_text(
        "#!/usr/bin/env python3\n"
        "import os\n"
        "import pathlib\n"
        "import resource\n"
        "import subprocess\n"
        "import sys\n"
        "import time\n"
        "if '--version' in sys.argv:\n"
        "    print('RNAfold 2.7.2')\n"
        "    raise SystemExit(0)\n"
        "if resource.getrlimit(resource.RLIMIT_NPROC) != (0, 0):\n"
        "    pathlib.Path(os.environ['ASSESSMENT_DESCENDANT_MARKER']).write_text('limit-bypass')\n"
        "subprocess.Popen([\n"
        "    sys.executable,\n"
        "    '-c',\n"
        "    'import os,pathlib,time; time.sleep(0.5); '"
        '    \'pathlib.Path(os.environ["ASSESSMENT_DESCENDANT_MARKER"]).write_text("alive")\',\n'
        "])\n"
        "time.sleep(5)\n",
        encoding="utf-8",
    )
    executable.chmod(0o755)
    monkeypatch.setenv("ASSESSMENT_DESCENDANT_MARKER", descendant_marker.as_posix())
    output = tmp_path / "timed-out-cli-assessment"

    with pytest.raises(FoldingExecutionError, match="timed out"):
        publish_structure_assessment(
            cli_assessment_request(executable, timeout_seconds=0.1),
            output_dir=output,
        )

    time.sleep(0.7)
    assert not descendant_marker.exists()
    assert not output.exists()


@pytest.mark.skipif(os.name != "posix", reason="kernel no-fork contract is POSIX-specific")
def test_structure_assessment_cli_backend_cannot_spawn_detached_descendants(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    executable = tmp_path / "fake-rnafold"
    descendant_marker = tmp_path / "descendant-survived"
    executable.write_text(
        "#!/usr/bin/env python3\n"
        "import os\n"
        "import subprocess\n"
        "import sys\n"
        "if '--version' in sys.argv:\n"
        "    print('RNAfold 2.7.2')\n"
        "    raise SystemExit(0)\n"
        "subprocess.Popen([\n"
        "    sys.executable,\n"
        "    '-c',\n"
        "    'import os,pathlib,time; time.sleep(0.5); '"
        '    \'pathlib.Path(os.environ["ASSESSMENT_DESCENDANT_MARKER"]).write_text("alive")\',\n'
        "], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, start_new_session=True)\n"
        "sys.stdin.read()\n"
        "print('>hop:encoding/example')\n"
        "print('GCAUGC')\n"
        "print('((..)) (-1.20)')\n",
        encoding="utf-8",
    )
    executable.chmod(0o755)
    monkeypatch.setenv("ASSESSMENT_DESCENDANT_MARKER", descendant_marker.as_posix())

    output = tmp_path / "assessment"
    with pytest.raises(FoldingExecutionError, match="RNAfold CLI exited with status"):
        publish_structure_assessment(
            cli_assessment_request(executable, timeout_seconds=2.0),
            output_dir=output,
        )

    time.sleep(0.7)
    assert not descendant_marker.exists()
    assert not output.exists()


@pytest.mark.skipif(os.name != "posix", reason="CLI stream limit is kernel-enforced on POSIX")
@pytest.mark.parametrize(("file_descriptor", "label"), [(1, "stdout"), (2, "stderr")])
def test_structure_assessment_bounds_cli_backend_streams(
    tmp_path: Path,
    file_descriptor: int,
    label: str,
) -> None:
    executable = tmp_path / "fake-rnafold"
    executable.write_text(
        "#!/usr/bin/env python3\n"
        "import os\n"
        "import sys\n"
        "if '--version' in sys.argv:\n"
        "    print('RNAfold 2.7.2')\n"
        "    raise SystemExit(0)\n"
        "sys.stdin.read()\n"
        f"os.write({file_descriptor}, b'x' * {folding_api._BACKEND_STREAM_LIMIT_BYTES + 1})\n",
        encoding="utf-8",
    )
    executable.chmod(0o755)
    output = tmp_path / "assessment"

    with pytest.raises(FoldingExecutionError, match=rf"backend {label} exceeded.*byte limit"):
        publish_structure_assessment(
            cli_assessment_request(executable, timeout_seconds=2.0),
            output_dir=output,
        )

    assert not output.exists()


@pytest.mark.skipif(os.name != "posix", reason="kernel no-fork contract is POSIX-specific")
def test_structure_assessment_python_backend_cannot_double_fork(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    child_pid_path = tmp_path / "detached-child.pid"
    module_root = tmp_path / "fake-backend"
    module_root.mkdir()
    (module_root / "RNA.py").write_text(
        "__version__ = 'test-1.0'\n"
        "import os\n"
        "import pathlib\n"
        "import resource\n"
        "import time\n"
        "class Compound:\n"
        "    def mfe(self):\n"
        "        if resource.getrlimit(resource.RLIMIT_NPROC) != (0, 0):\n"
        "            pathlib.Path(os.environ['ASSESSMENT_CHILD_PID']).write_text('limit-bypass')\n"
        "        child = os.fork()\n"
        "        if child == 0:\n"
        "            os.setsid()\n"
        "            grandchild = os.fork()\n"
        "            if grandchild == 0:\n"
        "                open(os.environ['ASSESSMENT_CHILD_PID'], 'w').write(str(os.getpid()))\n"
        "                time.sleep(10)\n"
        "            raise SystemExit(0)\n"
        "        return '((..))', -1.2\n"
        "def fold_compound(sequence):\n"
        "    return Compound()\n",
        encoding="utf-8",
    )
    existing = os.environ.get("PYTHONPATH", "")
    monkeypatch.setenv(
        "PYTHONPATH",
        os.pathsep.join(part for part in (module_root.as_posix(), existing) if part),
    )
    monkeypatch.setenv("ASSESSMENT_CHILD_PID", child_pid_path.as_posix())
    output = tmp_path / "assessment"
    with pytest.raises(FoldingExecutionError, match="worker failed"):
        publish_structure_assessment(
            assessment_request(timeout_seconds=2.0),
            output_dir=output,
        )
    assert not child_pid_path.exists()
    assert not output.exists()


@pytest.mark.skipif(os.name != "posix", reason="worker stream limit is kernel-enforced on POSIX")
@pytest.mark.parametrize(("file_descriptor", "label"), [(1, "stdout"), (2, "stderr")])
def test_structure_assessment_bounds_worker_streams(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    file_descriptor: int,
    label: str,
) -> None:
    module_root = tmp_path / "fake-backend"
    module_root.mkdir()
    (module_root / "RNA.py").write_text(
        "import os\n"
        f"os.write({file_descriptor}, b'x' * {assessment_execution._WORKER_STREAM_LIMIT_BYTES + 1})\n"
        "__version__ = 'test-1.0'\n"
        "class Compound:\n"
        "    def mfe(self):\n"
        "        return '((..))', -1.2\n"
        "def fold_compound(sequence):\n"
        "    return Compound()\n",
        encoding="utf-8",
    )
    existing = os.environ.get("PYTHONPATH", "")
    monkeypatch.setenv(
        "PYTHONPATH",
        os.pathsep.join(part for part in (module_root.as_posix(), existing) if part),
    )
    output = tmp_path / "assessment"

    with pytest.raises(FoldingExecutionError, match=rf"worker {label} exceeded.*byte limit"):
        publish_structure_assessment(
            assessment_request(timeout_seconds=2.0),
            output_dir=output,
        )

    assert not output.exists()


def test_assessment_worker_delegates_the_only_deadline_to_its_supervisor(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, object] = {}
    request = object()
    request_path = tmp_path / "request.json"

    monkeypatch.setattr(
        assessment_worker,
        "load_prediction_request",
        lambda _path: (request, request_path),
    )

    def run_prediction_request(
        observed_request: object,
        **kwargs: object,
    ) -> None:
        captured["request"] = observed_request
        captured.update(kwargs)

    monkeypatch.setattr(assessment_worker, "run_prediction_request", run_prediction_request)
    monkeypatch.setattr(
        assessment_worker,
        "_normalize_preflight_output_dir",
        lambda path: captured.update(normalized_output_dir=path),
    )

    assert assessment_worker.main([request_path.as_posix(), (tmp_path / "output").as_posix()]) == 0
    assert captured["request"] is request
    assert captured["backend_timeout_seconds"] is None
    assert captured["deny_backend_child_processes"] is True
    assert captured["normalized_output_dir"] == tmp_path / "output"


def test_worker_artifact_budget_error_still_cleans_up_process_group(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    events: list[str] = []

    class FakeProcess:
        pid = 4102
        returncode = 0
        stdout = None
        stderr = None

        def poll(self) -> int:
            return None

        def communicate(self, *, timeout: float) -> tuple[str, str]:
            del timeout
            events.append("communicate")
            return "", ""

        def kill(self) -> None:
            events.append("kill")

        def wait(self, *, timeout: float) -> int:
            del timeout
            events.append("wait")
            return self.returncode

    monkeypatch.setattr(subprocess, "Popen", lambda *args, **kwargs: FakeProcess())
    monkeypatch.setattr(os, "killpg", lambda pid, sig: events.append(f"killpg:{pid}:{sig}"))
    monkeypatch.setattr(
        assessment_execution,
        "_enforce_artifact_budget",
        lambda _root: (_ for _ in ()).throw(FoldingExecutionError("artifact budget exceeded")),
    )

    artifact_root_descriptor = os.open(tmp_path, os.O_RDONLY | os.O_DIRECTORY)
    try:
        with pytest.raises(FoldingExecutionError, match="artifact budget exceeded"):
            assessment_execution.run_worker(
                tmp_path / "request.json",
                tmp_path / "output.json",
                artifact_root_descriptor=artifact_root_descriptor,
                timeout_seconds=1.0,
            )
    finally:
        os.close(artifact_root_descriptor)

    assert f"killpg:4102:{signal.SIGKILL}" in events
    assert "communicate" in events


def test_worker_artifact_budget_bounds_entry_count(tmp_path: Path) -> None:
    artifact_root = tmp_path / "artifacts"
    artifact_root.mkdir()
    for index in range(ARTIFACT_ENTRY_COUNT_LIMIT + 1):
        (artifact_root / f"artifact-{index:03d}").touch()

    descriptor = os.open(artifact_root, os.O_RDONLY | os.O_DIRECTORY)
    try:
        with pytest.raises(FoldingExecutionError, match=rf"{ARTIFACT_ENTRY_COUNT_LIMIT}-entry limit"):
            assessment_execution._enforce_artifact_budget(descriptor)
    finally:
        os.close(descriptor)


def test_worker_artifact_budget_bounds_aggregate_bytes(tmp_path: Path) -> None:
    artifact_root = tmp_path / "artifacts"
    artifact_root.mkdir()
    for index in range(2):
        with (artifact_root / f"artifact-{index}").open("wb") as handle:
            handle.truncate(ARTIFACT_AGGREGATE_SIZE_LIMIT_BYTES // 2)

    descriptor = os.open(artifact_root, os.O_RDONLY | os.O_DIRECTORY)
    try:
        with pytest.raises(FoldingExecutionError, match=rf"{ARTIFACT_AGGREGATE_SIZE_LIMIT_BYTES}-byte aggregate"):
            assessment_execution._enforce_artifact_budget(descriptor)
    finally:
        os.close(descriptor)


def test_worker_artifact_budget_remains_bound_to_renamed_stage(tmp_path: Path) -> None:
    artifact_root = tmp_path / "artifacts"
    displaced_root = tmp_path / "displaced-artifacts"
    artifact_root.mkdir()
    descriptor = os.open(artifact_root, os.O_RDONLY | os.O_DIRECTORY)
    try:
        artifact_root.rename(displaced_root)
        artifact_root.mkdir()
        for index in range(ARTIFACT_ENTRY_COUNT_LIMIT + 1):
            (displaced_root / f"artifact-{index:03d}").touch()

        with pytest.raises(FoldingExecutionError, match=rf"{ARTIFACT_ENTRY_COUNT_LIMIT}-entry limit"):
            assessment_execution._enforce_artifact_budget(descriptor)
    finally:
        os.close(descriptor)
