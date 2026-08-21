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

import psutil
import pytest

import dnadesign.folding.src.assessment.worker as assessment_worker
from dnadesign.folding import FoldingExecutionError, publish_structure_assessment
from dnadesign.folding.src.assessment import execution as assessment_execution
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
        "import subprocess\n"
        "import sys\n"
        "import time\n"
        "if '--version' in sys.argv:\n"
        "    print('RNAfold 2.7.2')\n"
        "    raise SystemExit(0)\n"
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


@pytest.mark.skipif(os.name != "posix", reason="process-group cleanup contract is POSIX-specific")
def test_structure_assessment_success_terminates_residual_cli_descendants(
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
        "import time; time.sleep(0.05)\n"
        "sys.stdin.read()\n"
        "print('>hop:encoding/example')\n"
        "print('GCAUGC')\n"
        "print('((..)) (-1.20)')\n",
        encoding="utf-8",
    )
    executable.chmod(0o755)
    monkeypatch.setenv("ASSESSMENT_DESCENDANT_MARKER", descendant_marker.as_posix())

    publish_structure_assessment(
        cli_assessment_request(executable, timeout_seconds=2.0),
        output_dir=tmp_path / "assessment",
    )

    time.sleep(0.7)
    assert not descendant_marker.exists()


@pytest.mark.skipif(os.name != "posix", reason="detached pipe contract is POSIX-specific")
def test_structure_assessment_timeout_does_not_wait_for_detached_pipe_holder(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    child_pid_path = tmp_path / "detached-child.pid"
    module_root = tmp_path / "fake-backend"
    module_root.mkdir()
    (module_root / "RNA.py").write_text(
        "__version__ = 'test-1.0'\n"
        "import os\n"
        "import subprocess\n"
        "import sys\n"
        "import time\n"
        "class Compound:\n"
        "    def mfe(self):\n"
        "        child = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(10)'], "
        "start_new_session=True)\n"
        "        open(os.environ['ASSESSMENT_CHILD_PID'], 'w').write(str(child.pid))\n"
        "        time.sleep(10)\n"
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
    started = time.monotonic()

    with pytest.raises(FoldingExecutionError, match="timed out"):
        publish_structure_assessment(
            assessment_request(timeout_seconds=0.5),
            output_dir=tmp_path / "assessment",
        )
    assert time.monotonic() - started < 2.0
    child_pid = int(child_pid_path.read_text(encoding="utf-8"))
    deadline = time.monotonic() + 1.0
    while psutil.pid_exists(child_pid) and time.monotonic() < deadline:
        time.sleep(0.01)
    assert not psutil.pid_exists(child_pid)


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
    assert captured["normalized_output_dir"] == tmp_path / "output"


def test_worker_communication_error_still_cleans_up_process_group(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    events: list[str] = []

    class FakeProcess:
        pid = 4102
        returncode = 0
        stdout = None
        stderr = None

        def communicate(self, *, timeout: float) -> tuple[str, str]:
            del timeout
            events.append("communicate")
            raise UnicodeDecodeError("utf-8", b"\xff", 0, 1, "invalid start byte")

        def poll(self) -> int:
            return self.returncode

        def kill(self) -> None:
            events.append("kill")

        def wait(self, *, timeout: float) -> int:
            del timeout
            events.append("wait")
            return self.returncode

    monkeypatch.setattr(subprocess, "Popen", lambda *args, **kwargs: FakeProcess())
    monkeypatch.setattr(os, "killpg", lambda pid, sig: events.append(f"killpg:{pid}:{sig}"))

    with pytest.raises(UnicodeDecodeError):
        assessment_execution.run_worker(
            tmp_path / "request.json",
            tmp_path / "output.json",
            timeout_seconds=1.0,
        )

    assert f"killpg:4102:{signal.SIGKILL}" in events
    assert "wait" in events
