"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/thread/adapters/ligandmpnn/pinned_runtime.py

Execute attested LigandMPNN entrypoints with attested parser source.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path
from types import ModuleType

from dnadesign.thread.adapters.ligandmpnn.pinned_checkout import attested_working_tree_path_bytes

_ENTRYPOINTS = frozenset({"run.py", "score.py"})
_MODULE = "dnadesign.thread.adapters.ligandmpnn.pinned_runtime"


def pinned_runtime_prefix(
    *,
    checkout_root: Path,
    upstream_commit: str,
    entrypoint: str,
    python_executable: str,
) -> tuple[str, ...]:
    """Return the deterministic wrapper prefix for one official entrypoint."""

    if entrypoint not in _ENTRYPOINTS:
        raise ValueError(f"unsupported LigandMPNN entrypoint: {entrypoint!r}")
    return (
        python_executable,
        "-m",
        _MODULE,
        "--checkout-root",
        str(checkout_root),
        "--upstream-commit",
        upstream_commit,
        "--entrypoint",
        entrypoint,
        "--",
    )


def parse_pinned_runtime_prefix(
    argv: tuple[str, ...],
    *,
    upstream_commit: str,
    entrypoint: str,
) -> tuple[Path, str]:
    """Recover only the two caller-owned fields from an exact wrapper prefix."""

    if len(argv) < 10:
        raise ValueError("command does not use the pinned LigandMPNN runtime")
    expected = (
        argv[0],
        "-m",
        _MODULE,
        "--checkout-root",
        argv[4],
        "--upstream-commit",
        upstream_commit,
        "--entrypoint",
        entrypoint,
        "--",
    )
    if argv[:10] != expected:
        raise ValueError("command does not use the pinned LigandMPNN runtime")
    return Path(argv[4]), argv[0]


def execute_pinned_entrypoint(
    *,
    checkout_root: Path,
    upstream_commit: str,
    entrypoint: str,
    arguments: tuple[str, ...],
) -> None:
    """Execute only entrypoint and parser bytes attested to one Git commit."""

    if entrypoint not in _ENTRYPOINTS:
        raise ValueError(f"unsupported LigandMPNN entrypoint: {entrypoint!r}")
    checkout = checkout_root.expanduser().resolve()
    if not checkout.is_dir():
        raise ValueError("LigandMPNN checkout_root must be an existing directory")
    observed_commit = _git_head(checkout)
    if observed_commit != upstream_commit:
        raise ValueError(f"LigandMPNN checkout HEAD mismatch: expected {upstream_commit}, observed {observed_commit}")
    parser_bytes = _attested_regular_file(checkout, upstream_commit, "data_utils.py")
    entrypoint_bytes = _attested_regular_file(checkout, upstream_commit, entrypoint)
    parser_path = checkout / "data_utils.py"
    entrypoint_path = checkout / entrypoint

    parser_module = ModuleType("data_utils")
    parser_module.__file__ = str(parser_path)
    parser_module.__package__ = ""
    previous_parser = sys.modules.get("data_utils")
    previous_argv = sys.argv
    previous_sys_path = list(sys.path)
    sys.path.insert(0, str(checkout))
    try:
        sys.modules["data_utils"] = parser_module
        exec(compile(parser_bytes, str(parser_path), "exec"), parser_module.__dict__)
        sys.argv = [str(entrypoint_path), *arguments]
        entrypoint_globals = {
            "__builtins__": __builtins__,
            "__cached__": None,
            "__file__": str(entrypoint_path),
            "__name__": "__main__",
            "__package__": None,
        }
        exec(compile(entrypoint_bytes, str(entrypoint_path), "exec"), entrypoint_globals)
    finally:
        sys.argv = previous_argv
        sys.path[:] = previous_sys_path
        if previous_parser is None:
            sys.modules.pop("data_utils", None)
        else:
            sys.modules["data_utils"] = previous_parser


def _attested_regular_file(checkout: Path, commit: str, relative_path: str) -> bytes:
    path = checkout / relative_path
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"pinned LigandMPNN {relative_path} must be a regular file")
    source_bytes = attested_working_tree_path_bytes(checkout, commit, relative_path)
    if source_bytes is None:
        raise ValueError(f"{relative_path} must match the pinned commit")
    return source_bytes


def _git_head(checkout: Path) -> str:
    try:
        return subprocess.check_output(
            ["git", "--no-replace-objects", "-C", str(checkout), "rev-parse", "HEAD"],
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip()
    except (OSError, subprocess.CalledProcessError) as exc:
        raise ValueError("LigandMPNN checkout Git commit could not be read") from exc


def main() -> None:
    """Run one attested official entrypoint from a generated command."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkout-root", type=Path, required=True)
    parser.add_argument("--upstream-commit", required=True)
    parser.add_argument("--entrypoint", choices=sorted(_ENTRYPOINTS), required=True)
    parser.add_argument("arguments", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    arguments = tuple(args.arguments[1:] if args.arguments[:1] == ["--"] else args.arguments)
    execute_pinned_entrypoint(
        checkout_root=args.checkout_root,
        upstream_commit=args.upstream_commit,
        entrypoint=args.entrypoint,
        arguments=arguments,
    )


if __name__ == "__main__":
    main()
