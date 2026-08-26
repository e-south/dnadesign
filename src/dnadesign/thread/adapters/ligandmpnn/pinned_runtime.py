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
import hashlib
import subprocess
import sys
import tempfile
from pathlib import Path, PurePosixPath

_ENTRYPOINTS = frozenset({"run.py", "score.py"})
_MODULE = "dnadesign.thread.adapters.ligandmpnn.pinned_runtime"
_CHECKPOINT_FLAG = "--checkpoint_ligand_mpnn"
_PACKING_CHECKPOINT_FLAG = "--checkpoint_path_sc"


def pinned_runtime_prefix(
    *,
    checkout_root: Path,
    upstream_commit: str,
    checkpoint_sha256: str,
    packing_checkpoint_sha256: str | None,
    entrypoint: str,
    python_executable: str,
) -> tuple[str, ...]:
    """Return the deterministic wrapper prefix for one official entrypoint."""

    if entrypoint not in _ENTRYPOINTS:
        raise ValueError(f"unsupported LigandMPNN entrypoint: {entrypoint!r}")
    prefix = [
        python_executable,
        "-m",
        _MODULE,
        "--checkout-root",
        str(checkout_root),
        "--upstream-commit",
        upstream_commit,
        "--checkpoint-sha256",
        checkpoint_sha256,
    ]
    if packing_checkpoint_sha256 is not None:
        prefix.extend(["--packing-checkpoint-sha256", packing_checkpoint_sha256])
    prefix.extend(
        [
            "--entrypoint",
            entrypoint,
            "--",
        ]
    )
    return tuple(prefix)


def parse_pinned_runtime_prefix(
    argv: tuple[str, ...],
    *,
    upstream_commit: str,
    checkpoint_sha256: str,
    packing_checkpoint_sha256: str | None,
    entrypoint: str,
) -> tuple[Path, str]:
    """Recover only the two caller-owned fields from an exact wrapper prefix."""

    if len(argv) < 12:
        raise ValueError("command does not use the pinned LigandMPNN runtime")
    expected = pinned_runtime_prefix(
        checkout_root=Path(argv[4]),
        upstream_commit=upstream_commit,
        checkpoint_sha256=checkpoint_sha256,
        packing_checkpoint_sha256=packing_checkpoint_sha256,
        entrypoint=entrypoint,
        python_executable=argv[0],
    )
    if argv[: len(expected)] != expected:
        raise ValueError("command does not use the pinned LigandMPNN runtime")
    return Path(argv[4]), argv[0]


def execute_pinned_entrypoint(
    *,
    checkout_root: Path,
    upstream_commit: str,
    checkpoint_sha256: str,
    packing_checkpoint_sha256: str | None,
    entrypoint: str,
    arguments: tuple[str, ...],
) -> None:
    """Execute one pinned source snapshot with digest-verified weight bytes."""

    if entrypoint not in _ENTRYPOINTS:
        raise ValueError(f"unsupported LigandMPNN entrypoint: {entrypoint!r}")
    checkout = checkout_root.expanduser().resolve()
    if not checkout.is_dir():
        raise ValueError("LigandMPNN checkout_root must be an existing directory")
    observed_commit = _git_head(checkout)
    if observed_commit != upstream_commit:
        raise ValueError(f"LigandMPNN checkout HEAD mismatch: expected {upstream_commit}, observed {observed_commit}")
    with tempfile.TemporaryDirectory(prefix="dnadesign-ligandmpnn-") as temporary:
        snapshot = Path(temporary) / "source"
        snapshot.mkdir()
        _materialize_pinned_tree(checkout, upstream_commit, snapshot)
        entrypoint_path = snapshot / entrypoint
        if not entrypoint_path.is_file():
            raise ValueError(f"pinned LigandMPNN commit does not contain {entrypoint}")
        runtime_arguments = list(arguments)
        weights_root = snapshot / ".dnadesign-weights"
        weights_root.mkdir()
        _replace_verified_checkpoint(
            runtime_arguments,
            flag=_CHECKPOINT_FLAG,
            expected_sha256=checkpoint_sha256,
            destination=weights_root / "ligandmpnn.pt",
        )
        if packing_checkpoint_sha256 is None:
            if _PACKING_CHECKPOINT_FLAG in runtime_arguments:
                raise ValueError("packing checkpoint was supplied without a pinned digest")
        else:
            _replace_verified_checkpoint(
                runtime_arguments,
                flag=_PACKING_CHECKPOINT_FLAG,
                expected_sha256=packing_checkpoint_sha256,
                destination=weights_root / "packing.pt",
            )
        subprocess.run(
            [sys.executable, "-B", "-E", "-s", str(entrypoint_path), *runtime_arguments],
            check=True,
        )


def _materialize_pinned_tree(checkout: Path, commit: str, destination: Path) -> None:
    try:
        tree = subprocess.check_output(
            ["git", "--no-replace-objects", "-C", str(checkout), "ls-tree", "-r", "-z", "--full-tree", commit],
            stderr=subprocess.DEVNULL,
        )
    except (OSError, subprocess.CalledProcessError) as exc:
        raise ValueError("LigandMPNN pinned source tree could not be read") from exc
    for record in tree.split(b"\0"):
        if not record:
            continue
        try:
            metadata, raw_path = record.split(b"\t", 1)
            mode, object_type, object_id = metadata.decode("ascii").split()
            relative_path = raw_path.decode("utf-8")
        except (UnicodeDecodeError, ValueError) as exc:
            raise ValueError("LigandMPNN pinned source tree contains an invalid entry") from exc
        path = PurePosixPath(relative_path)
        if path.is_absolute() or not path.parts or ".." in path.parts:
            raise ValueError("LigandMPNN pinned source tree contains an unsafe path")
        if object_type != "blob" or mode not in {"100644", "100755"}:
            raise ValueError(f"LigandMPNN pinned source tree contains unsupported entry: {relative_path}")
        if path.suffix == ".pyc" or "__pycache__" in path.parts:
            continue
        try:
            payload = subprocess.check_output(
                ["git", "--no-replace-objects", "-C", str(checkout), "cat-file", "blob", object_id],
                stderr=subprocess.DEVNULL,
            )
        except (OSError, subprocess.CalledProcessError) as exc:
            raise ValueError(f"LigandMPNN pinned blob could not be read: {relative_path}") from exc
        output_path = destination.joinpath(*path.parts)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_bytes(payload)
        output_path.chmod(0o755 if mode == "100755" else 0o644)


def _replace_verified_checkpoint(
    arguments: list[str],
    *,
    flag: str,
    expected_sha256: str,
    destination: Path,
) -> None:
    if len(expected_sha256) != 64 or any(character not in "0123456789abcdef" for character in expected_sha256):
        raise ValueError(f"{flag} expected digest must be a lowercase SHA256")
    positions = [index for index, value in enumerate(arguments) if value == flag]
    if len(positions) != 1 or positions[0] + 1 >= len(arguments):
        raise ValueError(f"runtime arguments must contain exactly one {flag}")
    value_index = positions[0] + 1
    checkpoint_path = Path(arguments[value_index]).expanduser()
    if checkpoint_path.is_symlink() or not checkpoint_path.is_file():
        raise ValueError(f"{flag} must reference a regular checkpoint file")
    try:
        payload = checkpoint_path.read_bytes()
    except OSError as exc:
        raise ValueError(f"{flag} checkpoint could not be read") from exc
    observed_sha256 = hashlib.sha256(payload).hexdigest()
    if observed_sha256 != expected_sha256:
        raise ValueError(
            f"{flag} SHA256 mismatch: expected sha256:{expected_sha256}, observed sha256:{observed_sha256}"
        )
    destination.write_bytes(payload)
    destination.chmod(0o400)
    arguments[value_index] = str(destination)


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
    parser.add_argument("--checkpoint-sha256", required=True)
    parser.add_argument("--packing-checkpoint-sha256")
    parser.add_argument("--entrypoint", choices=sorted(_ENTRYPOINTS), required=True)
    parser.add_argument("arguments", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    arguments = tuple(args.arguments[1:] if args.arguments[:1] == ["--"] else args.arguments)
    execute_pinned_entrypoint(
        checkout_root=args.checkout_root,
        upstream_commit=args.upstream_commit,
        checkpoint_sha256=args.checkpoint_sha256,
        packing_checkpoint_sha256=args.packing_checkpoint_sha256,
        entrypoint=args.entrypoint,
        arguments=arguments,
    )


if __name__ == "__main__":
    main()
