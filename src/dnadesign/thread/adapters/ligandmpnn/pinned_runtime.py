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
from pathlib import Path

from dnadesign.thread.adapters.ligandmpnn.pinned_checkout import materialize_pinned_tree

_ENTRYPOINTS = frozenset({"run.py", "score.py"})
_MODULE = "dnadesign.thread.adapters.ligandmpnn.pinned_runtime"
_CHECKPOINT_FLAG = "--checkpoint_ligand_mpnn"
_PACKING_CHECKPOINT_FLAG = "--checkpoint_path_sc"
_PDB_FLAG = "--pdb_path"
_RESIDUE_ALPHABET_FLAG = "--omit_AA_per_residue"
_MODEL_TYPE_FLAG = "--model_type"
_OUTPUT_FOLDER_FLAG = "--out_folder"
_ALTERNATE_SOURCE_FLAGS = frozenset(
    {
        "--checkpoint_protein_mpnn",
        "--checkpoint_per_residue_label_membrane_mpnn",
        "--checkpoint_global_label_membrane_mpnn",
        "--checkpoint_soluble_mpnn",
        "--pdb_path_multi",
        "--fixed_residues_multi",
        "--redesigned_residues_multi",
        "--bias_AA_per_residue",
        "--bias_AA_per_residue_multi",
        "--omit_AA_per_residue_multi",
    }
)
_ATTESTATION_SENSITIVE_FLAGS = frozenset(
    {
        _MODEL_TYPE_FLAG,
        _CHECKPOINT_FLAG,
        _PACKING_CHECKPOINT_FLAG,
        _PDB_FLAG,
        _RESIDUE_ALPHABET_FLAG,
        _OUTPUT_FOLDER_FLAG,
        *_ALTERNATE_SOURCE_FLAGS,
    }
)


def pinned_runtime_prefix(
    *,
    checkout_root: Path,
    upstream_commit: str,
    checkpoint_sha256: str,
    pdb_sha256: str,
    packing_checkpoint_sha256: str | None,
    residue_alphabet_sha256: str | None,
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
        "--pdb-sha256",
        pdb_sha256,
    ]
    if packing_checkpoint_sha256 is not None:
        prefix.extend(["--packing-checkpoint-sha256", packing_checkpoint_sha256])
    if residue_alphabet_sha256 is not None:
        prefix.extend(["--residue-alphabet-sha256", residue_alphabet_sha256])
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
    pdb_sha256: str,
    packing_checkpoint_sha256: str | None,
    residue_alphabet_sha256: str | None,
    entrypoint: str,
) -> tuple[Path, str]:
    """Recover only the two caller-owned fields from an exact wrapper prefix."""

    if len(argv) < 14:
        raise ValueError("command does not use the pinned LigandMPNN runtime")
    expected = pinned_runtime_prefix(
        checkout_root=Path(argv[4]),
        upstream_commit=upstream_commit,
        checkpoint_sha256=checkpoint_sha256,
        pdb_sha256=pdb_sha256,
        packing_checkpoint_sha256=packing_checkpoint_sha256,
        residue_alphabet_sha256=residue_alphabet_sha256,
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
    pdb_sha256: str,
    packing_checkpoint_sha256: str | None,
    residue_alphabet_sha256: str | None,
    entrypoint: str,
    arguments: tuple[str, ...],
) -> None:
    """Execute one pinned source snapshot with digest-verified weight bytes."""

    if entrypoint not in _ENTRYPOINTS:
        raise ValueError(f"unsupported LigandMPNN entrypoint: {entrypoint!r}")
    _validate_runtime_option_contract(arguments)
    checkout = checkout_root.expanduser().resolve()
    if not checkout.is_dir():
        raise ValueError("LigandMPNN checkout_root must be an existing directory")
    observed_commit = _git_head(checkout)
    if observed_commit != upstream_commit:
        raise ValueError(f"LigandMPNN checkout HEAD mismatch: expected {upstream_commit}, observed {observed_commit}")
    with tempfile.TemporaryDirectory(prefix="dnadesign-ligandmpnn-") as temporary:
        snapshot = Path(temporary) / "source"
        snapshot.mkdir()
        materialize_pinned_tree(checkout, upstream_commit, snapshot)
        entrypoint_path = snapshot / entrypoint
        if not entrypoint_path.is_file():
            raise ValueError(f"pinned LigandMPNN commit does not contain {entrypoint}")
        runtime_arguments = list(arguments)
        weights_root = snapshot / ".dnadesign-weights"
        weights_root.mkdir()
        _replace_verified_file(
            runtime_arguments,
            flag=_CHECKPOINT_FLAG,
            expected_sha256=checkpoint_sha256,
            destination=weights_root / "ligandmpnn.pt",
        )
        if packing_checkpoint_sha256 is None:
            if _has_flag(runtime_arguments, _PACKING_CHECKPOINT_FLAG):
                raise ValueError("packing checkpoint was supplied without a pinned digest")
        else:
            _replace_verified_file(
                runtime_arguments,
                flag=_PACKING_CHECKPOINT_FLAG,
                expected_sha256=packing_checkpoint_sha256,
                destination=weights_root / "packing.pt",
            )
        inputs_root = snapshot / ".dnadesign-inputs"
        inputs_root.mkdir()
        staged_pdb = _replace_verified_file(
            runtime_arguments,
            flag=_PDB_FLAG,
            expected_sha256=pdb_sha256,
            destination=inputs_root / "pdb",
            preserve_source_name=True,
        )
        if residue_alphabet_sha256 is None:
            if _has_flag(runtime_arguments, _RESIDUE_ALPHABET_FLAG):
                raise ValueError("residue alphabet sidecar was supplied without a pinned digest")
        else:
            _replace_verified_file(
                runtime_arguments,
                flag=_RESIDUE_ALPHABET_FLAG,
                expected_sha256=residue_alphabet_sha256,
                destination=inputs_root / "residue-alphabet.json",
            )
        if entrypoint == "score.py":
            _reject_existing_score_output(runtime_arguments, pdb_path=staged_pdb)
        subprocess.run(
            [sys.executable, "-B", "-E", "-s", str(entrypoint_path), *runtime_arguments],
            check=True,
        )


def _replace_verified_file(
    arguments: list[str],
    *,
    flag: str,
    expected_sha256: str,
    destination: Path,
    preserve_source_name: bool = False,
) -> Path:
    if len(expected_sha256) != 64 or any(character not in "0123456789abcdef" for character in expected_sha256):
        raise ValueError(f"{flag} expected digest must be a lowercase SHA256")
    attached_prefix = f"{flag}="
    if any(value.startswith(attached_prefix) for value in arguments):
        raise ValueError(f"runtime arguments must use the split form of {flag} exactly once")
    positions = [index for index, value in enumerate(arguments) if value == flag]
    if len(positions) != 1 or positions[0] + 1 >= len(arguments):
        raise ValueError(f"runtime arguments must contain exactly one {flag}")
    value_index = positions[0] + 1
    source_path = Path(arguments[value_index]).expanduser()
    if source_path.is_symlink() or not source_path.is_file():
        raise ValueError(f"{flag} must reference a regular file")
    try:
        payload = source_path.read_bytes()
    except OSError as exc:
        raise ValueError(f"{flag} file could not be read") from exc
    observed_sha256 = hashlib.sha256(payload).hexdigest()
    if observed_sha256 != expected_sha256:
        raise ValueError(
            f"{flag} SHA256 mismatch: expected sha256:{expected_sha256}, observed sha256:{observed_sha256}"
        )
    staged_path = destination / source_path.name if preserve_source_name else destination
    staged_path.parent.mkdir(parents=True, exist_ok=True)
    staged_path.write_bytes(payload)
    staged_path.chmod(0o400)
    arguments[value_index] = str(staged_path)
    return staged_path


def _validate_runtime_option_contract(arguments: tuple[str, ...]) -> None:
    option_names = tuple(value.partition("=")[0] for value in arguments if value.startswith("--"))
    for option_name in option_names:
        if option_name in _ALTERNATE_SOURCE_FLAGS or (
            option_name not in _ATTESTATION_SENSITIVE_FLAGS
            and any(protected.startswith(option_name) for protected in _ATTESTATION_SENSITIVE_FLAGS)
        ):
            raise ValueError(f"unattested or ambiguous LigandMPNN runtime option: {option_name}")
    model_positions = [index for index, value in enumerate(arguments) if value == _MODEL_TYPE_FLAG]
    if (
        option_names.count(_MODEL_TYPE_FLAG) != 1
        or len(model_positions) != 1
        or model_positions[0] + 1 >= len(arguments)
        or arguments[model_positions[0] + 1] != "ligand_mpnn"
    ):
        raise ValueError(f"unattested or ambiguous LigandMPNN runtime option: {_MODEL_TYPE_FLAG}")


def _reject_existing_score_output(arguments: list[str], *, pdb_path: Path) -> None:
    output_flag = "--out_folder"
    attached_prefix = f"{output_flag}="
    if any(value.startswith(attached_prefix) for value in arguments):
        raise ValueError(f"runtime arguments must use the split form of {output_flag} exactly once")
    positions = [index for index, value in enumerate(arguments) if value == output_flag]
    if len(positions) != 1 or positions[0] + 1 >= len(arguments):
        raise ValueError(f"runtime arguments must contain exactly one {output_flag}")
    output_root = Path(arguments[positions[0] + 1]).expanduser()
    expected_output = output_root / f"{pdb_path.stem}.pt"
    if expected_output.exists() or expected_output.is_symlink():
        raise ValueError(f"score output already exists; refuse stale or ambiguous result: {expected_output}")


def _has_flag(arguments: list[str], flag: str) -> bool:
    attached_prefix = f"{flag}="
    return any(value == flag or value.startswith(attached_prefix) for value in arguments)


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
    parser.add_argument("--pdb-sha256", required=True)
    parser.add_argument("--packing-checkpoint-sha256")
    parser.add_argument("--residue-alphabet-sha256")
    parser.add_argument("--entrypoint", choices=sorted(_ENTRYPOINTS), required=True)
    parser.add_argument("arguments", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    arguments = tuple(args.arguments[1:] if args.arguments[:1] == ["--"] else args.arguments)
    execute_pinned_entrypoint(
        checkout_root=args.checkout_root,
        upstream_commit=args.upstream_commit,
        checkpoint_sha256=args.checkpoint_sha256,
        pdb_sha256=args.pdb_sha256,
        packing_checkpoint_sha256=args.packing_checkpoint_sha256,
        residue_alphabet_sha256=args.residue_alphabet_sha256,
        entrypoint=args.entrypoint,
        arguments=arguments,
    )


if __name__ == "__main__":
    main()
