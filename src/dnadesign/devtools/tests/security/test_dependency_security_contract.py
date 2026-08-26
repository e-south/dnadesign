"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/devtools/tests/security/test_dependency_security_contract.py

Security contract tests for dependency floors and safe artifact loading.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import ast
import pickle
import subprocess
import tomllib
from pathlib import Path

import pytest
import yaml


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[5]


def _pyproject() -> dict[str, object]:
    with (_repo_root() / "pyproject.toml").open("rb") as handle:
        return tomllib.load(handle)


def _repository_source_files(
    repo_root: Path,
    *,
    roots: tuple[str, ...],
    suffixes: frozenset[str],
) -> tuple[Path, ...]:
    completed = subprocess.run(
        [
            "git",
            "-C",
            str(repo_root),
            "ls-files",
            "-z",
            "--cached",
            "--others",
            "--exclude-standard",
            "--",
            *roots,
        ],
        check=False,
        capture_output=True,
    )
    if completed.returncode != 0:
        detail = completed.stderr.decode("utf-8", errors="replace").strip()
        raise RuntimeError(f"git ls-files failed: {detail or 'unknown error'}")

    sources: list[Path] = []
    for raw_path in completed.stdout.split(b"\0"):
        if not raw_path:
            continue
        relative_path = Path(raw_path.decode("utf-8"))
        if relative_path.suffix in suffixes and (repo_root / relative_path).is_file():
            sources.append(relative_path)
    return tuple(sorted(sources))


def test_security_floors_are_published_by_their_owning_dependency_sets() -> None:
    project = _pyproject()["project"]
    dependencies = tuple(project["dependencies"])
    evo2_dependencies = tuple(project["optional-dependencies"]["infer-evo2"])
    build_dependencies = tuple(_pyproject()["build-system"]["requires"])
    constraint_dependencies = tuple(_pyproject()["tool"]["uv"]["constraint-dependencies"])

    assert "click>=8.3.3" in dependencies
    assert "marimo>=0.23.16" in dependencies
    assert "pillow>=12.3.0" in dependencies
    assert "zstd>=1.5.7.2,!=1.5.7.3" in dependencies
    assert "idna>=3.15" in constraint_dependencies
    assert "pymdown-extensions>=11.0.1" in constraint_dependencies
    assert any(requirement.startswith("onnx>=1.22.0;") for requirement in evo2_dependencies)
    assert any(requirement.startswith("pip>=26.1.2;") for requirement in evo2_dependencies)
    assert any(requirement.startswith("setuptools>=84.0.0;") for requirement in evo2_dependencies)
    assert any(requirement.startswith("wheel>=0.48.0;") for requirement in evo2_dependencies)
    assert any(requirement.startswith("torch>=2.13,<2.14;") for requirement in dependencies)
    assert any(requirement.startswith("torch>=2.13,<2.14;") for requirement in evo2_dependencies)
    assert all(not requirement.startswith(("torchaudio", "torchvision")) for requirement in evo2_dependencies)
    assert "setuptools>=84.0.0" in build_dependencies
    assert "wheel>=0.48.0" in build_dependencies
    assert all("email" not in author for author in project["authors"])


def test_dependabot_covers_each_tracked_dependency_surface() -> None:
    config = yaml.safe_load((_repo_root() / ".github" / "dependabot.yml").read_text(encoding="utf-8"))

    assert config["version"] == 2
    updates = {entry["package-ecosystem"]: entry for entry in config["updates"]}
    assert set(updates) == {"uv", "github-actions", "pre-commit"}
    for entry in updates.values():
        assert entry["directory"] == "/"
        assert entry["schedule"] == {"interval": "weekly"}
        assert entry["cooldown"] == {"default-days": 7}
        assert entry["open-pull-requests-limit"] == 5
        assert tuple(entry["groups"].values())[0]["patterns"] == ["*"]


def test_unsupported_torch_execution_does_not_enter_supported_code() -> None:
    repo_root = _repo_root()
    forbidden = ("torch.jit." + "script",)
    violations: list[str] = []

    source_files = _repository_source_files(
        repo_root,
        roots=(".github", "src"),
        suffixes=frozenset({".json", ".py", ".toml", ".yaml", ".yml"}),
    )
    for relative_path in source_files:
        text = (repo_root / relative_path).read_text(encoding="utf-8")
        for symbol in forbidden:
            if symbol in text:
                violations.append(f"{relative_path}: {symbol}")

    assert violations == []


def _torch_aliases(tree: ast.Module) -> tuple[dict[str, str], dict[str, str]]:
    module_aliases = {"torch": "torch"}
    direct_aliases: dict[str, str] = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for imported in node.names:
                if imported.name == "torch" or imported.name.startswith("torch."):
                    bound_name = imported.asname or imported.name.split(".", maxsplit=1)[0]
                    module_aliases[bound_name] = imported.name if imported.asname else "torch"
        elif isinstance(node, ast.ImportFrom) and node.module:
            if node.module == "torch":
                for imported in node.names:
                    direct_aliases[imported.asname or imported.name] = f"torch.{imported.name}"
            elif node.module in {"torch.export", "torch.jit"}:
                for imported in node.names:
                    direct_aliases[imported.asname or imported.name] = f"{node.module}.{imported.name}"

    changed = True
    while changed:
        changed = False
        for node in ast.walk(tree):
            target: ast.expr | None = None
            value: ast.expr | None = None
            if isinstance(node, ast.Assign) and len(node.targets) == 1:
                target, value = node.targets[0], node.value
            elif isinstance(node, ast.AnnAssign):
                target, value = node.target, node.value
            if not isinstance(target, ast.Name) or value is None:
                continue
            resolved = _torch_expr_name(value, module_aliases=module_aliases, direct_aliases=direct_aliases)
            if resolved is not None and direct_aliases.get(target.id) != resolved:
                direct_aliases[target.id] = resolved
                changed = True
    return module_aliases, direct_aliases


def _torch_expr_name(
    expression: ast.expr,
    *,
    module_aliases: dict[str, str],
    direct_aliases: dict[str, str],
) -> str | None:
    if isinstance(expression, ast.Name):
        return direct_aliases.get(expression.id) or module_aliases.get(expression.id)
    if isinstance(expression, ast.Call):
        if (
            isinstance(expression.func, ast.Name)
            and expression.func.id == "getattr"
            and len(expression.args) >= 2
            and isinstance(expression.args[1], ast.Constant)
            and isinstance(expression.args[1].value, str)
        ):
            owner = _torch_expr_name(
                expression.args[0],
                module_aliases=module_aliases,
                direct_aliases=direct_aliases,
            )
            if owner is not None:
                return f"{owner}.{expression.args[1].value}"
        return None
    if not isinstance(expression, ast.Attribute):
        return None

    parts: list[str] = []
    current: ast.expr = expression
    while isinstance(current, ast.Attribute):
        parts.append(current.attr)
        current = current.value
    if not isinstance(current, ast.Name):
        return None
    module_prefix = module_aliases.get(current.id)
    if module_prefix:
        return ".".join((module_prefix, *reversed(parts)))
    prefix = direct_aliases.get(current.id)
    if prefix:
        return ".".join((prefix, *reversed(parts)))
    return None


def _torch_boundary_violations(source: str, *, filename: str) -> list[str]:
    tree = ast.parse(source, filename=filename)
    module_aliases, direct_aliases = _torch_aliases(tree)
    violations: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Constant) and isinstance(node.value, str) and ".pt2" in node.value.casefold():
            violations.append(f"{filename}:{node.lineno}: unsupported PT2 artifact")
        if not isinstance(node, ast.Call):
            continue
        call_name = _torch_expr_name(node.func, module_aliases=module_aliases, direct_aliases=direct_aliases)
        if call_name == "torch.load":
            keyword = next((item for item in node.keywords if item.arg == "weights_only"), None)
            if keyword is None or not (isinstance(keyword.value, ast.Constant) and keyword.value.value is True):
                violations.append(f"{filename}:{node.lineno}: torch.load must set weights_only=True")
        elif call_name in {"torch." + "export.load", "torch." + "jit.script"}:
            violations.append(f"{filename}:{node.lineno}: forbidden {call_name}")
    return violations


def test_torch_deserialization_and_execution_boundaries_fail_closed() -> None:
    repo_root = _repo_root()
    violations: list[str] = []
    source_files = _repository_source_files(
        repo_root,
        roots=("src/dnadesign",),
        suffixes=frozenset({".py"}),
    )
    for relative_path in source_files:
        if "tests" in relative_path.parts:
            continue
        path = repo_root / relative_path
        violations.extend(
            _torch_boundary_violations(
                path.read_text(encoding="utf-8"),
                filename=relative_path.as_posix(),
            )
        )

    assert violations == []


def test_torch_boundary_scope_includes_new_source_and_excludes_ignored_archives(tmp_path: Path) -> None:
    subprocess.run(["git", "init", "-q", str(tmp_path)], check=True)
    tracked = tmp_path / "src" / "dnadesign" / "live.py"
    tracked.parent.mkdir(parents=True)
    tracked.write_text("import torch\ntorch.load('safe.pt', weights_only=True)\n", encoding="utf-8")
    subprocess.run(["git", "-C", str(tmp_path), "add", tracked.relative_to(tmp_path)], check=True)

    candidate = tmp_path / "src" / "dnadesign" / "candidate.py"
    candidate.write_text("import torch\ntorch.load('unsafe.pt')\n", encoding="utf-8")

    (tmp_path / ".gitignore").write_text("src/dnadesign/archived/\n", encoding="utf-8")
    ignored = tmp_path / "src" / "dnadesign" / "archived" / "legacy.py"
    ignored.parent.mkdir(parents=True)
    ignored.write_text("import torch\ntorch.load('unsafe.pt')\n", encoding="utf-8")

    assert _repository_source_files(
        tmp_path,
        roots=("src/dnadesign",),
        suffixes=frozenset({".py"}),
    ) == (
        Path("src/dnadesign/candidate.py"),
        Path("src/dnadesign/live.py"),
    )


@pytest.mark.parametrize(
    "source",
    (
        "import torch\nloader = torch.load\nloader('unsafe.pt')\n",
        "import torch\nloader = getattr(torch, 'load')\nloader('unsafe.pt')\n",
        "import torch\ngetattr(torch, 'load')('unsafe.pt')\n",
        "from torch import load as loader\nloader('unsafe.pt')\n",
        "import torch\npath = 'model.pt2'\n",
    ),
)
def test_torch_boundary_guard_rejects_indirect_or_pt2_surfaces(source: str) -> None:
    assert _torch_boundary_violations(source, filename="fixture.py")


def test_torch_boundary_guard_accepts_restricted_alias_load() -> None:
    source = "import torch\nloader = torch.load\nloader('safe.pt', weights_only=True)\n"

    assert _torch_boundary_violations(source, filename="fixture.py") == []


class _UnsupportedCheckpointValue:
    pass


def test_restricted_torch_load_rejects_unsupported_globals(tmp_path: Path) -> None:
    import torch

    path = tmp_path / "unsupported.pt"
    torch.save([{"sequence": "ACGT", "value": _UnsupportedCheckpointValue()}], path)

    with pytest.raises((pickle.UnpicklingError, ValueError), match="Weights only load failed"):
        torch.load(path, map_location="cpu", weights_only=True)


def test_restricted_torch_load_preserves_plain_tensor_and_dict_payloads(tmp_path: Path) -> None:
    import torch

    path = tmp_path / "plain.pt"
    payload = [{"sequence": "ACGT", "embedding": torch.tensor([1.0, 2.0])}]
    torch.save(payload, path)

    loaded = torch.load(path, map_location="cpu", weights_only=True)

    assert loaded[0]["sequence"] == "ACGT"
    assert loaded[0]["embedding"].tolist() == [1.0, 2.0]
