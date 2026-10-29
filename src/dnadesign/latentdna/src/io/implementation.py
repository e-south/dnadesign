"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/latentdna/src/io/implementation.py

Content identity for the installed analysis implementation and numerical dependencies.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from functools import lru_cache
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

from .hashing import sha256_file, sha256_payload


@lru_cache(maxsize=1)
def execution_identity() -> dict[str, object]:
    package_root = Path(__file__).resolve().parents[3]
    sources = {}
    for relative in ("latentdna/src", "usr/src", "contracts"):
        for path in sorted((package_root / relative).rglob("*.py")):
            key = path.relative_to(package_root)
            if "tests" not in key.parts:
                sources[key.as_posix()] = sha256_file(path)
    dependencies = {}
    for name in ("numpy", "pyarrow", "scipy", "scikit-learn", "umap-learn", "pydantic"):
        try:
            dependencies[name] = version(name)
        except PackageNotFoundError:
            dependencies[name] = None
    return {"implementation_digest": sha256_payload(sources), "dependency_versions": dependencies}
