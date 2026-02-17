"""
--------------------------------------------------------------------------------
<dnadesign project>
src/dnadesign/opal/src/registries/loader.py

Centralizes registry discovery for built-ins and entry-point plugins. Raises
OpalError on import/load failures to enforce fail-fast behavior.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import importlib
import pkgutil
from typing import Callable, Iterable, Sequence

from ..core.utils import OpalError


def _format_error_list(errors: Sequence[str]) -> str:
    bullet = "\n  - "
    return f"{bullet}{bullet.join(errors)}"


def load_builtin_modules(
    package: str,
    *,
    label: str,
    debug: Callable[[str], None] | None = None,
) -> None:
    try:
        pkg = importlib.import_module(package)
    except Exception as exc:
        raise OpalError(f"Failed to import built-in {label} package '{package}': {exc}") from exc

    if debug is not None:
        debug(f"imported package: {pkg.__name__} ({getattr(pkg, '__file__', '?')})")

    try:
        pkg_path = pkg.__path__  # type: ignore[attr-defined]
    except Exception:
        pkg_path = []

    errors: list[str] = []
    for mod in pkgutil.iter_modules(pkg_path):
        if mod.name.startswith("_"):
            continue
        fq = f"{pkg.__name__}.{mod.name}"
        try:
            importlib.import_module(fq)
            if debug is not None:
                debug(f"imported built-in {label} module: {fq}")
        except Exception as exc:
            errors.append(f"{fq}: {exc}")
            if debug is not None:
                debug(f"FAILED importing {fq}: {exc!r}")

    if errors:
        raise OpalError(f"Failed to import built-in {label} modules:{_format_error_list(errors)}")


def _iter_entry_points(group: str) -> Iterable[object]:
    try:
        import importlib.metadata as _ilmd

        eps = _ilmd.entry_points()
        try:
            return list(eps.select(group=group))  # type: ignore[attr-defined]
        except Exception:
            return list(eps.get(group, []))  # type: ignore[index]
    except Exception:
        pass
    try:
        import pkg_resources

        return list(pkg_resources.iter_entry_points(group))
    except Exception as exc:
        raise OpalError(f"Failed to discover entry points for {group}: {exc}") from exc


def load_entry_points(
    group: str,
    *,
    label: str,
    debug: Callable[[str], None] | None = None,
) -> None:
    eps = _iter_entry_points(group)
    errors: list[str] = []
    for ep in eps:
        try:
            ep.load()
            if debug is not None:
                debug(f"loaded plugin entry point: {getattr(ep, 'name', '?')} from {getattr(ep, 'module', '?')}")
        except Exception as exc:
            errors.append(f"{ep!r}: {exc}")
            if debug is not None:
                debug(f"FAILED loading plugin entry point {ep!r}: {exc!r}")

    if errors:
        raise OpalError(f"Failed to load {label} plugins from entry points:{_format_error_list(errors)}")
