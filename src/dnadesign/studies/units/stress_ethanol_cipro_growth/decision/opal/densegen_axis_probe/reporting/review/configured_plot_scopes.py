"""Configured OPAL plot scope coverage checks."""

from __future__ import annotations

from typing import Any, Mapping


def _configured_spec_coverage_problems(
    plots: Any,
    expected_specs: Any,
    *,
    expected_final_round: Any,
) -> list[str]:
    if expected_final_round is None or not isinstance(expected_specs, list) or not expected_specs:
        return []
    final_round = int(expected_final_round)
    actual_scopes = _actual_plot_scopes(plots)
    expected_scopes: dict[tuple[str, str], set[str]] = {}
    for spec in expected_specs:
        if not isinstance(spec, Mapping) or spec.get("enabled") is False:
            continue
        name = str(spec.get("name") or "").strip()
        kind = str(spec.get("kind") or "").strip()
        if not name or not kind:
            continue
        expected_scopes[(name, kind)] = _expected_scope_keys(spec, final_round=final_round)

    problems: list[str] = []
    for (name, kind), expected in sorted(expected_scopes.items()):
        actual = actual_scopes.get((name, kind), set())
        missing = expected - actual
        if missing:
            problems.append(f"{name}:configured_plot_missing_scopes:{_format_scope_set(missing)}")

    for (name, kind), actual in sorted(actual_scopes.items()):
        expected = expected_scopes.get((name, kind))
        if expected is None:
            problems.append(f"{name}:configured_plot_not_declared:{kind}")
            continue
        extra = actual - expected
        if extra:
            problems.append(f"{name}:configured_plot_unexpected_scopes:{_format_scope_set(extra)}")
    return problems


def _round_scope_coverage_problems(plots: Any, *, expected_final_round: Any) -> list[str]:
    if expected_final_round is None:
        return []
    final_round = int(expected_final_round)
    grouped: dict[tuple[str, str], list[Mapping[str, Any]]] = {}
    for plot in plots:
        if not isinstance(plot, Mapping):
            continue
        key = (str(plot.get("name") or "unknown"), str(plot.get("kind") or "unknown"))
        grouped.setdefault(key, []).append(plot)

    problems: list[str] = []
    for (name, _kind), group in grouped.items():
        if not _expects_final_round_coverage(group):
            continue
        if any(_round_scope_covers_final(plot.get("rounds"), final_round=final_round) for plot in group):
            continue
        problems.append(f"{name}:round_scope_missing_final_round:{final_round}")
    return problems


def _actual_plot_scopes(plots: Any) -> dict[tuple[str, str], set[str]]:
    scopes: dict[tuple[str, str], set[str]] = {}
    for plot in plots:
        if not isinstance(plot, Mapping):
            continue
        name = str(plot.get("name") or "").strip()
        kind = str(plot.get("kind") or "").strip()
        if not name or not kind:
            continue
        scopes.setdefault((name, kind), set()).add(_scope_key(plot.get("rounds")))
    return scopes


def _expected_scope_keys(spec: Mapping[str, Any], *, final_round: int) -> set[str]:
    base_scope = _selector_scope_key(spec.get("round_selector"))
    variants = _round_variant_tokens(spec.get("round_variants"))
    if not variants:
        return {base_scope}
    scopes: set[str] = set()
    for variant in variants:
        if variant in {"configured", "default", "base"}:
            scopes.add(base_scope)
        elif variant == "each":
            scopes.update(f"r{round_index}" for round_index in range(final_round + 1))
        else:
            scopes.add(_selector_scope_key(variant))
    return scopes


def _round_variant_tokens(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, bool):
        return ["configured", "each"] if value else []
    if isinstance(value, str):
        return [value.strip().lower()]
    if isinstance(value, list):
        return [str(item).strip().lower() for item in value if str(item).strip()]
    return [str(value).strip().lower()]


def _selector_scope_key(value: Any) -> str:
    if value is None:
        return "unspecified"
    text = str(value).strip().lower()
    if text in {"", "unspecified"}:
        return "unspecified"
    if text in {"all", "latest"}:
        return text
    if text.isdigit():
        return f"r{int(text)}"
    return text


def _scope_key(rounds: Any) -> str:
    if isinstance(rounds, str):
        return _selector_scope_key(rounds)
    if isinstance(rounds, list):
        values = [int(value) for value in rounds if value is not None]
        if len(values) == 1:
            return f"r{values[0]}"
        return "r" + ",".join(str(value) for value in sorted(values))
    return _selector_scope_key(rounds)


def _format_scope_set(scopes: set[str]) -> str:
    return ",".join(sorted(scopes, key=_scope_sort_key))


def _scope_sort_key(scope: str) -> tuple[int, int | str]:
    if scope == "all":
        return (0, scope)
    if scope == "latest":
        return (1, scope)
    if scope.startswith("r") and scope[1:].isdigit():
        return (2, int(scope[1:]))
    return (3, scope)


def _expects_final_round_coverage(group: list[Mapping[str, Any]]) -> bool:
    for plot in group:
        rounds = plot.get("rounds")
        metadata = plot.get("metadata") if isinstance(plot.get("metadata"), Mapping) else {}
        capability = metadata.get("capability") if isinstance(metadata.get("capability"), Mapping) else {}
        round_scope = str(capability.get("round_scope") or "")
        name = str(plot.get("name") or "").lower()
        if name.endswith("_latest") or round_scope in {"single_round", "single_or_round_history"}:
            return True
        if isinstance(rounds, list):
            return True
    return False


def _round_scope_covers_final(rounds: Any, *, final_round: int) -> bool:
    if isinstance(rounds, str) and rounds in {"all", "latest"}:
        return True
    if isinstance(rounds, list):
        return final_round in {int(value) for value in rounds if value is not None}
    return False
