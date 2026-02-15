"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/usr/src/cli_merge_policy.py

Registry-backed duplicate merge-policy parsing for the USR CLI.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from .errors import SequencesError
from .merge_datasets import MergePolicy

_MERGE_POLICIES: dict[str, MergePolicy] = {}


def _normalize_policy_name(name: str | None) -> str:
    text = str(name or "").strip().lower()
    if not text:
        raise SequencesError("duplicate policy name must be a non-empty string")
    return text


def supported_merge_policies() -> tuple[str, ...]:
    return tuple(sorted(_MERGE_POLICIES))


def register_merge_policy(name: str, policy: MergePolicy) -> None:
    policy_name = _normalize_policy_name(name)
    if policy_name in _MERGE_POLICIES:
        raise SequencesError(f"duplicate policy name '{policy_name}' is already registered")
    if not isinstance(policy, MergePolicy):
        raise SequencesError("policy value must be a MergePolicy enum value")
    _MERGE_POLICIES[policy_name] = policy


def resolve_merge_policy(name: str) -> MergePolicy:
    policy_name = _normalize_policy_name(name)
    policy = _MERGE_POLICIES.get(policy_name)
    if policy is None:
        allowed = ", ".join(supported_merge_policies())
        raise SequencesError(f"Unsupported duplicate policy '{name}'. Supported values: {allowed}")
    return policy


register_merge_policy("error", MergePolicy.ERROR)
register_merge_policy("skip", MergePolicy.SKIP)
register_merge_policy("prefer-src", MergePolicy.PREFER_SRC)
register_merge_policy("prefer-dest", MergePolicy.PREFER_DEST)
