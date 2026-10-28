"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/devtools/tests/support/dependencies.py

Resolve local extras when checking published dependency contracts.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from packaging.requirements import Requirement


def profile_requirements(project: dict, profile: str) -> set[str]:
    requirements: set[str] = set()
    pending = [profile]
    visited: set[str] = set()
    while pending:
        current = pending.pop()
        if current in visited:
            continue
        visited.add(current)
        for value in project["optional-dependencies"][current]:
            requirement = Requirement(value)
            if requirement.name == project["name"]:
                pending.extend(requirement.extras)
            else:
                requirements.add(value)
    return requirements
