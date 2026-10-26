"""Resolve repository-relative README links for the installed package description."""

import re
import tomllib
from pathlib import Path

from setuptools import setup

ROOT = Path(__file__).parent
PROJECT = tomllib.loads((ROOT / "pyproject.toml").read_text())["project"]


def package_link(match):
    target = match.group(1)
    if target.startswith(("https://", "http://", "#", "mailto:")):
        return match.group(0)
    path = target.split("#", 1)[0]
    kind = "tree" if (ROOT / path).is_dir() else "blob"
    return f"](https://github.com/e-south/dnadesign/{kind}/v{PROJECT['version']}/{target})"


setup(
    long_description=re.sub(r"\]\(([^)]+)\)", package_link, (ROOT / "README.md").read_text()),
    long_description_content_type="text/markdown",
)
