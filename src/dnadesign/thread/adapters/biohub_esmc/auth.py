"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/thread/adapters/biohub_esmc/auth.py

Runtime-only Biohub credential loading.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

DEFAULT_KEY_LABEL = "bu-dunlop-lab"


@dataclass(frozen=True)
class BiohubCredential:
    """Biohub API credential with a non-secret label and hidden token."""

    key_label: str
    token: str = field(repr=False)

    @property
    def redacted_token(self) -> str:
        """Return a stable redaction marker for manifests and logs."""

        return "<redacted>"


def load_biohub_credential(path: Path, *, expected_label: str = DEFAULT_KEY_LABEL) -> BiohubCredential:
    """Load a two-line Biohub key file without printing or persisting the token."""

    key_path = path.expanduser().resolve()
    if not key_path.exists():
        raise FileNotFoundError(key_path)
    lines = key_path.read_text(encoding="utf-8").splitlines()
    if len(lines) < 2:
        raise ValueError("Biohub key file must contain a label line and a token line")
    key_label = lines[0].strip()
    token = lines[1].strip()
    if key_label != expected_label:
        raise ValueError(f"Biohub key label must be {expected_label!r}")
    if not token:
        raise ValueError("Biohub key token must be non-empty")
    return BiohubCredential(key_label=key_label, token=token)
