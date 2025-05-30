"""
--------------------------------------------------------------------------------
<dnadesign project>
dnadesign/cruncher/parse/registry.py

Lookup helper that (1) lower-cases TF names, (2) caches PWMs,
(3) searches allowed extensions according to config.

Module Author(s): Eric J. South
Dunlop Lab
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict

from .backend import load_pwm
from .model import PWM


class Registry:
    def __init__(self, root: Path, ext_map: Dict[str, str]):
        if not root.exists():
            raise FileNotFoundError(f"Motif root directory '{root}' does not exist")
        self.root = root
        self.ext_map = {k.lower(): v for k, v in ext_map.items()}
        self._cache: Dict[str, PWM] = {}

    # public
    def load(self, tf: str) -> PWM:
        tf_key = tf.lower().strip()
        if tf_key in self._cache:
            return self._cache[tf_key]

        path, fmtname = self._find_path(tf_key)
        pwm = load_pwm(path, fmt=fmtname)
        self._cache[tf_key] = pwm
        return pwm

    # helpers
    def _find_path(self, tf_key: str) -> tuple[Path, str]:
        # exact‐match using your formats map
        for ext, fmtname in self.ext_map.items():
            candidate = self.root / f"{tf_key}{ext}"
            if candidate.exists():
                return candidate, fmtname

        # fallback: case‐insensitive stem match on any extension
        for path in self.root.rglob("*"):
            if path.is_file() and path.stem.lower() == tf_key:
                fmtname = self.ext_map.get(path.suffix.lower(), path.suffix.lstrip(".").upper())
                return path, fmtname

        raise FileNotFoundError(f"No PWM file found for TF '{tf_key}' under {self.root}")
