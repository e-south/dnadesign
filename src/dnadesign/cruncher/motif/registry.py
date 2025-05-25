"""
--------------------------------------------------------------------------------
<dnadesign project>
cruncher/motif/registry.py

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

        path = self._find_path(tf_key)
        pwm = load_pwm(path)
        self._cache[tf_key] = pwm
        return pwm

    # helpers
    def _find_path(self, tf_key: str) -> Path:
        for ext in self.ext_map:
            candidate = self.root / f"{tf_key}{ext}"
            if candidate.exists():
                return candidate
        # fallback: recursive search
        matches = list(self.root.rglob(f"{tf_key}.*"))
        if matches:
            return matches[0]
        raise FileNotFoundError(f"No PWM file found for TF '{tf_key}' under {self.root}")
