"""
--------------------------------------------------------------------------------
<cruncher project>
src/dnadesign/cruncher/src/app/parse_signature.py

Compute deterministic parse signatures from lockfile + motif-store config.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import json

from dnadesign.cruncher.config.schema_v3 import CruncherConfig
from dnadesign.cruncher.store.lockfile import Lockfile, lockfile_fingerprint
from dnadesign.cruncher.utils.hashing import sha256_bytes


def compute_parse_signature(
    *,
    cfg: CruncherConfig,
    lockfile: Lockfile,
    tfs: list[str],
) -> tuple[str, dict[str, object]]:
    lock_sig, _ = lockfile_fingerprint(lockfile)
    payload = {
        "tfs": sorted(tfs),
        "lockfile_fingerprint": lock_sig,
        "motif_store": {
            "pwm_source": cfg.catalog.pwm_source,
            "combine_sites": cfg.catalog.combine_sites,
            "site_kinds": cfg.catalog.site_kinds,
            "site_window_lengths": cfg.catalog.site_window_lengths,
            "site_window_center": cfg.catalog.site_window_center,
            "min_sites_for_pwm": cfg.catalog.min_sites_for_pwm,
            "allow_low_sites": cfg.catalog.allow_low_sites,
            "pseudocounts": cfg.catalog.pseudocounts,
        },
    }
    signature = sha256_bytes(json.dumps(payload, sort_keys=True).encode("utf-8"))
    return signature, payload
