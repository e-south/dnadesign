"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/notify/http.py

HTTP delivery helpers for notifier webhooks.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import json
import time
import urllib.error
import urllib.request
from typing import Any

from .errors import NotifyDeliveryError


def post_json(
    url: str,
    payload: dict[str, Any],
    *,
    timeout: float = 10.0,
    retries: int = 0,
    user_agent: str = "dnadesign-notify",
) -> None:
    body = json.dumps(payload).encode("utf-8")
    headers = {
        "Content-Type": "application/json",
        "User-Agent": user_agent,
    }
    attempt = 0
    while True:
        attempt += 1
        req = urllib.request.Request(url, data=body, headers=headers, method="POST")
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                status = int(resp.getcode() or 0)
                if status < 200 or status >= 300:
                    raise NotifyDeliveryError(f"Webhook returned status {status}")
                return
        except (urllib.error.URLError, NotifyDeliveryError) as exc:
            if attempt > retries:
                raise NotifyDeliveryError(f"Webhook delivery failed: {exc}") from exc
            time.sleep(2 ** (attempt - 1))
