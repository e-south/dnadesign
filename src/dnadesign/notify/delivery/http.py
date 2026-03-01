"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/notify/delivery/http.py

HTTP delivery helpers for notifier webhooks.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import json
import ssl
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

from ..errors import NotifyDeliveryError


def post_json(
    url: str,
    payload: dict[str, Any],
    *,
    timeout: float = 10.0,
    retries: int = 0,
    user_agent: str = "dnadesign-notify",
    tls_ca_bundle: Path | None = None,
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
            scheme = urlparse(url).scheme.lower()
            if scheme == "https":
                if tls_ca_bundle is None:
                    raise NotifyDeliveryError(
                        "HTTPS webhook delivery requires a CA bundle. Pass --tls-ca-bundle or set SSL_CERT_FILE."
                    )
                context = ssl.create_default_context(cafile=str(tls_ca_bundle))
                response = urllib.request.urlopen(req, timeout=timeout, context=context)
            else:
                response = urllib.request.urlopen(req, timeout=timeout)
            with response as resp:
                status = int(resp.getcode() or 0)
                if status < 200 or status >= 300:
                    raise NotifyDeliveryError(f"Webhook returned status {status}")
                return
        except (urllib.error.URLError, NotifyDeliveryError) as exc:
            if attempt > retries:
                raise NotifyDeliveryError(f"Webhook delivery failed: {exc}") from exc
            time.sleep(2 ** (attempt - 1))
