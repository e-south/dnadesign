"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/notify/cli.py

Command-line notifier for dnadesign batch runs.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import typer

from .errors import NotifyError
from .http import post_json
from .payload import build_payload
from .providers import format_payload
from .validation import resolve_webhook_url

app = typer.Typer(help="Send notifications for dnadesign runs via webhooks.")


def _load_meta(meta_path: Path | None) -> dict[str, Any]:
    if meta_path is None:
        return {}
    if not meta_path.exists():
        raise NotifyError(f"meta file not found: {meta_path}")
    try:
        data = json.loads(meta_path.read_text())
    except json.JSONDecodeError as exc:
        raise NotifyError(f"meta file is not valid JSON: {meta_path}") from exc
    if not isinstance(data, dict):
        raise NotifyError("meta file must contain a JSON object")
    return data


@app.command("send")
def send(
    status: str = typer.Option(..., help="Status: success|failure|started|running."),
    tool: str = typer.Option(..., help="Tool name (densegen, infer, opal, etc.)."),
    run_id: str = typer.Option(..., help="Run identifier."),
    provider: str = typer.Option(..., help="Provider: generic|slack|discord."),
    url: str | None = typer.Option(None, help="Webhook URL."),
    url_env: str | None = typer.Option(None, help="Environment variable holding webhook URL."),
    message: str | None = typer.Option(None, help="Optional message."),
    meta: Path | None = typer.Option(None, help="Path to JSON metadata file."),
    timeout: float = typer.Option(10.0, help="HTTP timeout (seconds)."),
    retries: int = typer.Option(0, help="Number of retries on failure."),
    dry_run: bool = typer.Option(False, help="Print payload and exit without sending."),
) -> None:
    try:
        webhook_url = resolve_webhook_url(url=url, url_env=url_env)
        meta_data = _load_meta(meta)
        payload = build_payload(
            status=status,
            tool=tool,
            run_id=run_id,
            message=message,
            meta=meta_data,
        )
        formatted = format_payload(provider, payload)
        if dry_run:
            typer.echo(json.dumps(formatted, indent=2, sort_keys=True))
            return
        post_json(webhook_url, formatted, timeout=timeout, retries=retries)
        typer.echo("Notification sent.")
    except NotifyError as exc:
        typer.echo(f"Notification failed: {exc}")
        raise typer.Exit(code=1)


def main() -> None:
    app()


if __name__ == "__main__":
    main()
