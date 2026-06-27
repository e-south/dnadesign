#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import urllib.parse
import urllib.request
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


def _sha256(path: Path) -> str | None:
    if not path.exists():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _yaml_scalar(value: object) -> str:
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, int | float):
        return str(value)
    text = str(value).replace("\\", "\\\\").replace('"', '\\"')
    return f'"{text}"'


def _read_simple_manifest(path: Path) -> dict[str, str]:
    values: dict[str, str] = {}
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or ":" not in line:
            continue
        key, value = line.split(":", 1)
        values[key.strip()] = value.strip().strip('"')
    return values


def _resolve_manifest_defaults(args: argparse.Namespace) -> None:
    if not args.session_manifest:
        return
    values = _read_simple_manifest(args.session_manifest)
    if args.port is None and values.get("port"):
        args.port = int(values["port"])
    if not args.chimerax_executable and values.get("chimerax_executable"):
        args.chimerax_executable = values["chimerax_executable"]
    if args.structure_path is None and values.get("source_structure_path"):
        args.structure_path = Path(values["source_structure_path"])
    if args.opened_model_id is None and values.get("opened_model_id"):
        args.opened_model_id = values["opened_model_id"]


def _send(*, port: int, command: str, timeout_seconds: float) -> dict[str, Any]:
    query = urllib.parse.urlencode({"command": command})
    url = f"http://127.0.0.1:{port}/run?{query}"
    with urllib.request.urlopen(url, timeout=timeout_seconds) as response:  # noqa: S310
        payload = response.read().decode("utf-8")
    parsed = json.loads(payload)
    if parsed.get("error") is not None:
        raise RuntimeError(f"ChimeraX returned error for {command!r}: {parsed['error']}")
    return parsed


def _write_manifest(
    *,
    manifest_path: Path,
    status: str,
    failure_reason: str | None,
    pose_id: str,
    captured_at_utc: str,
    chimerax_executable: str,
    structure_path: Path | None,
    source_url: str | None,
    opened_model_id: str | None,
    preopened_session: bool,
    port: int,
    rest_stopped: bool,
    camera_mode: str,
    background_color: str,
    title: str | None,
    session_path: Path,
    image_path: Path,
    command_log_path: Path,
    commands: list[dict[str, str | None]],
) -> None:
    lines = [
        "schema_version: chimerax_pose_manifest_v1",
        f"status: {_yaml_scalar(status)}",
        f"failure_reason: {_yaml_scalar(failure_reason)}",
        f"pose_id: {_yaml_scalar(pose_id)}",
        f"captured_at_utc: {_yaml_scalar(captured_at_utc)}",
        f"chimerax_executable: {_yaml_scalar(chimerax_executable)}",
        "inputs:",
        f"  structure_path: {_yaml_scalar(str(structure_path) if structure_path else None)}",
        f"  structure_sha256: {_yaml_scalar(_sha256(structure_path) if structure_path else None)}",
        f"  source_url: {_yaml_scalar(source_url)}",
        f"  opened_model_id: {_yaml_scalar(opened_model_id)}",
        f"  preopened_session: {_yaml_scalar(preopened_session)}",
        "control:",
        '  host: "127.0.0.1"',
        f"  port: {port}",
        f"  rest_stopped: {_yaml_scalar(rest_stopped)}",
        "scene:",
        f"  camera_mode: {_yaml_scalar(camera_mode)}",
        f"  background_color: {_yaml_scalar(background_color)}",
        f"  title: {_yaml_scalar(title)}",
        "outputs:",
        f"  session_path: {_yaml_scalar(str(session_path))}",
        f"  session_sha256: {_yaml_scalar(_sha256(session_path))}",
        f"  image_path: {_yaml_scalar(str(image_path))}",
        f"  image_sha256: {_yaml_scalar(_sha256(image_path))}",
        f"  command_log_path: {_yaml_scalar(str(command_log_path))}",
        f"  command_log_sha256: {_yaml_scalar(_sha256(command_log_path))}",
        "commands:",
    ]
    for entry in commands:
        lines.extend(
            [
                f"  - key: {_yaml_scalar(entry['key'])}",
                f"    command: {_yaml_scalar(entry['command'])}",
                f"    status: {_yaml_scalar(entry['status'])}",
                f"    error: {_yaml_scalar(entry.get('error'))}",
            ]
        )
    manifest_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="Capture the current view from a local ChimeraX REST session.")
    parser.add_argument("--port", type=int)
    parser.add_argument("--session-manifest", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--pose-id", required=True)
    parser.add_argument("--title", default="")
    parser.add_argument("--width", type=int, default=1800)
    parser.add_argument("--height", type=int, default=1200)
    parser.add_argument("--supersample", type=int, default=2)
    parser.add_argument("--camera-mode", choices=("ortho", "mono"), default="ortho")
    parser.add_argument("--background-color", default="white")
    parser.add_argument("--chimerax-executable", default="")
    parser.add_argument("--structure-path", type=Path)
    parser.add_argument("--source-url")
    parser.add_argument("--opened-model-id")
    parser.add_argument("--preopened-session", action="store_true")
    parser.add_argument("--timeout-seconds", type=float, default=15.0)
    parser.add_argument("--keep-rest-open", action="store_true")
    args = parser.parse_args()

    _resolve_manifest_defaults(args)
    if args.port is None:
        parser.error("provide --port or --session-manifest")
    if args.port < 1024 or args.port > 65535:
        parser.error("--port must be from 1024 to 65535")
    if args.structure_path and not args.structure_path.exists():
        parser.error(f"--structure-path does not exist: {args.structure_path}")
    if not args.structure_path and not args.preopened_session:
        parser.error("provide --structure-path or declare --preopened-session")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    session_path = args.output_dir / f"{args.pose_id}.cxs"
    image_path = args.output_dir / f"{args.pose_id}.png"
    command_log_path = args.output_dir / f"{args.pose_id}.commands.jsonl"
    manifest_path = args.output_dir / f"{args.pose_id}.pose_manifest.yaml"
    captured_at_utc = datetime.now(UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z")

    command_entries = [
        {"key": "set_background", "command": f"set bgColor {args.background_color}"},
        {"key": "camera_mode", "command": f"camera {args.camera_mode}"},
        {"key": "name_view", "command": f"view name {args.pose_id}"},
    ]
    if args.title:
        escaped_title = args.title.replace('"', '\\"')
        command_entries.append({"key": "title_label_cleanup", "command": "2dlabels delete all"})
        command_entries.append(
            {
                "key": "title_label",
                "command": f'2dlabels text "{escaped_title}" xpos 0.035 ypos 0.89 size 30 color black bgColor none',
            }
        )
    command_entries.extend(
        [
            {"key": "save_session", "command": f'save "{session_path}"'},
            {
                "key": "save_image",
                "command": (
                    f'save "{image_path}" width {args.width} height {args.height} supersample {args.supersample}'
                ),
            },
        ]
    )
    if not args.keep_rest_open:
        command_entries.append({"key": "rest_stop", "command": "remotecontrol rest stop"})

    command_results: list[dict[str, str | None]] = []
    with command_log_path.open("w", encoding="utf-8") as log_handle:
        for entry in command_entries:
            result = {"key": entry["key"], "command": entry["command"], "status": "accepted", "error": None}
            try:
                response = _send(port=args.port, command=entry["command"], timeout_seconds=args.timeout_seconds)
                log_handle.write(json.dumps({"request": entry, "response": response}, sort_keys=True) + "\n")
            except Exception as exc:  # noqa: BLE001
                result["status"] = "failed"
                result["error"] = str(exc)
                log_handle.write(json.dumps({"request": entry, "error": str(exc)}, sort_keys=True) + "\n")
                command_results.append(result)
                break
            command_results.append(result)

    rest_stopped = bool(
        not args.keep_rest_open
        and command_results
        and command_results[-1]["key"] == "rest_stop"
        and command_results[-1]["status"] == "accepted"
    )
    failed_entries = [entry for entry in command_results if entry["status"] == "failed"]
    status = "failed" if failed_entries else "accepted"
    failure_reason = failed_entries[0]["error"] if failed_entries else None
    _write_manifest(
        manifest_path=manifest_path,
        status=status,
        failure_reason=failure_reason,
        pose_id=args.pose_id,
        captured_at_utc=captured_at_utc,
        chimerax_executable=args.chimerax_executable,
        structure_path=args.structure_path,
        source_url=args.source_url,
        opened_model_id=args.opened_model_id,
        preopened_session=args.preopened_session,
        port=args.port,
        rest_stopped=rest_stopped,
        camera_mode=args.camera_mode,
        background_color=args.background_color,
        title=args.title or None,
        session_path=session_path,
        image_path=image_path,
        command_log_path=command_log_path,
        commands=command_results,
    )
    print(f"pose_manifest={manifest_path}")
    if any(entry["status"] == "failed" for entry in command_results):
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
