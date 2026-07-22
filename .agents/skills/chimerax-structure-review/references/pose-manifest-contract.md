# Pose Manifest Contract

The pose manifest is the durable record for a captured ChimeraX view.

## Required Fields

```yaml
schema_version: chimerax_pose_manifest_v1
status: accepted | failed
failure_reason: string | null
pose_id: string
captured_at_utc: string
chimerax_executable: string
inputs:
  structure_path: string | null
  structure_sha256: string | null
  source_url: string | null
  opened_model_id: string | null
  preopened_session: bool
control:
  host: 127.0.0.1
  port: int
  rest_stopped: bool
scene:
  camera_mode: string
  background_color: string
  title: string | null
outputs:
  session_path: string
  session_sha256: string | null
  image_path: string
  image_sha256: string | null
  command_log_path: string
  command_log_sha256: string | null
commands:
  - key: string
    command: string
    status: accepted | failed
    error: string | null
```

## Rules

- The `.cxs` session is a human-review artifact and may contain local path state.
- The manifest is the reproducibility contract.
- Record the input structure path and hash whenever the agent opened the model.
- If the model was already open, set `preopened_session: true` and record the model ID when known.
- Save camera mode explicitly because named views do not store camera mode.
- Use stable pose IDs, such as `reference_front_v1` or `candidate_panel_thumb_down_v1`.
- Hash every file written by the capture helper.

## Failure Rows

If capture fails after writing partial outputs, write a manifest with:

```yaml
status: failed
failure_reason: string
```

and include any command responses available.
