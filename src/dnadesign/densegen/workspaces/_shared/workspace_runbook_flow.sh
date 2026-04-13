#!/usr/bin/env bash

_densegen_require_command() {
  local cmd="$1"
  if command -v "$cmd" >/dev/null 2>&1; then
    return 0
  fi
  echo "Missing required command: $cmd" >&2
  return 127
}


_densegen_workspace_runbook_usage() {
  cat <<'EOF'
Usage: ./runbook.sh [--mode fresh|resume|analysis]

Runbook modes:
  fresh     run validate -> run --fresh -> inspect -> plot -> notebook
  resume    run validate -> run --resume -> inspect -> plot -> notebook
  analysis  run inspect -> plot -> notebook from existing outputs only
EOF
}


_densegen_log_contains_any() {
  local log_path="$1"
  shift
  local needle
  for needle in "$@"; do
    if grep -Fqi "$needle" "$log_path"; then
      return 0
    fi
  done
  return 1
}


_densegen_resolve_analysis_records_path() {
  local config="$1"
  local workspace_dir="$2"
  uv run python -c '
import sys
from pathlib import Path

from dnadesign.densegen.src.config.base import resolve_outputs_scoped_path, resolve_usr_root_scoped_path
from dnadesign.densegen.src.config.root import load_config

cfg_path = Path(sys.argv[1]).resolve()
workspace_dir = Path(sys.argv[2]).resolve()
loaded = load_config(cfg_path)
out_cfg = loaded.root.densegen.output
targets = list(out_cfg.targets)
if not targets:
    raise SystemExit("output.targets must contain at least one sink")
if len(targets) > 1:
    plots_cfg = loaded.root.plots
    if plots_cfg is None or plots_cfg.source is None:
        raise SystemExit("plots.source must be set when output.targets has multiple sinks")
    source = str(plots_cfg.source).strip()
    if source not in targets:
        raise SystemExit("plots.source must be one of output.targets")
else:
    source = str(targets[0]).strip()

if source == "parquet":
    pq_cfg = out_cfg.parquet
    if pq_cfg is None:
        raise SystemExit("output.parquet is required when analysis source resolves to parquet")
    print(resolve_outputs_scoped_path(cfg_path, workspace_dir, pq_cfg.path, label="output.parquet.path"))
elif source == "usr":
    usr_cfg = out_cfg.usr
    if usr_cfg is None:
        raise SystemExit("output.usr is required when analysis source resolves to usr")
    dataset = str(usr_cfg.dataset or "").strip()
    if not dataset:
        raise SystemExit("output.usr.dataset must be a non-empty string")
    usr_root = resolve_usr_root_scoped_path(cfg_path, usr_cfg.root, label="output.usr.root")
    print(usr_root / dataset / "records.parquet")
else:
    raise SystemExit(f"unsupported analysis source: {source}")
' "$config" "$workspace_dir"
}


densegen_workspace_runbook_flow() {
  local config=""
  local notebook=""
  local runner=""
  local ensure_usr_registry="false"
  local require_fimo="false"
  local run_mode="fresh"

  while [[ $# -gt 0 ]]; do
    case "$1" in
      --config)
        config="$2"
        shift 2
        ;;
      --notebook)
        notebook="$2"
        shift 2
        ;;
      --runner)
        runner="$2"
        shift 2
        ;;
      --ensure-usr-registry)
        ensure_usr_registry="$2"
        shift 2
        ;;
      --require-fimo)
        require_fimo="$2"
        shift 2
        ;;
      --mode)
        run_mode="$2"
        shift 2
        ;;
      --help|-h)
        _densegen_workspace_runbook_usage
        return 0
        ;;
      *)
        echo "Unknown densegen_workspace_runbook_flow option: $1" >&2
        return 2
        ;;
    esac
  done

  if [[ -z "$config" || -z "$notebook" || -z "$runner" ]]; then
    echo "densegen_workspace_runbook_flow requires --config, --notebook, and --runner" >&2
    return 2
  fi
  if [[ "$run_mode" != "fresh" && "$run_mode" != "resume" && "$run_mode" != "analysis" ]]; then
    echo "Unsupported --mode value: $run_mode (expected fresh|resume|analysis)" >&2
    return 2
  fi

  if [[ ! -f "$config" ]]; then
    echo "DenseGen config not found at: $config" >&2
    return 2
  fi

  local -a dense_cmd
  case "$runner" in
    uv)
      dense_cmd=(uv run dense)
      ;;
    pixi)
      dense_cmd=(pixi run dense)
      ;;
    *)
      echo "Unsupported --runner value: $runner (expected uv|pixi)" >&2
      return 2
      ;;
  esac

  _densegen_require_command uv
  _densegen_require_command git
  if [[ "$runner" == "pixi" || "$require_fimo" == "true" ]]; then
    _densegen_require_command pixi
  fi

  if [[ "$ensure_usr_registry" == "true" ]]; then
    local usr_root
    usr_root="$(
      uv run python -c '
import sys
from pathlib import Path
import yaml
cfg = yaml.safe_load(Path(sys.argv[1]).read_text()) or {}
usr = (((cfg.get("densegen") or {}).get("output") or {}).get("usr") or {})
root = str(usr.get("root") or "").strip()
if not root:
    raise SystemExit("missing densegen.output.usr.root")
workspace_dir = Path(sys.argv[2]).resolve()
print((workspace_dir / Path(root)).resolve())
' "$config" "$PWD"
    )"
    local usr_registry="$usr_root/registry.yaml"
    local root_registry
    root_registry="$(git rev-parse --show-toplevel)/src/dnadesign/usr/datasets/registry.yaml"
    if [[ ! -f "$root_registry" ]]; then
      echo "USR registry source not found at: $root_registry" >&2
      return 2
    fi
    if [[ ! -f "$usr_registry" ]]; then
      mkdir -p "$(dirname "$usr_registry")"
      cp "$root_registry" "$usr_registry"
    fi
  fi

  if [[ "$require_fimo" == "true" && "$run_mode" != "analysis" ]]; then
    pixi run fimo --version
  fi

  if [[ "$run_mode" == "analysis" ]]; then
    local records_source_path
    local plot_manifest
    records_source_path="$(_densegen_resolve_analysis_records_path "$config" "$PWD")"
    plot_manifest="$(dirname "$config")/outputs/plots/plot_manifest.json"
    if [[ ! -f "$records_source_path" ]]; then
      echo "Analysis mode requires existing outputs at: $records_source_path" >&2
      echo "Run ./runbook.sh --mode fresh first to generate artifacts, then rerun with --mode analysis." >&2
      return 2
    fi
    local inspect_log
    inspect_log="$(mktemp)"
    set +e
    "${dense_cmd[@]}" inspect run --events --library -c "$config" >"$inspect_log" 2>&1
    local inspect_status=$?
    set -e
    cat "$inspect_log"
    if [[ $inspect_status -ne 0 ]]; then
      if _densegen_log_contains_any "$inspect_log" "Run manifest not found" "run_manifest.json"; then
        echo "Analysis mode inspection skipped: workspace lacks finalized run metadata, but records-derived analysis can continue." >&2
        echo "Continuing with plots and notebook refresh from existing records artifacts." >&2
      else
        echo "Analysis mode inspection failed. Existing artifacts may be stale or schema-incompatible." >&2
        echo "Run ./runbook.sh --mode fresh, then retry --mode analysis." >&2
        rm -f "$inspect_log"
        return "$inspect_status"
      fi
    fi
    rm -f "$inspect_log"
    local plot_log
    plot_log="$(mktemp)"
    set +e
    "${dense_cmd[@]}" plot -c "$config" >"$plot_log" 2>&1
    local plot_status=$?
    set -e
    cat "$plot_log"
    if [[ $plot_status -ne 0 ]]; then
      if [[ -f "$plot_manifest" ]] && _densegen_log_contains_any "$plot_log" "pool manifest not found" "pool_manifest.json" "attempts.parquet not found" "composition.parquet not found"; then
        echo "Analysis mode plot refresh completed with partial success; notebook will show generated plots and explicit local-artifact gaps." >&2
      else
        rm -f "$plot_log"
        return "$plot_status"
      fi
    fi
    rm -f "$plot_log"
    "${dense_cmd[@]}" notebook generate --force -c "$config"
  else
    "${dense_cmd[@]}" validate-config --probe-solver -c "$config"

    set +e
    if [[ "$run_mode" == "fresh" ]]; then
      "${dense_cmd[@]}" run --fresh --no-plot -c "$config"
    else
      "${dense_cmd[@]}" run --resume --no-plot -c "$config"
    fi
    local run_status=$?
    set -e

    "${dense_cmd[@]}" inspect run --events --library -c "$config"

    if [[ $run_status -ne 0 ]]; then
      echo "dense run ($run_mode mode) exited with status $run_status. inspect output above summarizes generated state." >&2
      return "$run_status"
    fi

    "${dense_cmd[@]}" plot -c "$config"
    "${dense_cmd[@]}" notebook generate -c "$config"
  fi

  if [[ ! -f "$notebook" ]]; then
    echo "DenseGen notebook was not generated at: $notebook" >&2
    return 2
  fi
  uv run marimo check "$notebook"
}
