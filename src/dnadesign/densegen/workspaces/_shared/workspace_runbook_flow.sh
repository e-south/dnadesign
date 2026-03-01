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
    local usr_registry="$PWD/outputs/usr_datasets/registry.yaml"
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
    local records_table
    records_table="$(dirname "$config")/outputs/tables/records.parquet"
    if [[ ! -f "$records_table" ]]; then
      echo "Analysis mode requires existing outputs at: $records_table" >&2
      echo "Run ./runbook.sh --mode fresh first to generate artifacts, then rerun with --mode analysis." >&2
      return 2
    fi
    set +e
    "${dense_cmd[@]}" inspect run --events --library -c "$config"
    local inspect_status=$?
    set -e
    if [[ $inspect_status -ne 0 ]]; then
      echo "Analysis mode inspection failed. Existing artifacts may be stale or schema-incompatible." >&2
      echo "Run ./runbook.sh --mode fresh, then retry --mode analysis." >&2
      return "$inspect_status"
    fi
    "${dense_cmd[@]}" plot -c "$config"
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
