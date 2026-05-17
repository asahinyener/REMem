#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${repo_root}"

if [[ -f .env ]]; then
  set -a
  # shellcheck disable=SC1091
  source .env
  set +a
fi

export PATH="${DENO_BIN_DIR:-/root/.deno/bin}:${PATH}"
export PYTHONDONTWRITEBYTECODE="${PYTHONDONTWRITEBYTECODE:-1}"
export TIMELINEKGQA_DATASET_DIR="${TIMELINEKGQA_DATASET_DIR:-../external/timelinekgqa/Datasets}"
export TIMELINEKGQA_QUESTION_ID="${TIMELINEKGQA_QUESTION_ID:-25107}"
export TIMELINEKGQA_RLM_SUBLM_DEMO="${TIMELINEKGQA_RLM_SUBLM_DEMO:-true}"
export AGENT_MAX_STEPS="${AGENT_MAX_STEPS:-8}"
export RLM_MAX_TOKENS="${RLM_MAX_TOKENS:-8192}"
export FORCE_INDEX="${FORCE_INDEX:-false}"
export FORCE_OPENIE="${FORCE_OPENIE:-false}"
export LITELLM_DROP_PARAMS="${LITELLM_DROP_PARAMS:-true}"

if [[ -d ../external/dspy_recursive ]]; then
  export PYTHONPATH="../external/dspy_recursive:src${PYTHONPATH:+:${PYTHONPATH}}"
else
  export PYTHONPATH="src${PYTHONPATH:+:${PYTHONPATH}}"
fi

python examples/timelinekgqa/check_reproduction.py
python examples/timelinekgqa/run_timelinekgqa_temporal_rlm_trace.py
