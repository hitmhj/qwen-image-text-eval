#!/usr/bin/env bash
set -euo pipefail

RUN_DIR="${RUN_DIR:-/请填写输出目录/首次基线}"
PYTHON_BIN="${PYTHON_BIN:-python3}"
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "$SCRIPT_DIR/.." && pwd)"
export PYTHONPATH="$REPO_ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
export HF_HUB_OFFLINE=1
export HF_HUB_DISABLE_TELEMETRY=1

"$PYTHON_BIN" -m qwen_text_eval rules --run-dir "$RUN_DIR"
"$PYTHON_BIN" -m qwen_text_eval prepare-reference --run-dir "$RUN_DIR"
"$PYTHON_BIN" -m qwen_text_eval report --run-dir "$RUN_DIR"
printf '%s\n' '规则和阶段报告已生成。事实参考及语义评分等待接口或人工标注回包；没有执行联网调用。'
