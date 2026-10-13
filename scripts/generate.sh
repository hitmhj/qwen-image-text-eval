#!/usr/bin/env bash
set -euo pipefail

# 在华为机器上填写本地目录；也可以在运行前设置同名环境变量。
MODEL_DIR="${MODEL_DIR:-/请填写本地模型目录/Qwen3.5-35B-A3B}"
PRIMARY_IMAGE_ROOT="${PRIMARY_IMAGE_ROOT:-/请填写原始图片根目录}"
OUTPUT_ROOT="${OUTPUT_ROOT:-/请填写输出目录/首次三任务基线}"
DATASET="${DATASET:-both}"
# 单卡填 0，多卡如 0,1；需要足够的总显存并留出推理空间。
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
PYTHON_BIN="${PYTHON_BIN:-python3}"

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "$SCRIPT_DIR/.." && pwd)"
GENERATION_CONFIG="${GENERATION_CONFIG:-$REPO_ROOT/configs/generation.json}"
MENU_IMAGE_ROOT="${MENU_IMAGE_ROOT:-$REPO_ROOT/datasets/menu_reference}"
REQUIREMENTS_POOL="${REQUIREMENTS_POOL:-$REPO_ROOT/configs/requirements_pool.json}"

# 只读本地模型；本脚本不下载模型、依赖或调用任何外部接口。
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_HUB_DISABLE_TELEMETRY=1
export DO_NOT_TRACK=1
export TOKENIZERS_PARALLELISM=false
export PYTHONPATH="$REPO_ROOT/src${PYTHONPATH:+:$PYTHONPATH}"

if [[ ! -f "$MODEL_DIR/config.json" ]]; then
  printf '%s\n' '模型目录不完整：请填写含 config.json 的本地 Hugging Face 模型目录。' >&2
  exit 2
fi
EXTRA_ARGS=()
if [[ -n "${FX_CONTEXT:-}" ]]; then
  EXTRA_ARGS+=(--fx-context "$FX_CONTEXT")
fi
exec "$PYTHON_BIN" "$REPO_ROOT/generate.py" \
  --model-dir "$MODEL_DIR" \
  --primary-images "$PRIMARY_IMAGE_ROOT" \
  --menu-images "$MENU_IMAGE_ROOT" \
  --output-root "$OUTPUT_ROOT" \
  --dataset "$DATASET" \
  --requirements-pool "$REQUIREMENTS_POOL" \
  --config "$GENERATION_CONFIG" "${EXTRA_ARGS[@]}" "$@"
