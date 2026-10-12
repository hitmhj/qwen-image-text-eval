#!/usr/bin/env bash
set -euo pipefail

# 在华为机器上填写本地目录；也可以在运行前设置同名环境变量。
MODEL_DIR="${MODEL_DIR:-/请填写本地模型目录/Qwen3.5-35B-A3B}"
IMAGE_ROOT="${IMAGE_ROOT:-/请填写图片根目录}"
RUN_DIR="${RUN_DIR:-/请填写输出目录/首次基线}"
# 单卡填 0，多卡如 0,1；需要足够的总显存并留出推理空间。
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
PYTHON_BIN="${PYTHON_BIN:-python3}"

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "$SCRIPT_DIR/.." && pwd)"
GENERATION_CONFIG="${GENERATION_CONFIG:-$REPO_ROOT/configs/generation.json}"

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
if [[ ! -d "$IMAGE_ROOT" ]]; then
  printf '%s\n' '图片根目录不存在，请填写 IMAGE_ROOT。' >&2
  exit 2
fi

exec "$PYTHON_BIN" -m qwen_text_eval generate \
  --model-dir "$MODEL_DIR" \
  --image-root "$IMAGE_ROOT" \
  --run-dir "$RUN_DIR" \
  --config "$GENERATION_CONFIG" "$@"
