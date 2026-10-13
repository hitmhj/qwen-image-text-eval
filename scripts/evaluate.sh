#!/usr/bin/env bash
set -euo pipefail
STEP="${STEP:-report}"
GENERATION_INPUT="${GENERATION_INPUT:-/请填写单个数据集/generation_results.jsonl}"
OUTPUT_DIR="${OUTPUT_DIR:-/请填写对应数据集的评测目录}"
PYTHON_BIN="${PYTHON_BIN:-python3}"
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "$SCRIPT_DIR/.." && pwd)"
ARGS=("$STEP" --input "$GENERATION_INPUT" --output-dir "$OUTPUT_DIR")
if [[ -n "${IMAGE_ROOT:-}" ]]; then ARGS+=(--image-root "$IMAGE_ROOT"); fi
if [[ -n "${GROUND_TRUTH:-}" ]]; then ARGS+=(--ground-truth "$GROUND_TRUTH"); fi
if [[ -n "${SCORES:-}" ]]; then ARGS+=(--scores "$SCORES"); fi
if [[ -n "${RESPONSES:-}" ]]; then ARGS+=(--responses "$RESPONSES"); fi
exec "$PYTHON_BIN" "$REPO_ROOT/evaluate.py" "${ARGS[@]}" "$@"
