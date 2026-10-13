#!/usr/bin/env bash
# 第一阶段简易启动：填写下面的目录，然后执行 bash start.sh。
set -eo pipefail
REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

# ===== 必填：均填写 Linux 绝对路径；路径有空格也保留双引号 =====
ENV_DIR="${ENV_DIR:-}"                        # qwen-local 环境目录，内含 bin/python
MODEL_DIR="${MODEL_DIR:-}"                    # 本地模型目录，内含 config.json 和完整权重
PRIMARY_IMAGE_ROOT="${PRIMARY_IMAGE_ROOT:-}"  # 原始图片根目录，其下为语种/截图或实拍

# ===== 可选：通常保持默认即可 =====
OUTPUT_ROOT="${OUTPUT_ROOT:-$REPO_ROOT/results/simple_run}"  # 输出目录；换模型/参数请换新目录
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"           # 单卡 0；多卡 0,1
DATASET="${DATASET:-both}"                                 # both / primary / menu_reference
MENU_IMAGE_ROOT="${MENU_IMAGE_ROOT:-$REPO_ROOT/datasets/menu_reference}"  # 辅助菜单集
GENERATION_CONFIG="${GENERATION_CONFIG:-$REPO_ROOT/configs/generation.json}"  # 长度/采样/显存等
FX_CONTEXT="${FX_CONTEXT:-}"                               # 可选汇率 JSON 文件；留空不猜汇率
CONDA_BASE="${CONDA_BASE:-}"                               # Conda 安装目录，通常自动找到；不是环境目录

fail() { printf '启动失败：%s\n' "$*" >&2; exit 2; }
[[ "$ENV_DIR" == /* && -x "$ENV_DIR/bin/python" ]] || fail '请填写 ENV_DIR 的绝对路径，且该目录中必须有 bin/python。'
[[ "$MODEL_DIR" == /* && -f "$MODEL_DIR/config.json" ]] || fail '请填写 MODEL_DIR 的绝对路径，指向完整的本地模型目录。'
case "$DATASET" in
  both|primary) [[ "$PRIMARY_IMAGE_ROOT" == /* && -d "$PRIMARY_IMAGE_ROOT" ]] || fail '请填写 PRIMARY_IMAGE_ROOT 的绝对路径。' ;;
  menu_reference) ;;
  *) fail 'DATASET 只能填写 both、primary 或 menu_reference。' ;;
esac

# 自动寻找 Conda 安装目录；无需 conda init 或手动提前激活环境。
if [[ -z "$CONDA_BASE" ]] && command -v conda >/dev/null 2>&1; then
  CONDA_BASE="$(conda info --base)"
fi
if [[ -z "$CONDA_BASE" ]]; then
  for candidate in "$HOME/anaconda3" "$HOME/miniconda3" "$(dirname -- "$(dirname -- "$ENV_DIR")")"; do
    if [[ -f "$candidate/etc/profile.d/conda.sh" ]]; then CONDA_BASE="$candidate"; break; fi
  done
fi
[[ -f "$CONDA_BASE/etc/profile.d/conda.sh" ]] || fail '未找到 Conda；请在脚本顶部填写 CONDA_BASE 安装目录。'
source "$CONDA_BASE/etc/profile.d/conda.sh"
conda activate "$ENV_DIR"
# 使用环境内的解释器，避免提示已激活但实际运行到 base 的 Python。
PYTHON_BIN="$ENV_DIR/bin/python"
"$PYTHON_BIN" -c 'import sys; sys.exit(0 if sys.version_info >= (3, 10) else "启动失败：环境需要 Python 3.10 或以上。")'
set -u

export MODEL_DIR PRIMARY_IMAGE_ROOT OUTPUT_ROOT CUDA_VISIBLE_DEVICES DATASET
export MENU_IMAGE_ROOT GENERATION_CONFIG FX_CONTEXT PYTHON_BIN
printf '环境：%s\n模型：%s\n图片：%s\n输出：%s/handoff.json\n' "$ENV_DIR" "$MODEL_DIR" "$PRIMARY_IMAGE_ROOT" "$OUTPUT_ROOT"
exec bash "$REPO_ROOT/scripts/generate.sh" "$@"
