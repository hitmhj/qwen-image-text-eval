"""平台一入口：原始客户图与清晰菜单图分别运行，分别导出三任务结果。"""

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from qwen_text_eval.cli import main as generate_main


def main(argv=None):
    parser = argparse.ArgumentParser(description="两套独立测试集的本地 Qwen 三任务生成")
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--primary-images", type=Path, help="您原来的图片根目录")
    parser.add_argument("--menu-images", type=Path, default=ROOT / "datasets" / "menu_reference")
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--dataset", choices=["both", "primary", "menu_reference"], default="both")
    parser.add_argument("--config", type=Path, default=ROOT / "configs" / "generation.json")
    parser.add_argument("--requirements-pool", type=Path, default=ROOT / "configs" / "requirements_pool.json")
    parser.add_argument("--fx-context", type=Path)
    parser.add_argument("--no-resume", action="store_true")
    args = parser.parse_args(argv)
    selected = []
    if args.dataset in {"both", "primary"}:
        if args.primary_images is None or not args.primary_images.is_dir():
            parser.error("需要填写有效的 --primary-images；仅测菜单时使用 --dataset menu_reference")
        selected.append(("primary", args.primary_images))
    if args.dataset in {"both", "menu_reference"}:
        if not args.menu_images.is_dir():
            parser.error("清晰菜单目录不存在，请填写 --menu-images")
        selected.append(("menu_reference", args.menu_images))
    if len(selected) == 2:
        primary, auxiliary = (path.resolve() for _, path in selected)
        if primary == auxiliary or primary.is_relative_to(auxiliary) or auxiliary.is_relative_to(primary):
            parser.error("两个测试集的输入根目录不能相同或互相包含")
    result = 0
    for dataset_id, images in selected:
        print(f"开始独立生成：{dataset_id}", flush=True)
        command = ["generate", "--model-dir", str(args.model_dir), "--image-root", str(images),
                   "--run-dir", str(args.output_root / dataset_id), "--dataset-id", dataset_id,
                   "--config", str(args.config), "--requirements-pool", str(args.requirements_pool)]
        if args.fx_context:
            command.extend(["--fx-context", str(args.fx_context)])
        if args.no_resume:
            command.append("--no-resume")
        code = generate_main(command)
        result = max(result, code)
        # 释放上一数据集的 Python 对象及 CUDA 缓存，避免跨批次权重堆积。
        import gc
        gc.collect()
        torch = sys.modules.get("torch")
        if torch is not None and torch.cuda.is_available():
            torch.cuda.empty_cache()
    return result


if __name__ == "__main__":
    raise SystemExit(main())
