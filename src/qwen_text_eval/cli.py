"""各阶段独立执行；没有联网接口或自动数据上传。"""

import argparse
import json
import sys
import sysconfig
from pathlib import Path

from .dataset import create_manifest, load_manifest
from .evaluation import (evaluate_run_rules, import_judgements, import_references,
                         prepare_judge_requests, prepare_reference_requests)
from .reporting import build_report
from .storage import atomic_json, read_json
from .validation import JUDGE_SCHEMA, REFERENCE_SCHEMA, validate_rubric


def assets_root() -> Path:
    repository = Path(__file__).resolve().parents[2]
    if (repository / "prompts" / "v04_summary.json").exists():
        return repository
    installed = Path(sysconfig.get_path("data")) / "share" / "qwen-text-eval"
    if (installed / "prompts" / "v04_summary.json").exists():
        return installed
    raise FileNotFoundError("找不到提示词和配置；请在仓库中运行或正确安装本项目")


def load_prompts(directory: Path | None = None) -> dict:
    directory = directory or assets_root() / "prompts"
    result = {"summary": read_json(directory / "v04_summary.json"),
              "interpretation": read_json(directory / "v04_interpret.json"),
              "recommendation": read_json(directory / "v01_recommendation.json")}
    for task, prompt in result.items():
        if prompt.get("task") != task or not all(isinstance(prompt.get(key), str) and prompt[key] for key in ["version", "system", "user"]):
            raise ValueError("提示词必须包含正确任务以及非空 version、system、user")
    return result


def rubric_for_run(run_dir: Path, explicit: Path | None = None) -> dict:
    saved = run_dir / "rubric.json"
    rubric = read_json(explicit) if explicit else read_json(saved if saved.exists() else assets_root() / "configs" / "rubric.json")
    validate_rubric(rubric)
    if saved.exists() and read_json(saved) != rubric:
        raise ValueError("评分配置与运行快照不同，不能混合旧评审结果")
    return rubric


def parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser(description="本地图片文本生成与细粒度测评；所有阶段默认无网络")
    commands = root.add_subparsers(dest="command", required=True)
    scan = commands.add_parser("scan", help="扫描全部语种与截图／实拍，建立清单")
    scan.add_argument("--image-root", type=Path, required=True, help="图片大目录")
    scan.add_argument("--run-dir", type=Path, required=True, help="新的输出目录")
    scan.add_argument("--dataset-id", choices=["primary", "menu_reference"], default="primary")
    generation = commands.add_parser("generate", help="读取本地 Qwen3.5-35B-A3B，每张图生成三个任务")
    generation.add_argument("--image-root", type=Path, required=True)
    generation.add_argument("--run-dir", type=Path, required=True)
    generation.add_argument("--model-dir", type=Path, required=True, help="本地 Hugging Face 模型目录")
    generation.add_argument("--config", type=Path, help="生成配置；默认使用 configs/generation.json")
    generation.add_argument("--prompts-dir", type=Path, help="可选提示词目录")
    generation.add_argument("--no-resume", action="store_true", help="拒绝复用已有成功样本")
    generation.add_argument("--dataset-id", choices=["primary", "menu_reference"], default="primary")
    generation.add_argument("--requirements-pool", type=Path)
    generation.add_argument("--fx-context", type=Path, help="可选的冻结汇率文件；不自动获取汇率")
    export = commands.add_parser("export", help="导出每图一行的三任务结果给另一平台")
    export.add_argument("--run-dir", type=Path, required=True)
    export.add_argument("--output", type=Path)
    reference = commands.add_parser("prepare-reference", help="准备独立看图标注请求，不提供 Qwen 答案")
    reference.add_argument("--run-dir", type=Path, required=True)
    reference.add_argument("--system-prompt", type=Path)
    reference_import = commands.add_parser("import-reference", help="校验并导入独立看图模型或人工标注的 JSONL")
    reference_import.add_argument("--run-dir", type=Path, required=True)
    reference_import.add_argument("--input", type=Path, required=True)
    reference_import.add_argument("--human-verified", action="store_true", help="仅对操作者已逐条核验的整份输入使用")
    reference_import.add_argument("--replace", action="store_true")
    judge = commands.add_parser("prepare-judge", help="准备基于事实参考的逐断言评审请求")
    judge.add_argument("--run-dir", type=Path, required=True)
    judge.add_argument("--system-prompt", type=Path)
    judge.add_argument("--rubric", type=Path)
    judge.add_argument("--prompts-dir", type=Path)
    judge_import = commands.add_parser("import-judge", help="校验评审 JSONL，拒绝过期回包或错误引用")
    judge_import.add_argument("--run-dir", type=Path, required=True)
    judge_import.add_argument("--input", type=Path, required=True)
    judge_import.add_argument("--rubric", type=Path)
    judge_import.add_argument("--replace", action="store_true")
    rules = commands.add_parser("rules", help="对成功生成的原文执行规则测评")
    rules.add_argument("--run-dir", type=Path, required=True)
    report = commands.add_parser("report", help="汇总已完成环节，未测语义指标为空")
    report.add_argument("--run-dir", type=Path, required=True)
    report.add_argument("--rubric", type=Path)
    schema = commands.add_parser("schema", help="输出未来接口回包的 JSON Schema")
    schema.add_argument("kind", choices=["reference", "judge"])
    return root


def main(argv=None) -> int:
    args = parser().parse_args(argv)
    if args.command not in {"scan", "generate", "export"}:
        print("旧包内评审入口已停用；三任务参考、评分和HTML请使用根目录 evaluate.py。", file=sys.stderr)
        return 2
    try:
        if args.command == "scan":
            result = create_manifest(args.image_root, args.run_dir, args.dataset_id)
        elif args.command == "generate":
            from .generation import run_generation
            if not (args.run_dir / "run.json").exists():
                create_manifest(args.image_root, args.run_dir, args.dataset_id)
            else:
                load_manifest(args.run_dir)
                if read_json(args.run_dir / "run.json").get("dataset_id", "primary") != args.dataset_id:
                    raise ValueError("数据集标识与已有运行不同，请分开生成和测评")
            prompts = load_prompts(args.prompts_dir)
            config = read_json(args.config or assets_root() / "configs" / "generation.json")
            # 模型加载前只写未存在的快照；已有快照由生成模块验证。
            for task, prompt in prompts.items():
                snapshot = args.run_dir / "prompts" / f"{task}.json"
                if snapshot.exists() and read_json(snapshot) != prompt:
                    raise ValueError("提示词与旧运行不同，请另建运行目录")
            for task, prompt in prompts.items():
                atomic_json(args.run_dir / "prompts" / f"{task}.json", prompt)
            pool = read_json(args.requirements_pool or assets_root() / "configs" / "requirements_pool.json")
            fx = read_json(args.fx_context) if args.fx_context else None
            result = run_generation(args.run_dir, args.model_dir, args.image_root, config, prompts,
                                    resume=not args.no_resume, requirements_pool=pool, fx_context=fx)
            from .exporting import export_generation_results
            result["export"] = export_generation_results(args.run_dir)
        elif args.command == "export":
            from .exporting import export_generation_results
            result = export_generation_results(args.run_dir, args.output)
        elif args.command == "prepare-reference":
            result = prepare_reference_requests(args.run_dir, args.system_prompt or assets_root() / "prompts" / "reference_system.txt")
        elif args.command == "import-reference":
            result = import_references(args.run_dir, args.input, args.human_verified, args.replace)
        elif args.command == "prepare-judge":
            result = prepare_judge_requests(args.run_dir, args.system_prompt or assets_root() / "prompts" / "judge_system.txt",
                                            rubric_for_run(args.run_dir, args.rubric), load_prompts(args.prompts_dir))
        elif args.command == "import-judge":
            result = import_judgements(args.run_dir, args.input, rubric_for_run(args.run_dir, args.rubric), args.replace)
        elif args.command == "rules":
            result = evaluate_run_rules(args.run_dir)
        elif args.command == "report":
            result = build_report(args.run_dir, rubric_for_run(args.run_dir, args.rubric))
        else:
            result = REFERENCE_SCHEMA if args.kind == "reference" else JUDGE_SCHEMA
        print(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False))
        return 1 if args.command == "generate" and (result.get("generated_error", 0) or result.get("skipped_invalid_images", 0)) else 0
    except (ValueError, TypeError, KeyError, FileNotFoundError, RuntimeError, OSError) as error:
        print(f"错误：{error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
