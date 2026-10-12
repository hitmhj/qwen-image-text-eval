"""在本地读取 Qwen3.5 多模态模型，逐图生成并保存原始结果。"""

from __future__ import annotations

import copy
import importlib.metadata
import json
import math
import os
import re
import time
from pathlib import Path, PurePosixPath
from typing import Any

from .storage import atomic_json, fingerprint, read_jsonl, sha256_file, utc_now

TASKS = ("summary", "interpretation")
DEFAULT_CONFIG: dict[str, Any] = {
    "model_label": "Qwen3.5-35B-A3B",
    "dtype": "bfloat16",
    "device_map": "auto",
    "attn_implementation": "sdpa",
    "seed": 20261012,
    "enable_thinking": False,
    "do_sample": False,
    "temperature": 0.7,
    "top_p": 0.8,
    "top_k": 20,
    "repetition_penalty": 1.0,
    "image_min_pixels": 65536,
    "image_max_pixels": 2097152,
    "max_new_tokens": {"summary": 192, "interpretation": 2048},
    "max_memory": None,
    "offload_folder": None,
}


def _effective_config(config: dict[str, Any]) -> dict[str, Any]:
    unknown = set(config) - set(DEFAULT_CONFIG)
    if unknown:
        raise ValueError(f"生成配置存在未知字段：{', '.join(sorted(unknown))}")
    merged = copy.deepcopy(DEFAULT_CONFIG)
    merged.update(copy.deepcopy(config))
    if "max_new_tokens" in config:
        if not isinstance(config["max_new_tokens"], dict):
            raise ValueError("max_new_tokens 必须分别设置 summary 与 interpretation")
        if set(config["max_new_tokens"]) - set(TASKS):
            raise ValueError("max_new_tokens 存在未知任务")
        merged["max_new_tokens"] = {
            **DEFAULT_CONFIG["max_new_tokens"], **config["max_new_tokens"]
        }
    for name in ("enable_thinking", "do_sample"):
        if type(merged[name]) is not bool:
            raise ValueError(f"{name} 必须为布尔值")
    for name in ("seed", "top_k", "image_min_pixels", "image_max_pixels"):
        if type(merged[name]) is not int or merged[name] < 0:
            raise ValueError(f"{name} 必须为非负整数")
    if not 0 < merged["image_min_pixels"] <= merged["image_max_pixels"]:
        raise ValueError("图片像素上限必须不小于正数的像素下限")
    for task in TASKS:
        value = merged["max_new_tokens"][task]
        if type(value) is not int or value <= 0:
            raise ValueError(f"{task} 的生成令牌上限必须为正整数")
    for name in ("temperature", "repetition_penalty"):
        value = merged[name]
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or not value > 0:
            raise ValueError(f"{name} 必须为正数")
    if isinstance(merged["top_p"], bool) or not isinstance(merged["top_p"], (int, float)):
        raise ValueError("top_p 必须为数值")
    if not math.isfinite(merged["top_p"]) or not 0 < merged["top_p"] <= 1:
        raise ValueError("top_p 必须在 (0, 1] 范围内")
    if merged["dtype"] not in ("bfloat16", "float16", "float32", "auto"):
        raise ValueError("dtype 只支持 bfloat16、float16、float32、auto")
    if not isinstance(merged["device_map"], (str, dict)):
        raise ValueError("device_map 必须为字符串或设备映射对象")
    if not isinstance(merged["attn_implementation"], str):
        raise ValueError("attn_implementation 必须为字符串")
    if merged["max_memory"] is not None and not isinstance(merged["max_memory"], dict):
        raise ValueError("max_memory 必须为设备到内存额度的映射或 null")
    if merged["offload_folder"] is not None and not isinstance(merged["offload_folder"], str):
        raise ValueError("offload_folder 必须为本地目录字符串或 null")
    return merged


def _model_identity(model_dir: Path) -> dict[str, Any]:
    if not model_dir.is_dir() or not (model_dir / "config.json").is_file():
        raise ValueError("模型路径必须是含 config.json 的本地模型目录")
    model_config = json.loads((model_dir / "config.json").read_text(encoding="utf-8"))
    if model_config.get("model_type") != "qwen3_5_moe":
        raise ValueError("当前实现要求 Qwen3.5-35B-A3B 的 qwen3_5_moe 多模态模型")
    artifacts = []
    for path in sorted(model_dir.iterdir(), key=lambda p: p.name):
        if not path.is_file():
            continue
        if path.suffix not in (".json", ".jinja", ".safetensors", ".bin", ".txt"):
            continue
        if path.name in ("README.md", "LICENSE"):
            continue
        stat = path.stat()
        item = {"name": path.name, "bytes": stat.st_size}
        if path.suffix in (".safetensors", ".bin"):
            # 大权重记录大小和修改时间；目录保持不可变，避免每次读取约 70GB。
            item["mtime_ns"] = stat.st_mtime_ns
        else:
            item["sha256"] = sha256_file(path)
        artifacts.append(item)
    return {"model_dir": str(model_dir), "model_type": "qwen3_5_moe", "artifacts": artifacts}


def _software_versions() -> dict[str, str | None]:
    versions = {}
    for name in ("torch", "torchvision", "transformers", "accelerate", "Pillow"):
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    return versions


def _image_path(image_root: Path, record: dict[str, Any]) -> Path:
    relative = PurePosixPath(record["relative_path"])
    if relative.is_absolute() or ".." in relative.parts or "\\" in record["relative_path"]:
        raise ValueError("清单图片路径必须是根目录内的相对路径")
    image_path = (image_root / Path(*relative.parts)).resolve()
    if not image_path.is_relative_to(image_root):
        raise ValueError("图片路径或符号链接越过图片根目录")
    if not image_path.is_file():
        raise FileNotFoundError("清单中的图片已经不存在")
    if sha256_file(image_path) != record["sha256"]:
        raise ValueError("图片内容与清单 SHA256 不一致，请重新扫描到新运行目录")
    return image_path


def _stable_seed(seed: int, image_id: str, task: str) -> int:
    return (seed + int(fingerprint({"image_id": image_id, "task": task})[:8], 16)) % (2**32)


class _QwenRuntime:
    def __init__(self, model_dir: Path, config: dict[str, Any]):
        # 此处才导入 GPU 依赖，扫描、汇总和单元测试不需要安装 torch。
        try:
            import torch
            from PIL import Image, ImageOps
            import transformers
            from transformers import AutoProcessor, Qwen3_5MoeForConditionalGeneration, set_seed
        except ImportError as exc:
            raise RuntimeError(
                "生成依赖不可用：请离线安装支持 Qwen3.5 的 transformers>=5.3.0、"
                "accelerate，以及与 NVIDIA CUDA 匹配的 torch/torchvision 和 Pillow"
            ) from exc
        if not torch.cuda.is_available():
            raise RuntimeError("未发现可用的 NVIDIA CUDA GPU，请检查驱动和 torch 的 CUDA 版本")
        if config["dtype"] == "bfloat16" and not all(
            torch.cuda.get_device_capability(i)[0] >= 8
            for i in range(torch.cuda.device_count())
        ):
            raise RuntimeError("当前可见 GPU 不支持原生 bfloat16，请显式改用 float16 并建立新运行目录")
        self.torch = torch
        self.Image = Image
        self.ImageOps = ImageOps
        self.set_seed = set_seed
        self.config = config
        self.processor = AutoProcessor.from_pretrained(
            str(model_dir), local_files_only=True, trust_remote_code=False
        )
        model_kwargs: dict[str, Any] = {
            "local_files_only": True,
            "trust_remote_code": False,
            "dtype": config["dtype"] if config["dtype"] == "auto" else getattr(torch, config["dtype"]),
            "device_map": config["device_map"],
            "attn_implementation": config["attn_implementation"],
        }
        if config["max_memory"]:
            model_kwargs["max_memory"] = {
                int(key) if str(key).isdigit() else key: value
                for key, value in config["max_memory"].items()
            }
        if config["offload_folder"]:
            model_kwargs["offload_folder"] = config["offload_folder"]
        self.model = Qwen3_5MoeForConditionalGeneration.from_pretrained(str(model_dir), **model_kwargs)
        self.model.eval()
        self.info = {
            "model_class": type(self.model).__name__,
            "transformers_version": transformers.__version__,
            "torch_version": torch.__version__,
            "cuda_version": torch.version.cuda,
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "gpu_names": [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())],
            "hf_device_map": {str(k): str(v) for k, v in getattr(self.model, "hf_device_map", {}).items()},
        }

    def _synchronize(self) -> None:
        for device in range(self.torch.cuda.device_count()):
            self.torch.cuda.synchronize(device)

    def cleanup_after_error(self) -> None:
        self.torch.cuda.empty_cache()

    def generate(self, image_path: Path, prompt: dict[str, Any], task: str, seed: int) -> dict[str, Any]:
        self.set_seed(seed)
        self._synchronize()
        start = time.perf_counter()
        with self.Image.open(image_path) as source:
            image = self.ImageOps.exif_transpose(source).convert("RGB")
            messages = [
                {"role": "system", "content": [{"type": "text", "text": prompt["system"]}]},
                {"role": "user", "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": prompt["user"]},
                ]},
            ]
            inputs = self.processor.apply_chat_template(
                messages,
                tokenize=True,
                add_generation_prompt=True,
                enable_thinking=self.config["enable_thinking"],
                return_dict=True,
                return_tensors="pt",
                images_kwargs={"size": {
                    "shortest_edge": self.config["image_min_pixels"],
                    "longest_edge": self.config["image_max_pixels"],
                }},
            )
        device = self.model.device
        if getattr(device, "type", None) == "meta":
            raise RuntimeError("模型输入设备为 meta，请调整 device_map 或显存额度")
        inputs = inputs.to(device)
        input_tokens = int(inputs["input_ids"].shape[-1])
        kwargs: dict[str, Any] = {
            "max_new_tokens": self.config["max_new_tokens"][task],
            "do_sample": self.config["do_sample"],
            "repetition_penalty": self.config["repetition_penalty"],
            "use_cache": True,
        }
        if self.config["do_sample"]:
            kwargs.update({name: self.config[name] for name in ("temperature", "top_p", "top_k")})
        else:
            # 覆盖模型目录中的采样默认值，避免误将采样参数用于贪心基线。
            kwargs.update(temperature=None, top_p=None, top_k=None)
        with self.torch.inference_mode():
            generated = self.model.generate(**inputs, **kwargs)
        token_ids = generated[0][input_tokens:].detach().cpu().tolist()
        eos = self.model.generation_config.eos_token_id
        eos_ids = set(eos if isinstance(eos, list) else [eos]) if eos is not None else set()
        protocol_end_ids = eos_ids.copy()
        pad = self.processor.tokenizer.pad_token_id
        if pad is not None:
            protocol_end_ids.add(pad)
        text_ids = token_ids.copy()
        while text_ids and text_ids[-1] in protocol_end_ids:
            text_ids.pop()
        # 只去末尾传输结束符；保留思考标签、空行、超长、外语和所有格式违规。
        raw_text = self.processor.decode(token_ids, skip_special_tokens=False, clean_up_tokenization_spaces=False)
        text = self.processor.decode(text_ids, skip_special_tokens=False, clean_up_tokenization_spaces=False)
        self._synchronize()
        return {
            "text": text,
            "raw_text": raw_text,
            "raw_output_token_ids": token_ids,
            "input_tokens": input_tokens,
            "output_tokens": len(token_ids),
            "finish_reason": "stop" if token_ids and token_ids[-1] in eos_ids else (
                "length" if len(token_ids) >= kwargs["max_new_tokens"] else "stop"
            ),
            "latency_seconds": round(time.perf_counter() - start, 6),
        }


def _load_runtime(model_dir: Path, config: dict[str, Any]) -> _QwenRuntime:
    return _QwenRuntime(model_dir, config)


def run_generation(
    run_dir: Path,
    model_dir: Path,
    image_root: Path,
    config: dict[str, Any],
    prompts: dict[str, dict[str, Any]],
    resume: bool = True,
) -> dict[str, Any]:
    """两项任务互相独立；成功结果断点续跑，错误下次重试，改配置需新目录。"""
    os.environ.update({
        "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1",
        "HF_HUB_DISABLE_TELEMETRY": "1", "DO_NOT_TRACK": "1",
    })
    run_dir, model_dir, image_root = (Path(p).resolve() for p in (run_dir, model_dir, image_root))
    effective = _effective_config(config)
    if set(prompts) != set(TASKS):
        raise ValueError("必须同时提供 summary 与 interpretation 两套提示词")
    for task in TASKS:
        if any(not isinstance(prompts[task].get(key), str) or not prompts[task][key]
               for key in ("version", "system", "user")):
            raise ValueError(f"{task} 缺少非空 version、system 或 user")
    run = json.loads((run_dir / "run.json").read_text(encoding="utf-8"))
    manifest = read_jsonl(run_dir / "manifest.jsonl")
    identity = _model_identity(model_dir)
    settings = {"config": effective, "model": identity, "software_versions": _software_versions(), "prompts": prompts}
    settings_fingerprint = fingerprint(settings)
    meta_path = run_dir / "generation_meta.json"
    if meta_path.exists():
        old_meta = json.loads(meta_path.read_text(encoding="utf-8"))
        if old_meta.get("settings_fingerprint") != settings_fingerprint:
            raise ValueError("生成配置、提示词、模型或依赖已变化，请使用新的运行目录")
    else:
        atomic_json(meta_path, {
            "schema_version": 1, "created_at": utc_now(), "run_id": run["run_id"],
            "settings_fingerprint": settings_fingerprint, **settings,
        })
    stats: dict[str, Any] = {
        "run_id": run["run_id"], "generated_ok": 0, "generated_error": 0,
        "resumed_ok": 0, "skipped_invalid_images": 0, "settings_fingerprint": settings_fingerprint,
    }
    pending = []
    seen_ids = set()
    # 先检查所有旧结果，拒绝混用结果后才加载权重或生成新内容。
    for record in manifest:
        if record.get("status") != "ok":
            stats["skipped_invalid_images"] += 1
            continue
        image_id = record.get("image_id", "")
        if not isinstance(image_id, str) or not re.fullmatch(r"[A-Za-z0-9_-]+", image_id):
            raise ValueError("清单 image_id 非法，不能安全创建样本目录")
        if image_id in seen_ids:
            raise ValueError("清单存在重复 image_id")
        seen_ids.add(image_id)
        for task in TASKS:
            output_path = run_dir / "samples" / image_id / f"{task}.json"
            if output_path.exists():
                cached = json.loads(output_path.read_text(encoding="utf-8"))
                if cached.get("settings_fingerprint") != settings_fingerprint or cached.get("image_sha256") != record["sha256"]:
                    raise ValueError("已有样本与生成设置或图片内容不一致，请使用新的运行目录")
                if cached.get("run_id") != run["run_id"] or cached.get("image_id") != image_id or cached.get("task") != task:
                    raise ValueError("已有样本身份与运行清单不一致")
                if resume and cached.get("status") == "ok":
                    # 即使跳过生成，也验证原始文件没有被替换。
                    _image_path(image_root, record)
                    stats["resumed_ok"] += 1
                    continue
            pending.append((record, task, output_path))
    if not pending:
        return stats
    runtime = _load_runtime(model_dir, effective)
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    meta["runtime"] = runtime.info
    atomic_json(meta_path, meta)
    for index, (record, task, output_path) in enumerate(pending, 1):
        seed = _stable_seed(effective["seed"], record["image_id"], task)
        result: dict[str, Any] = {
            "schema_version": 1, "run_id": run["run_id"], "image_id": record["image_id"],
            "task": task, "prompt_version": prompts[task]["version"],
            "prompt_sha256": fingerprint(prompts[task]), "image_sha256": record["sha256"],
            "settings_fingerprint": settings_fingerprint, "seed": seed,
            "created_at": utc_now(), "status": "error", "text": "", "raw_text": "",
            "latency_seconds": None, "input_tokens": None, "output_tokens": None,
            "finish_reason": "error", "error": None,
        }
        started = time.perf_counter()
        failed = False
        try:
            path = _image_path(image_root, record)
            result.update(runtime.generate(path, prompts[task], task, seed))
            result["status"] = "ok"
            stats["generated_ok"] += 1
        except Exception as exc:
            # 保存失败，不通过重新提示或修正文案掩盖模型的真实输出。
            result["error"] = {"type": type(exc).__name__, "message": str(exc)}
            result["latency_seconds"] = round(time.perf_counter() - started, 6)
            stats["generated_error"] += 1
            failed = True
        if failed:
            runtime.cleanup_after_error()
        atomic_json(output_path, result)
        print(f"生成 {index}/{len(pending)}：{record['image_id']} / {task} / {result['status']}", flush=True)
    return stats
