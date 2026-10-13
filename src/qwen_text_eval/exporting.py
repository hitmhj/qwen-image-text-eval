"""导出三任务结果及可单文件跨平台复制的原图交接包。"""

import base64
import hashlib
import io
import json
import os
import tempfile
from pathlib import Path

from .dataset import load_manifest
from .storage import fingerprint, load_generated_result, read_json, read_jsonl, safe_image_path, write_jsonl

TASKS = ["summary", "interpretation", "recommendation"]


def export_generation_results(run_dir: Path, output: Path | None = None) -> dict:
    run_dir = Path(run_dir)
    metadata = read_json(run_dir / "run.json")
    dataset_id = metadata.get("dataset_id", "primary")
    prompts = {}
    for task in TASKS:
        snapshot = run_dir / "prompts" / f"{task}.json"
        if not snapshot.exists():
            raise ValueError("导出三任务结果前需要完整的提示词快照；旧双任务运行请重新生成到新目录")
        prompts[task] = read_json(snapshot)
    rows = []
    for image in load_manifest(run_dir):
        result = {"schema_version": 2, "run_id": metadata["run_id"], "dataset_id": dataset_id,
                  "image_id": image["image_id"], "relative_path": image["relative_path"],
                  "language": image["language"], "capture_type": image["capture_type"],
                  "image_sha256": image["sha256"], "width": image["width"], "height": image["height"],
                  "image_status": image["status"], "image_error": image.get("error"),
                  "duplicate_of": image.get("duplicate_of"), "prompts": prompts,
                  "recommendation_requirement": None}
        for task in TASKS:
            candidate = load_generated_result(run_dir, image, task) if image["status"] == "ok" else None
            if candidate and candidate.get("prompt_sha256") != fingerprint(prompts[task]):
                raise ValueError("候选结果的提示词与运行快照不一致")
            result[task] = candidate
            if task == "recommendation" and candidate:
                result["recommendation_requirement"] = candidate.get("recommendation_requirement")
        rows.append(result)
    target = output or run_dir / "generation_results.jsonl"
    write_jsonl(target, rows)
    return {"path": str(target), "dataset_id": dataset_id, "image_count": len(rows),
            "expected_task_count": len(rows) * 3}


def export_handoff(datasets: list[tuple[str, Path, Path]], output: Path) -> dict:
    """将独立数据集及原始图片字节写入一个原子 JSON，复制此文件即可评审。

    datasets 中每项为 (dataset_id, run_dir, image_root)。不改变原图，也不把
    第一平台的绝对路径作为图片依赖；逐行写入避免同时持有全部图片的 base64。
    """
    from PIL import Image

    if not datasets or len({item[0] for item in datasets}) != len(datasets):
        raise ValueError("交接包需要至少一个数据集，且数据集标识不能重复")
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{output.name}.", dir=output.parent)
    image_count = embedded_count = 0
    summaries = []
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as target:
            target.write('{"schema_version":1,"artifact_type":"qwen_eval_handoff","datasets":[')
            for position, (dataset_id, run_dir, image_root) in enumerate(datasets):
                run_dir = Path(run_dir)
                metadata = read_json(run_dir / "run.json")
                if metadata.get("dataset_id") != dataset_id:
                    raise ValueError("交接包数据集标识与运行记录不一致")
                exported = export_generation_results(run_dir)
                rows = read_jsonl(Path(exported["path"]))
                if len(rows) != metadata["image_count"]:
                    raise ValueError("交接包图片数量与运行记录不一致")
                if position:
                    target.write(",")
                header = {"dataset_id": dataset_id, "run_id": metadata["run_id"]}
                target.write(json.dumps(header, ensure_ascii=False, allow_nan=False)[:-1])
                target.write(',"generation_results":[')
                for index, row in enumerate(rows):
                    if row["dataset_id"] != dataset_id or row["run_id"] != metadata["run_id"]:
                        raise ValueError("交接包结果混入了其他数据集或运行")
                    row["image"] = None
                    if row["image_status"] == "ok":
                        original = safe_image_path(image_root, row["relative_path"])
                        data = original.read_bytes()
                        sha256 = hashlib.sha256(data).hexdigest()
                        if sha256 != row["image_sha256"]:
                            raise ValueError(f"原图与生成时哈希不一致：{row['relative_path']}")
                        with Image.open(io.BytesIO(data)) as image:
                            mime_type = Image.MIME.get(image.format)
                        if not mime_type or not mime_type.startswith("image/"):
                            raise ValueError(f"无法识别图片类型：{row['relative_path']}")
                        row["image"] = {"mime_type": mime_type,
                                        "base64": base64.b64encode(data).decode("ascii"),
                                        "sha256": sha256}
                        embedded_count += 1
                    if index:
                        target.write(",")
                    target.write(json.dumps(row, ensure_ascii=False, allow_nan=False))
                    image_count += 1
                target.write("]}")
                summaries.append({"dataset_id": dataset_id, "image_count": len(rows)})
            target.write("]}\n")
            target.flush()
            os.fsync(target.fileno())
        os.replace(temporary, output)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)
    return {"path": str(output), "datasets": summaries, "image_count": image_count,
            "embedded_image_count": embedded_count, "expected_task_count": image_count * 3}
