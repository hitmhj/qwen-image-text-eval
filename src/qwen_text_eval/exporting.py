"""导出每图一行的三任务结果，供另一平台的单文件评审器读取。"""

from pathlib import Path

from .dataset import load_manifest
from .evaluation import load_candidate
from .storage import fingerprint, read_json, write_jsonl

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
            candidate = load_candidate(run_dir, image, task) if image["status"] == "ok" else None
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
