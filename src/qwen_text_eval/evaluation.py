"""无网络的评审请求准备、回包导入和规则测评。"""

from pathlib import Path

from .dataset import load_manifest
from .storage import atomic_json, atomic_text, fingerprint, read_json, read_jsonl, text_sha256, utc_now, write_jsonl
from .validation import JUDGE_SCHEMA, REFERENCE_SCHEMA, TASKS, validate_judgement, validate_reference, validate_rubric


def sample_path(run_dir: Path, image_id: str, task: str) -> Path:
    return Path(run_dir) / "samples" / image_id / f"{task}.json"


def load_candidate(run_dir: Path, manifest: dict, task: str) -> dict | None:
    path = sample_path(run_dir, manifest["image_id"], task)
    if not path.exists():
        return None
    result = read_json(path)
    if result.get("image_id") != manifest["image_id"] or result.get("task") != task:
        raise ValueError("生成结果的图片标识或任务不一致")
    if result.get("image_sha256") != manifest["sha256"]:
        raise ValueError("生成结果与清单图片不一致")
    if result.get("status") == "ok" and not isinstance(result.get("text"), str):
        raise ValueError("成功生成的结果必须包含文本")
    return result


def frozen_system(run_dir: Path, prompt_path: Path, stage: str) -> tuple[str, Path]:
    system = Path(prompt_path).read_text(encoding="utf-8")
    snapshot = Path(run_dir) / "evaluation_prompts" / f"{stage}_system.txt"
    if snapshot.exists() and snapshot.read_text(encoding="utf-8") != system:
        raise ValueError("标注或评审提示词与运行快照不同，请使用新的运行目录")
    return system, snapshot


def prepare_reference_requests(run_dir: Path, prompt_path: Path) -> dict:
    rows = []
    system, snapshot = frozen_system(run_dir, prompt_path, "reference")
    for image in load_manifest(run_dir):
        if image["status"] != "ok":
            continue
        rows.append({"schema_version": 1, "request_id": f"reference:{image['image_id']}", "stage": "reference",
                     "system": system, "input": {"image_id": image["image_id"], "image_sha256": image["sha256"]},
                     "adapter_required_metadata": ["annotator.model", "annotator.version"],
                     "attachments": [{"kind": "local_image", "relative_path": image["relative_path"], "sha256": image["sha256"]}],
                     "response_schema": REFERENCE_SCHEMA})
    target = Path(run_dir) / "requests" / "reference.jsonl"
    write_jsonl(target, rows)
    atomic_text(snapshot, system)
    return {"request_count": len(rows), "path": str(target), "network_calls": 0}


def import_references(run_dir: Path, input_file: Path, human_verified: bool = False, replace: bool = False) -> dict:
    manifest = {row["image_id"]: row for row in load_manifest(run_dir) if row["status"] == "ok"}
    rows = read_jsonl(input_file)
    pending, seen = [], set()
    for source_row in rows:
        row = dict(source_row)
        image_id = row.get("image_id")
        if image_id not in manifest:
            raise ValueError("参考标注包含清单之外或无法读取的图片")
        if image_id in seen:
            raise ValueError("导入文件存在重复图片的参考标注")
        seen.add(image_id)
        if row.get("reference_status") == "human_verified" and not human_verified:
            raise ValueError("人工核验状态需要操作者显式指定 --human-verified")
        if human_verified:
            row["reference_status"] = "human_verified"
        validate_reference(row, manifest[image_id])
        target = Path(run_dir) / "references" / f"{image_id}.json"
        if target.exists() and read_json(target) != row and not replace:
            raise ValueError("参考标注已存在且内容不同；确认修改后使用 --replace，并重新评审相关文本")
        pending.append((target, row))
    for target, row in pending:
        atomic_json(target, row)
    atomic_json(Path(run_dir) / "references" / "import_meta.json", {"imported_at": utc_now(), "count": len(rows),
                "human_verified_by_operator": human_verified, "source_sha256": fingerprint(rows)})
    return {"imported": len(rows)}


def prepare_judge_requests(run_dir: Path, system_path: Path, rubric: dict, prompts: dict) -> dict:
    validate_rubric(rubric)
    existing = Path(run_dir) / "rubric.json"
    if existing.exists() and read_json(existing) != rubric:
        raise ValueError("本次运行的评分配置已有快照；修改评分方案请另建运行目录")
    system, system_snapshot = frozen_system(run_dir, system_path, "judge")
    rows, skipped = [], {"missing_reference": 0, "missing_or_failed_candidate": 0}
    for image in load_manifest(run_dir):
        if image["status"] != "ok":
            continue
        reference_path = Path(run_dir) / "references" / f"{image['image_id']}.json"
        if not reference_path.exists():
            skipped["missing_reference"] += 2
            continue
        reference = read_json(reference_path)
        validate_reference(reference, image)
        for task in TASKS:
            candidate = load_candidate(run_dir, image, task)
            if not candidate or candidate["status"] != "ok":
                skipped["missing_or_failed_candidate"] += 1
                continue
            # 使用运行时快照，避免后来修改的系统提示词影响旧实验评分。
            task_snapshot = Path(run_dir) / "prompts" / f"{task}.json"
            task_prompt = read_json(task_snapshot) if task_snapshot.exists() else prompts[task]
            if candidate.get("prompt_sha256") != fingerprint(task_prompt):
                raise ValueError("候选文本的提示词摘要与运行快照不一致")
            rows.append({"schema_version": 1, "request_id": f"judge:{image['image_id']}:{task}", "stage": "judge",
                         "adapter_required_metadata": ["judge.model", "judge.version"],
                         "system": system, "input": {"image_id": image["image_id"], "task": task,
                         "candidate_sha256": text_sha256(candidate["text"]), "reference_sha256": fingerprint(reference),
                         "judge_prompt_sha256": text_sha256(system),
                         "candidate_text": candidate["text"], "task_prompt": task_prompt,
                         "reference": reference, "rubric": rubric["tasks"][task], "score_scale": rubric["scale"]},
                         "attachments": [{"kind": "local_image", "relative_path": image["relative_path"], "sha256": image["sha256"]}],
                         "response_schema": JUDGE_SCHEMA})
    target = Path(run_dir) / "requests" / "judge.jsonl"
    write_jsonl(target, rows)
    # 权重也是实验的一部分，变更后不得混合汇总。
    atomic_json(existing, rubric)
    atomic_text(system_snapshot, system)
    return {"request_count": len(rows), "skipped": skipped, "path": str(target), "network_calls": 0}


def import_judgements(run_dir: Path, input_file: Path, rubric: dict, replace: bool = False) -> dict:
    validate_rubric(rubric)
    saved_rubric = Path(run_dir) / "rubric.json"
    if saved_rubric.exists() and read_json(saved_rubric) != rubric:
        raise ValueError("评分配置与运行快照不同，不能混合评审")
    system_snapshot = Path(run_dir) / "evaluation_prompts" / "judge_system.txt"
    if not system_snapshot.exists():
        raise ValueError("导入评审前请先执行 prepare-judge，固定评审提示词和请求")
    system_hash = text_sha256(system_snapshot.read_text(encoding="utf-8"))
    model_snapshot = Path(run_dir) / "judge_meta.json"
    model_identity = read_json(model_snapshot)["judge"] if model_snapshot.exists() else None
    manifest = {row["image_id"]: row for row in load_manifest(run_dir) if row["status"] == "ok"}
    rows = read_jsonl(input_file)
    pending, seen = [], set()
    for row in rows:
        image_id, task = row.get("image_id"), row.get("task")
        if image_id not in manifest or task not in TASKS:
            raise ValueError("评审回包的图片或任务不在清单中")
        key = (image_id, task)
        if key in seen:
            raise ValueError("导入文件包含重复的图片任务评审")
        seen.add(key)
        reference = read_json(Path(run_dir) / "references" / f"{image_id}.json")
        validate_reference(reference, manifest[image_id])
        candidate = load_candidate(run_dir, manifest[image_id], task)
        if not candidate or candidate["status"] != "ok":
            raise ValueError("缺少成功的候选文本，无法导入评审")
        validate_judgement(row, reference, candidate, rubric, system_hash)
        if model_identity is not None and model_identity != row["judge"]:
            raise ValueError("同一运行不能混用不同评审模型或版本")
        model_identity = row["judge"]
        target = Path(run_dir) / "judgements" / image_id / f"{task}.json"
        if target.exists() and read_json(target) != row and not replace:
            raise ValueError("评审已存在且不同，替换时需要 --replace")
        pending.append((target, row))
    for target, row in pending:
        atomic_json(target, row)
    if model_identity is not None:
        atomic_json(model_snapshot, {"judge": model_identity, "judge_prompt_sha256": system_hash})
    if not (Path(run_dir) / "rubric.json").exists():
        atomic_json(Path(run_dir) / "rubric.json", rubric)
    return {"imported": len(rows)}


def evaluate_run_rules(run_dir: Path) -> dict:
    from .rules import evaluate_rules

    rows = []
    missing = 0
    for image in load_manifest(run_dir):
        if image["status"] != "ok":
            continue
        reference_file = Path(run_dir) / "references" / f"{image['image_id']}.json"
        reference = read_json(reference_file) if reference_file.exists() else None
        if reference:
            validate_reference(reference, image)
        for task in TASKS:
            candidate = load_candidate(run_dir, image, task)
            if not candidate or candidate["status"] != "ok":
                missing += 1
                continue
            rules = evaluate_rules(candidate["text"], task, reference["scene"] if reference else None)
            row = {"image_id": image["image_id"], "task": task, "candidate_sha256": text_sha256(candidate["text"]),
                   "reference_sha256": fingerprint(reference) if reference else None, **rules}
            atomic_json(Path(run_dir) / "rules" / image["image_id"] / f"{task}.json", row)
            rows.append(row)
    write_jsonl(Path(run_dir) / "rules" / "details.jsonl", rows)
    return {"evaluated": len(rows), "missing_or_failed": missing}
