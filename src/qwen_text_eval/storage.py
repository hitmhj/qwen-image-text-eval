"""稳定序列化、原子落盘及路径约束。"""

import hashlib
import json
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def fingerprint(obj) -> str:
    encoded = json.dumps(obj, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def text_sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path: Path):
    with Path(path).open(encoding="utf-8-sig") as source:
        return json.load(source)


def load_generated_result(run_dir: Path, manifest: dict, task: str) -> dict | None:
    """读取生成断点；不依赖评审协议或 jsonschema。"""
    path = Path(run_dir) / "samples" / manifest["image_id"] / f"{task}.json"
    if not path.exists():
        return None
    result = read_json(path)
    if not isinstance(result, dict):
        raise ValueError("生成结果必须是 JSON 对象")
    if result.get("image_id") != manifest["image_id"] or result.get("task") != task:
        raise ValueError("生成结果的图片标识或任务不一致")
    if result.get("image_sha256") != manifest["sha256"]:
        raise ValueError("生成结果与清单图片不一致")
    if result.get("status") == "ok" and not isinstance(result.get("text"), str):
        raise ValueError("成功生成的结果必须包含文本")
    return result


def read_jsonl(path: Path) -> list[dict]:
    result = []
    with Path(path).open(encoding="utf-8-sig") as source:
        for number, line in enumerate(source, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"{path.name} 第 {number} 行不是合法 JSON：{error.msg}") from error
            if not isinstance(row, dict):
                raise ValueError(f"{path.name} 第 {number} 行必须是对象")
            result.append(row)
    return result


def atomic_text(path: Path, content: str) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as target:
            target.write(content)
            target.flush()
            os.fsync(target.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def atomic_json(path: Path, obj) -> None:
    atomic_text(path, json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=False) + "\n")


def write_jsonl(path: Path, rows) -> None:
    atomic_text(path, "".join(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n" for row in rows))


def safe_image_path(root: Path, relative_path: str) -> Path:
    relative = Path(relative_path)
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError("图片相对路径不能是绝对路径或包含上级目录")
    resolved_root = Path(root).resolve(strict=True)
    target = (resolved_root / relative).resolve(strict=True)
    if not target.is_relative_to(resolved_root):
        raise ValueError("图片路径超出数据目录")
    if not target.is_file():
        raise ValueError("图片路径不是文件")
    return target
