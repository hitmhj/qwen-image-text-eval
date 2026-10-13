"""扫描语种／来源文件夹；截图和实拍进入同一批任务。"""

from pathlib import Path
from uuid import uuid4

from .storage import atomic_json, fingerprint, read_json, read_jsonl, sha256_file, utc_now, write_jsonl

EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"}
CAPTURE_ALIASES = {
    "截图": "screenshot", "截屏": "screenshot", "screenshot": "screenshot", "screenshots": "screenshot",
    "实拍": "photo", "拍摄": "photo", "photo": "photo", "photos": "photo", "camera": "photo",
}


def scan_images(root: Path) -> list[dict]:
    from PIL import Image, ImageOps

    root = Path(root).resolve(strict=True)
    if not root.is_dir():
        raise ValueError("图片根路径必须是目录")
    paths = sorted((p for p in root.rglob("*") if p.is_file() and p.suffix.lower() in EXTENSIONS),
                   key=lambda p: p.relative_to(root).as_posix())
    if not paths:
        raise ValueError("没有找到支持的图片，请检查图片路径")
    rows, seen = [], {}
    for path in paths:
        relative = path.relative_to(root).as_posix()
        parts = path.relative_to(root).parts
        row = {"schema_version": 1, "image_id": fingerprint(relative)[:24], "relative_path": relative,
               "language": parts[0] if len(parts) >= 3 else "unknown",
               "capture_type": CAPTURE_ALIASES.get(parts[1].lower(), "unknown") if len(parts) >= 3 else "unknown",
               "source_folder": parts[1] if len(parts) >= 3 else None,
               "sha256": None, "width": None, "height": None, "status": "ok", "error": None,
               "duplicate_of": None}
        try:
            safe_path = path.resolve(strict=True)
            if not safe_path.is_relative_to(root):
                raise ValueError("符号链接指向数据目录外部")
            row["sha256"] = sha256_file(safe_path)
            with Image.open(safe_path) as image:
                image.verify()
            with Image.open(safe_path) as image:
                image.load()
                oriented = ImageOps.exif_transpose(image)
                row["width"], row["height"] = oriented.size
            row["duplicate_of"] = seen.get(row["sha256"])
            seen.setdefault(row["sha256"], row["image_id"])
        except Exception as error:
            row["status"] = "error"
            row["error"] = f"{type(error).__name__}: {error}"
        rows.append(row)
    return rows


def create_manifest(image_root: Path, run_dir: Path, dataset_id: str = "primary") -> dict:
    if dataset_id not in {"primary", "menu_reference"}:
        raise ValueError("数据集标识只能是 primary 或 menu_reference")
    run_dir = Path(run_dir).resolve()
    # 整批结果必须隔离，避免重扫描掩盖图片或参数变化。
    if (run_dir / "run.json").exists() or (run_dir / "manifest.jsonl").exists():
        raise ValueError("该运行目录已有清单；继续生成时直接使用原清单，重新扫描请换一个目录")
    rows = scan_images(image_root)
    if len({row["image_id"] for row in rows}) != len(rows):
        raise ValueError("图片标识冲突，清单未写入")
    metadata = {"schema_version": 2, "run_id": f"{dataset_id}-{uuid4().hex[:12]}", "dataset_id": dataset_id, "created_at": utc_now(),
                "dataset_root": str(Path(image_root).resolve()), "manifest_sha256": fingerprint(rows),
                "image_count": len(rows), "valid_images": sum(r["status"] == "ok" for r in rows),
                "invalid_images": sum(r["status"] != "ok" for r in rows),
                "duplicate_images": sum(r["duplicate_of"] is not None for r in rows),
                "tasks": ["summary", "interpretation", "recommendation"]}
    write_jsonl(run_dir / "manifest.jsonl", rows)
    atomic_json(run_dir / "run.json", metadata)
    return metadata


def load_manifest(run_dir: Path) -> list[dict]:
    rows = read_jsonl(Path(run_dir) / "manifest.jsonl")
    metadata = read_json(Path(run_dir) / "run.json")
    if fingerprint(rows) != metadata["manifest_sha256"]:
        raise ValueError("清单内容发生变化，请使用新的运行目录")
    return rows
