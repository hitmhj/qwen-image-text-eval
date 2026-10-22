from __future__ import annotations

import base64
import hashlib
import importlib
import json
import math
import mimetypes
import os
import re
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path

from contracts import VERSION, validate

IMAGE_TYPES = {".jpg": "image/jpeg", ".jpeg": "image/jpeg", ".png": "image/png", ".webp": "image/webp", ".bmp": "image/bmp"}


def strict_loads(text):
    def finite_float(value):
        parsed = float(value)
        if not math.isfinite(parsed):
            raise ValueError("nonfinite JSON number")
        return parsed
    def pairs(items):
        result = {}
        for k, v in items:
            if k in result:
                raise ValueError(f"duplicate JSON key: {k}")
            result[k] = v
        return result
    return json.loads(text, object_pairs_hook=pairs, parse_float=finite_float,
                      parse_constant=lambda v: (_ for _ in ()).throw(ValueError(f"nonfinite JSON: {v}")))


def read_json(path):
    return strict_loads(Path(path).read_text(encoding="utf-8-sig"))


def read_jsonl(path):
    rows = []
    for n, line in enumerate(Path(path).read_text(encoding="utf-8-sig").splitlines(), 1):
        if not line.strip():
            continue
        try:
            row = strict_loads(line)
            if not isinstance(row, dict):
                raise ValueError("each row must be an object")
            rows.append(row)
        except ValueError as exc:
            raise ValueError(f"{path}:{n}: {exc}") from exc
    return rows


def atomic_text(path, text):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix="." + path.name, dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as f:
            f.write(text)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)


def write_jsonl(path, rows):
    atomic_text(path, "".join(json.dumps(x, ensure_ascii=False, allow_nan=False) + "\n" for x in rows))


def fingerprint(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                     separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def file_hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def char_count(text):
    return sum(not c.isspace() for c in text)


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def safe_image(root, relative, digest=None):
    root = Path(root).resolve(strict=True)
    path = Path(relative)
    if path.is_absolute() or ".." in path.parts:
        raise ValueError("image_path must be a relative path without '..'")
    result = (root / path).resolve(strict=True)
    if not result.is_relative_to(root) or not result.is_file() or result.suffix.lower() not in IMAGE_TYPES:
        raise ValueError("invalid or escaping image path")
    if digest is not None and file_hash(result) != digest:
        raise ValueError(f"image SHA-256 mismatch: {relative}")
    return result


def data_url(path):
    mime = IMAGE_TYPES.get(Path(path).suffix.lower())
    if mime is None:
        raise ValueError("unsupported image format")
    return f"data:{mime};base64," + base64.b64encode(Path(path).read_bytes()).decode("ascii")


def ensure_distinct_paths(*paths):
    values = [Path(p).resolve() for p in paths if p is not None]
    if len(values) != len(set(values)):
        raise ValueError("input/output paths must be different")


def load_api_config(path):
    c = read_json(path)
    allowed = {"account", "sdk_module", "model_agent", "reference_model", "reference_version",
               "judge_model", "judge_version", "timeout", "max_retries", "retry_wait", "request_options"}
    if not isinstance(c, dict) or set(c) - allowed:
        raise ValueError("API config has unknown keys")
    defaults = {"account": "", "sdk_module": "hm_aigc", "model_agent": "yibu",
                "reference_model": "", "reference_version": "", "judge_model": "", "judge_version": "",
                "timeout": 120, "max_retries": 3, "retry_wait": 1, "request_options": {}}
    defaults.update(c)
    for key in ("account", "reference_model", "reference_version", "judge_model", "judge_version"):
        defaults[key] = os.environ.get("FG_EVAL_" + key.upper(), defaults[key])
    for key in ("account", "sdk_module", "model_agent", "reference_model", "reference_version", "judge_model", "judge_version"):
        if not isinstance(defaults[key], str):
            raise ValueError(f"{key} must be a string")
    for key in ("timeout", "max_retries"):
        if type(defaults[key]) is not int or defaults[key] < 1:
            raise ValueError(f"{key} must be a positive integer")
    if type(defaults["retry_wait"]) not in (int, float) or not math.isfinite(defaults["retry_wait"]) or defaults["retry_wait"] < 0:
        raise ValueError("invalid retry_wait")
    if not isinstance(defaults["request_options"], dict) or {"model", "messages"} & defaults["request_options"].keys():
        raise ValueError("request_options cannot override model/messages")
    return defaults


class HMClient:
    """Exactly the supplied hm_aigc.aigc_managed call convention."""
    def __init__(self, config, role):
        self.config, self.role = config, role
        self.model, self.version = config[role + "_model"], config[role + "_version"]
        if not config["account"] or not self.model or not self.version:
            raise ValueError(f"configure account, {role}_model and {role}_version before live API calls")
        module = importlib.import_module(config["sdk_module"])
        factory = getattr(module, "create_client", None)
        self.sdk = factory() if callable(factory) else getattr(module, "hm_aigc", module)
        if not callable(getattr(self.sdk, "aigc_managed", None)):
            raise RuntimeError("SDK must expose aigc_managed, hm_aigc.aigc_managed or create_client()")

    def call(self, request, schema, semantic_validator=None):
        last = None
        for attempt in range(self.config["max_retries"]):
            try:
                body = dict(self.config["request_options"])
                content = [{"type": "text", "text": json.dumps({"input": request["input"],
                            "response_schema": schema}, ensure_ascii=False, allow_nan=False)}]
                content.append({"type": "image_url", "image_url": {"url": request["image_data_url"]}})
                body.update(model=self.model, messages=[{"role": "system", "content": request["system"]},
                                                      {"role": "user", "content": content}])
                started = time.perf_counter()
                result = self.sdk.aigc_managed(model_agent=self.config["model_agent"], req_data=body,
                    sub_account_name=self.config["account"], model=self.model, timeout=self.config["timeout"])
                choice = result["choices"][0]
                if choice.get("finish_reason") == "length":
                    raise ValueError("API output truncated (finish_reason=length)")
                raw = choice["message"]["content"]
                if not isinstance(raw, str):
                    raise ValueError("message.content must be a JSON string")
                fenced = re.fullmatch(r"\s*```(?:json)?\s*(.*?)\s*```\s*", raw, re.S | re.I)
                answer = strict_loads(fenced.group(1) if fenced else raw)
                validate(answer, schema)
                if semantic_validator:
                    semantic_validator(answer)
                record = {"model": self.model, "version": self.version, "role": self.role,
                          "attempts": attempt + 1, "elapsed_seconds": round(time.perf_counter()-started, 4),
                          "finish_reason": choice.get("finish_reason"), "usage": result.get("usage"),
                          "request_sha256": fingerprint({"stage": request["stage"], "input": request["input"],
                                                          "image_sha256": request["image_sha256"],
                                                          "system": request["system"], "schema": schema,
                                                          "model": self.model, "options": self.config["request_options"]})}
                return answer, record
            except Exception as exc:
                last = exc
                if attempt + 1 < self.config["max_retries"]:
                    time.sleep(self.config["retry_wait"] * (attempt + 1))
        raise RuntimeError(f"{request['stage']} API/response validation failed after retries: {type(last).__name__}: {last}") from last
