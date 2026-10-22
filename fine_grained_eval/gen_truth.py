#!/usr/bin/env python3
"""Image directory -> frozen, machine-reviewed reference JSONL. See PROTOCOL.md."""
from __future__ import annotations

import argparse
import copy
import logging
from pathlib import Path

from common import (HMClient, IMAGE_TYPES, char_count, data_url, ensure_distinct_paths,
                    file_hash, fingerprint, load_api_config, read_json, read_jsonl,
                    utc_now, write_jsonl)
from contracts import (CONFIG_SCHEMA, KNOWLEDGE_SCHEMA, MANIFEST_SCHEMA, REFERENCE_SCHEMA, SUMMARY_REFERENCE_SCHEMA,
                       TASKS, TRUTH_SCHEMA, VERIFY_SCHEMA, VERSION, check_reference, unique_ids, validate)
from prompts import REFERENCE_SYSTEM, SUMMARY_SYSTEM, VERIFY_SYSTEM


def task_config(path=None):
    system = "根据图片和用户请求完成任务。忠实于图片，区分事实与推断，不编造缺失信息。"
    prompts = {"summary": "请概括这张图片的主要内容。", "interpretation": "请你解读一下这张图片。",
               "recommendation": "请推荐一套菜品组合。",
               "phrase": "请推荐一套菜品组合，并给出在当地点餐时能用的表达。"}
    config = read_json(path) if path else {"schema_version": VERSION,
        "context": {"locale": None, "acceptable_languages": []},
        "tasks": {t: {"prompt": {"system": system, "user": u}} for t, u in prompts.items()}}
    validate(config, CONFIG_SCHEMA)
    if not config["tasks"]:
        raise ValueError("task config must enable at least one task")
    return config


def scan_images(root):
    root = Path(root).resolve(strict=True)
    if not root.is_dir():
        raise ValueError("input must be an image directory")
    paths = sorted((p for p in root.rglob("*") if p.is_file() and p.suffix.lower() in IMAGE_TYPES),
                   key=lambda p: p.relative_to(root).as_posix())
    seen = {}
    for path in paths:
        if not path.resolve().is_relative_to(root):
            raise ValueError("image symlink escapes root")
        # Case-folded collision is rejected for portable Windows/Linux name indexing.
        key = path.name.casefold()
        if key in seen:
            raise ValueError(f"duplicate image basename: {seen[key]} / {path}; rename before running")
        seen[key] = path
    if not paths:
        raise ValueError("image directory is empty")
    return paths


def reference_id(row):
    return fingerprint({k: row[k] for k in ("schema_version", "image_name", "image_sha256", "tasks",
                       "context", "provided_evidence", "reference", "summary_references", "reference_length")})


def validate_summary(answer, reference):
    validate(answer, SUMMARY_REFERENCE_SCHEMA)
    ids = {x["id"] for x in reference["summary"]["required_facts"]}
    if set(answer["covered_ids"]) != ids or len(answer["covered_ids"]) != len(ids):
        raise ValueError("reference summary must cover every required fact exactly once")
    evidence = {x["id"] for x in reference["evidence"]}
    if not set(answer["evidence_ids"]) <= evidence:
        raise ValueError("summary references unknown evidence")


def validate_reference_for_input(ref, row):
    check_reference(ref, row["context"])
    supplied = {e["id"]: e for e in row["provided_evidence"]}
    for ev in ref["evidence"]:
        if ev["source"] == "external" and supplied.get(ev["id"]) != ev:
            raise ValueError("external reference must exactly match supplied evidence, not a fabricated source")
    for task in TASKS:
        if task not in row["tasks"] and ref["applicability"][task]["applicable"]:
            raise ValueError("disabled task marked applicable")


def check_truth(row):
    import base64
    import hashlib
    import re
    validate(row, TRUTH_SCHEMA)
    required = {"schema_version", "image_name", "image_path", "image_sha256", "image_data_url", "tasks", "context", "provided_evidence",
                "status", "error", "input_signature", "created_at", "reference", "summary_references",
                "reference_length", "verification", "human_verified", "calls", "reference_id"}
    if not isinstance(row, dict) or set(row) != required or row["schema_version"] != VERSION:
        raise ValueError("invalid reference envelope")
    if not row["image_name"] or any(c in row["image_name"] for c in "/\\"):
        raise ValueError("reference image_name must be basename")
    if Path(row["image_path"]).name != row["image_name"] or Path(row["image_path"]).is_absolute() or ".." in Path(row["image_path"]).parts:
        raise ValueError("invalid image_path")
    match = re.fullmatch(r"data:(image/(?:png|jpeg|webp|bmp));base64,([A-Za-z0-9+/]+={0,2})", row["image_data_url"])
    if not match or hashlib.sha256(base64.b64decode(match[2], validate=True)).hexdigest() != row["image_sha256"]:
        raise ValueError("reference embedded image hash mismatch")
    validate({"schema_version": VERSION, "context": row["context"], "tasks": row["tasks"]}, CONFIG_SCHEMA)
    if type(row["human_verified"]) is not bool:
        raise ValueError("human_verified must be boolean")
    check_provided_evidence(row["provided_evidence"])
    if row["status"] == "error":
        if not isinstance(row["error"], str) or not row["error"]:
            raise ValueError("failed reference needs error")
        return
    if row["status"] != "ok" or row["error"] is not None:
        raise ValueError("invalid reference status")
    validate_reference_for_input(row["reference"], row)
    summaries = row["summary_references"]
    verify = row["verification"]
    validate(verify, VERIFY_SCHEMA)
    if verify["accepted"] is not True:
        raise ValueError("reference must pass independent machine review")
    applicable = row["reference"]["applicability"]["summary"]["applicable"]
    if len(summaries) != (3 if applicable else 0) or len(verify["summary_valid"]) != len(summaries):
        raise ValueError("three independent candidate summaries are required when applicable")
    lengths = []
    for s, valid in zip(summaries, verify["summary_valid"]):
        validate_summary(s, row["reference"])
        if valid:
            lengths.append(char_count(s["text"]))
    if applicable:
        from statistics import median
        if len(lengths) < 2 or row["reference_length"] != median(lengths):
            raise ValueError("reference length must be median of at least two accepted candidates")
    elif row["reference_length"] is not None:
        raise ValueError("inapplicable summary cannot have reference length")
    if row["reference_id"] != reference_id(row):
        raise ValueError("reference content changed without recomputing reference_id")


def check_provided_evidence(items):
    from urllib.parse import urlparse
    unique_ids(items, "provided evidence")
    for ev in items:
        url = urlparse(ev["url"] or "")
        if ev["source"] != "external" or url.scheme not in ("http", "https") or not url.netloc or not ev["version"]:
            raise ValueError("provided evidence needs external source, actual URL and version/date")


def generate(args):
    knowledge_path = getattr(args, "knowledge", None)
    ensure_distinct_paths(args.output, args.api_config, args.task_config, args.manifest, args.export_inputs, knowledge_path)
    root = args.images.resolve(strict=True)
    paths = scan_images(root)
    source_paths = {p.resolve() for p in paths}
    if any(Path(p).resolve() in source_paths for p in (args.output, args.export_inputs) if p is not None):
        raise ValueError("output/export path must not overwrite a source image")
    config = task_config(args.task_config)
    api = load_api_config(args.api_config)
    overrides = {}
    if args.manifest:
        for m in read_jsonl(args.manifest):
            validate(m, MANIFEST_SCHEMA)
            if m["image_name"] in overrides:
                raise ValueError("duplicate manifest image_name")
            overrides[m["image_name"]] = m
        unknown = set(overrides) - {p.name for p in paths}
        if unknown:
            raise ValueError(f"manifest names not found: {sorted(unknown)}")
    knowledge = {}
    if knowledge_path:
        for record in read_jsonl(knowledge_path):
            validate(record, KNOWLEDGE_SCHEMA)
            check_provided_evidence(record["evidence"])
            if record["image_name"] in knowledge or record["image_name"] not in {p.name for p in paths}:
                raise ValueError("duplicate/unknown image_name in knowledge")
            knowledge[record["image_name"]] = record["evidence"]
    planned = []
    for p in paths:
        local = copy.deepcopy(config)
        override = overrides.get(p.name, {})
        if "context" in override:
            local["context"] = override["context"]
        local["tasks"].update(override.get("tasks", {}))
        row = {"schema_version": VERSION, "image_name": p.name, "image_path": p.relative_to(root).as_posix(),
               "image_sha256": file_hash(p), "image_data_url": data_url(p),
               "tasks": local["tasks"], "context": local["context"], "provided_evidence": knowledge.get(p.name, []),
               "status": "error", "error": "not run",
               "input_signature": "", "created_at": utc_now(), "reference": None, "summary_references": [],
               "reference_length": None, "verification": None, "human_verified": False, "calls": [], "reference_id": None}
        row["input_signature"] = fingerprint({"image_sha256": row["image_sha256"], "tasks": row["tasks"],
            "context": row["context"], "provided_evidence": row["provided_evidence"], "api": {k: v for k, v in api.items() if k != "account"},
            "protocol": VERSION, "prompts": [REFERENCE_SYSTEM, SUMMARY_SYSTEM, VERIFY_SYSTEM]})
        planned.append(row)
    if args.export_inputs:
        write_jsonl(args.export_inputs, [{"image_name": r["image_name"], "image_sha256": r["image_sha256"],
            "image_path": r["image_path"], "tasks": r["tasks"], "context": r["context"]} for r in planned])
    if args.prepare_only:
        out = args.output.with_suffix(".requests.jsonl")
        ensure_distinct_paths(out, args.output, args.api_config, args.task_config, args.manifest, args.export_inputs, knowledge_path)
        write_jsonl(out, [make_request(r, "reference", REFERENCE_SYSTEM) for r in planned])
        return {"requests": str(out), "images": len(planned), "live_calls": 0}
    if args.output.exists() and not args.resume:
        raise ValueError("output exists; use --resume or a new output path")
    old = {}
    if args.resume and args.output.exists():
        for row in read_jsonl(args.output):
            check_truth(row)
            if row["image_name"] in old:
                raise ValueError("duplicate image_name in existing truth")
            old[row["image_name"]] = row
        if set(old) - {r["image_name"] for r in planned}:
            raise ValueError("resume input removed images; use a new output")
        for r in planned:
            if r["image_name"] in old and old[r["image_name"]]["input_signature"] != r["input_signature"]:
                raise ValueError("resume signature mismatch; use a new output")
    generator = verifier = None
    rows = []
    for row in planned:
        cached = old.get(row["image_name"])
        if cached and cached["status"] == "ok":
            rows.append(cached)
            continue
        logging.info("reference %s", row["image_name"])
        if generator is None:
            generator, verifier = HMClient(api, "reference"), HMClient(api, "judge")
        try:
            ref, record = generator.call(make_request(row, "reference", REFERENCE_SYSTEM), REFERENCE_SCHEMA,
                lambda a: validate_reference_for_input(a, row))
            row["reference"], row["calls"] = ref, [record]
            if ref["applicability"]["summary"]["applicable"]:
                for n in range(3):
                    request = make_request(row, "summary_reference", SUMMARY_SYSTEM)
                    request["input"].update(reference=ref, independent_candidate_index=n)
                    summary, record = generator.call(request, SUMMARY_REFERENCE_SCHEMA, lambda a: validate_summary(a, ref))
                    row["summary_references"].append(summary)
                    row["calls"].append(record)
            request = make_request(row, "verify", VERIFY_SYSTEM)
            request["input"].update(reference=ref, summary_references=row["summary_references"])
            review, record = verifier.call(request, VERIFY_SCHEMA)
            row["verification"] = review
            row["calls"].append(record)
            if review["accepted"] and len(review["summary_valid"]) == len(row["summary_references"]):
                from statistics import median
                lengths = [char_count(s["text"]) for s, good in zip(row["summary_references"], review["summary_valid"]) if good]
                row["reference_length"] = median(lengths) if len(lengths) >= 2 else None
            row["status"], row["error"] = "ok", None
            row["reference_id"] = reference_id(row)
            check_truth(row)
        except Exception as exc:
            row["status"], row["error"] = "error", str(exc)
            logging.error("reference failed %s: %s", row["image_name"], type(exc).__name__)
        rows.append(row)
        # Preserve all prior records, including not-yet-visited resumable ones, on interruption.
        saved = {r["image_name"]: r for r in old.values()}
        saved.update({r["image_name"]: r for r in rows})
        write_jsonl(args.output, [saved[r["image_name"]] for r in planned if r["image_name"] in saved])
    write_jsonl(args.output, rows)
    return {"output": str(args.output), "images": len(rows), "ok": sum(r["status"] == "ok" for r in rows),
            "errors": sum(r["status"] == "error" for r in rows), "human_verified": False}


def make_request(row, stage, system):
    return {"stage": stage, "system": system, "image_sha256": row["image_sha256"],
            "image_data_url": row["image_data_url"], "input": {"image_name": row["image_name"],
            "task_prompts": row["tasks"], "shared_context": row["context"], "provided_evidence": row["provided_evidence"]}}


def main(argv=None):
    import json
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("images", type=Path)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--api-config", type=Path, required=True)
    p.add_argument("--task-config", type=Path)
    p.add_argument("--manifest", type=Path, help="per-image overrides, JSONL")
    p.add_argument("--knowledge", type=Path, help="optional actual external evidence by image_name, JSONL")
    p.add_argument("--export-inputs", type=Path, help="export prompt/image-only manifest for change.py")
    p.add_argument("--prepare-only", action="store_true")
    p.add_argument("--resume", action="store_true")
    args = p.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    result = generate(args)
    print(json.dumps(result, ensure_ascii=False))
    return 1 if result.get("errors") else 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (ValueError, OSError, RuntimeError, ImportError) as exc:
        raise SystemExit(str(exc))
