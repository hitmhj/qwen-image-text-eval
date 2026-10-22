#!/usr/bin/env python3
"""Frozen reference JSONL + strict prediction JSONL -> item-labelled evaluation JSONL."""
from __future__ import annotations

import argparse
import logging
from pathlib import Path

from common import (HMClient, ensure_distinct_paths, fingerprint, load_api_config,
                    read_jsonl, utc_now, write_jsonl)
from contracts import JUDGE_SCHEMAS, TASKS, VERSION, check_prediction
from gen_truth import check_truth
from metrics import calculate, validate_judgement
from prompts import JUDGE_SYSTEM


def key(row):
    return row["image_name"], row["task"], row["model_id"], row["run_id"]


def evaluate(args):
    ensure_distinct_paths(args.truth, args.predictions, args.output, args.api_config)
    if len(args.tasks) != len(set(args.tasks)):
        raise ValueError("duplicate --tasks")
    truths = {}
    for r in read_jsonl(args.truth):
        check_truth(r)
        if r["image_name"] in truths:
            raise ValueError("duplicate reference image_name")
        truths[r["image_name"]] = r
    if not truths:
        raise ValueError("empty truth file")
    predictions = {}
    groups = set()
    for p in read_jsonl(args.predictions):
        check_prediction(p)
        if key(p) in predictions:
            raise ValueError(f"duplicate prediction key {key(p)}")
        if p["image_name"] not in truths:
            raise ValueError(f"prediction image not found in truth: {p['image_name']}")
        t = truths[p["image_name"]]
        if p["image_sha256"] != t["image_sha256"]:
            raise ValueError("prediction/reference image hash mismatch")
        if p["task"] not in t["tasks"] or p["prompt"] != t["tasks"][p["task"]]["prompt"] or p["context"] != t["context"]:
            raise ValueError("prediction prompt/context does not match frozen reference; regenerate truth for the actual input")
        predictions[key(p)] = p
        groups.add((p["model_id"], p["run_id"]))
    if bool(args.model_id) != bool(args.run_id):
        raise ValueError("--model-id and --run-id must be supplied together")
    if args.model_id:
        groups = {(args.model_id, args.run_id)}
    if not groups:
        raise ValueError("no model/run groups; supply --model-id and --run-id to score an entirely missing run")
    api = load_api_config(args.api_config)
    old = {}
    if args.output.exists() and not args.resume and not args.prepare_only:
        raise ValueError("output exists; use --resume or a new output path")
    if args.resume and args.output.exists():
        from gen_html import validate_result
        for row in read_jsonl(args.output):
            validate_result(row)
            if key(row) in old:
                raise ValueError("duplicate cached evaluation key")
            old[key(row)] = row
    planned = []
    requests = []
    for model_id, run_id in sorted(groups):
        for name, truth in truths.items():
            for task in args.tasks:
                if task not in truth["tasks"]:
                    continue
                p = predictions.get((name, task, model_id, run_id))
                paired = predictions.get((name, "recommendation", model_id, run_id)) if task == "phrase" else None
                # Only the same joint generation can supply selections to the phrase judge.
                # A separate recommendation request may have different constraints or stochastic selections.
                if paired and (not p or paired["prompt"] != p["prompt"] or paired["output"] != p["output"]):
                    paired = None
                signature = fingerprint({"truth": truth, "prediction": p, "paired": paired,
                    "judge": {k: v for k, v in api.items() if k != "account"}, "protocol": VERSION,
                    "judge_system": JUDGE_SYSTEM})
                row = {"schema_version": VERSION, "image_name": name, "image_sha256": truth["image_sha256"],
                    "task": task, "model_id": model_id, "run_id": run_id, "created_at": utc_now(),
                    "input_signature": signature, "truth": truth, "prediction": p,
                    "evaluation_status": "pending", "score": None, "details": {}, "judgement": None,
                    "judge_call": None, "error": None, "warnings": [],
                    "paired_recommendation_output": paired["output"] if paired and paired["status"] == "ok" else None}
                cached = old.get(key(row))
                if cached and cached["input_signature"] != signature:
                    raise ValueError("resume signature mismatch; choose a new evaluation output")
                if cached and cached["evaluation_status"] not in ("judge_error", "pending"):
                    planned.append((cached, None))
                    continue
                if truth["status"] == "error":
                    row.update(evaluation_status="reference_error", error=truth["error"])
                elif not truth["reference"]["applicability"][task]["applicable"]:
                    row.update(evaluation_status="not_applicable", error=truth["reference"]["applicability"][task]["reason"])
                elif p is None:
                    row.update(evaluation_status="missing_prediction", score=0.0,
                               error="expected model output is missing; retained in denominator")
                elif p["status"] == "error" or not p["output"].strip():
                    row.update(evaluation_status="model_error", score=0.0, error=p["error"] or "empty output")
                else:
                    if p["generation"]["finish_reason"] == "length":
                        row["warnings"].append("target output reached token limit; full original answer still scored")
                    request = {"stage": "judge", "system": JUDGE_SYSTEM, "image_sha256": truth["image_sha256"],
                        "image_data_url": truth["image_data_url"], "input": {"image_name": name, "task": task,
                        "prompt": p["prompt"], "shared_context": truth["context"], "reference": truth["reference"],
                        "reference_length": truth["reference_length"], "candidate_output": p["output"],
                        "paired_recommendation_output": row["paired_recommendation_output"]}}
                    requests.append(request)
                    planned.append((row, request))
                    continue
                planned.append((row, None))
    planned_keys = {key(r) for r, _ in planned}
    if set(old) - planned_keys:
        raise ValueError("resume task/model/image scope changed; use a new output")
    if args.prepare_only:
        out = args.output.with_suffix(".requests.jsonl")
        ensure_distinct_paths(out, args.output, args.truth, args.predictions, args.api_config)
        write_jsonl(out, requests)
        return {"requests": str(out), "planned_records": len(planned), "live_calls": 0}
    # No SDK initialization is necessary if all cases are failed/missing/not applicable/cached.
    client = HMClient(api, "judge") if requests else None
    rows = []
    for row, request in planned:
        if request:
            logging.info("judge %s %s", row["image_name"], row["task"])
            try:
                judgement, call = client.call(request, JUDGE_SCHEMAS[row["task"]],
                    lambda a: validate_judgement(a, row["truth"], row["task"], row["prediction"]["output"]))
                result = calculate(row["truth"], row["task"], judgement, row["prediction"]["output"])
                row.update(result)
                row.update(evaluation_status="scored", judgement=judgement, judge_call=call)
            except Exception as exc:
                row.update(evaluation_status="judge_error", error=str(exc))
                logging.error("judge failed %s %s: %s", row["image_name"], row["task"], type(exc).__name__)
        rows.append(row)
        saved = dict(old)
        saved.update({key(r): r for r in rows})
        write_jsonl(args.output, [saved[key(r)] for r, _ in planned if key(r) in saved])
    write_jsonl(args.output, rows)
    return {"output": str(args.output), "records": len(rows),
            "scored_or_model_failed": sum(r["score"] is not None for r in rows),
            "pending_reference_or_judge": sum(r["evaluation_status"] in ("reference_error", "judge_error") for r in rows)}


def main(argv=None):
    import json
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("truth", type=Path)
    p.add_argument("predictions", type=Path)
    p.add_argument("--output", required=True, type=Path)
    p.add_argument("--api-config", required=True, type=Path)
    p.add_argument("--tasks", nargs="+", choices=TASKS, default=list(TASKS))
    p.add_argument("--model-id")
    p.add_argument("--run-id")
    p.add_argument("--prepare-only", action="store_true")
    p.add_argument("--resume", action="store_true")
    args = p.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    result = evaluate(args)
    print(json.dumps(result, ensure_ascii=False))
    return 1 if result.get("pending_reference_or_judge") else 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (ValueError, OSError, RuntimeError, ImportError) as exc:
        raise SystemExit(str(exc))
