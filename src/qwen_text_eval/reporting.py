"""按任务、语种、截图／实拍和场景汇总，保留缺失及不可判定分母。"""

import csv
import io
from collections import defaultdict
from pathlib import Path
from statistics import mean

from .dataset import load_manifest
from .evaluation import load_candidate
from .storage import atomic_json, atomic_text, fingerprint, read_json, text_sha256, write_jsonl
from .validation import TASKS, validate_judgement, validate_reference, validate_rubric


def ratio(numerator, denominator):
    return numerator / denominator if denominator else None


def score_judgement(judgement: dict, reference: dict, rubric: dict) -> dict:
    dimensions = rubric["tasks"][judgement["task"]]["dimensions"]
    applicable = [(info["score"], dimensions[name]["weight"])
                  for name, info in judgement["dimensions"].items() if info["score"] is not None]
    weight = sum(w for _, w in applicable)
    score = sum(value / 4 * w for value, w in applicable) / weight * 100 if weight else None
    claims = [claim for claim in judgement["claims"] if claim["verdict"] != "background"]
    counts = {verdict: sum(c["verdict"] == verdict for c in claims)
              for verdict in ["supported", "contradicted", "unsupported", "unverifiable"]}
    decidable = counts["supported"] + counts["contradicted"] + counts["unsupported"]
    required = {f["fact_id"]: f for f in reference["facts"]
                if f["confidence"] == "high" and judgement["task"] in f["required_for"]}
    required_weight = sum(f["importance"] for f in required.values())
    covered = set(judgement["covered_fact_ids"]) & set(required)
    numeric = [c for c in claims if c.get("category") in {"price", "currency", "quantity", "time"}]
    critical_claims = [c for c in claims if c["severity"] == "critical" and c["verdict"] in {"contradicted", "unsupported"}]
    major_claims = [c for c in claims if c["severity"] in {"major", "critical"} and c["verdict"] in {"contradicted", "unsupported"}]
    return {"quality_score": score, "applicable_weight": weight, "claim_counts": counts,
            "factual_precision": ratio(counts["supported"], decidable),
            "conservative_factual_precision": ratio(counts["supported"], len(claims)),
            "hallucination_rate": ratio(counts["contradicted"] + counts["unsupported"], decidable),
            "unverifiable_rate": ratio(counts["unverifiable"], len(claims)),
            "weighted_key_fact_coverage": ratio(sum(required[fid]["importance"] for fid in covered), required_weight),
            "required_fact_count": len(required), "covered_fact_count": len(covered),
            "numeric_claim_accuracy": ratio(sum(c["verdict"] == "supported" for c in numeric), len(numeric)),
            "numeric_claim_count": len(numeric),
            "critical_error": bool(critical_claims or judgement["critical_errors"]),
            "major_error": bool(major_claims or judgement["critical_errors"]),
            "major_error_claim_count": len(major_claims),
            "dimension_scores": {name: value["score"] for name, value in judgement["dimensions"].items()}}


def mean_metric(rows, field):
    values = [row[field] for row in rows if isinstance(row.get(field), (int, float)) and not isinstance(row[field], bool)]
    return {"mean": mean(values) if values else None, "n": len(values)}


def aggregate(rows: list[dict]) -> dict:
    success = [row for row in rows if row["generation_status"] == "ok"]
    evaluated = [row for row in rows if row.get("judge_status") == "ok"]
    rules = [row for row in rows if row.get("rules_status") == "ok"]
    result = {"expected_pairs": len(rows), "generation_ok": len(success),
              "generation_errors": sum(row["generation_status"] == "error" for row in rows),
              "generation_missing": sum(row["generation_status"] == "missing" for row in rows),
              "invalid_image_pairs": sum(row["generation_status"] == "invalid_image" for row in rows),
              "generation_success_rate": ratio(len(success), len(rows)),
              "judge_evaluated": len(evaluated), "judge_missing": sum(row.get("judge_status") == "missing" for row in success),
              "judge_stale": sum(row.get("judge_status") == "stale" for row in success),
              "judge_coverage_of_generated": ratio(len(evaluated), len(success)),
              "rules_evaluated": len(rules), "rule_review_pairs": sum(row.get("rule_review", False) for row in rules),
              "mechanical_format_pass_rate": ratio(sum(row["mechanical_format_pass"] for row in rules), len(rules)),
              "full_hard_pass_known_count": sum(row.get("hard_pass") is not None for row in rules),
              "full_hard_pass_rate": ratio(sum(row.get("hard_pass") is True for row in rules),
                                            sum(row.get("hard_pass") is not None for row in rules)),
              "critical_error_count": sum(row.get("critical_error", False) for row in evaluated),
              "critical_error_rate": ratio(sum(row.get("critical_error", False) for row in evaluated), len(evaluated)),
              "major_error_count": sum(row.get("major_error", False) for row in evaluated),
              "major_error_rate": ratio(sum(row.get("major_error", False) for row in evaluated), len(evaluated)),
              "human_verified_reference_pairs": sum(row.get("reference_status") == "human_verified" for row in evaluated),
              "needs_human_review": sum(row["needs_human_review"] for row in rows)}
    for field in ["quality_score", "factual_precision", "conservative_factual_precision", "hallucination_rate",
                  "unverifiable_rate", "weighted_key_fact_coverage", "numeric_claim_accuracy", "latency_seconds"]:
        result[field] = mean_metric(rows, field)
    claim_counts = {verdict: sum(row.get("claim_counts", {}).get(verdict, 0) for row in evaluated)
                    for verdict in ["supported", "contradicted", "unsupported", "unverifiable"]}
    known = sum(claim_counts[v] for v in ["supported", "contradicted", "unsupported"])
    result["micro_claim_metrics"] = {"counts": claim_counts, "factual_precision": ratio(claim_counts["supported"], known),
        "hallucination_rate": ratio(claim_counts["contradicted"] + claim_counts["unsupported"], known),
        "unverifiable_rate": ratio(claim_counts["unverifiable"], sum(claim_counts.values()))}
    dimensions = sorted({name for row in evaluated for name in row.get("dimension_scores", {})})
    result["dimensions"] = {name: mean_metric([{"score": row.get("dimension_scores", {}).get(name)} for row in evaluated], "score")
                             for name in dimensions}
    failures = defaultdict(int)
    for row in rules:
        for rule_id in row["failed_rule_ids"]:
            failures[rule_id] += 1
    result["failed_rules"] = dict(sorted(failures.items()))
    return result


def write_csv(path: Path, rows: list[dict], fields: list[str]) -> None:
    output = io.StringIO(newline="")
    writer = csv.DictWriter(output, fieldnames=fields, extrasaction="ignore")
    writer.writeheader()
    for row in rows:
        sanitized = {}
        for field in fields:
            value = row.get(field)
            if isinstance(value, (list, dict)):
                import json
                value = json.dumps(value, ensure_ascii=False)
            if isinstance(value, str) and value.startswith(("=", "+", "-", "@", "\t", "\r")):
                value = "'" + value
            sanitized[field] = value
        writer.writerow(sanitized)
    atomic_text(path, "\ufeff" + output.getvalue())


def build_report(run_dir: Path, rubric: dict) -> dict:
    validate_rubric(rubric)
    system_snapshot = Path(run_dir) / "evaluation_prompts" / "judge_system.txt"
    system_hash = text_sha256(system_snapshot.read_text(encoding="utf-8")) if system_snapshot.exists() else None
    model_snapshot = Path(run_dir) / "judge_meta.json"
    model_identity = read_json(model_snapshot)["judge"] if model_snapshot.exists() else None
    rows = []
    for image in load_manifest(run_dir):
        reference_path = Path(run_dir) / "references" / f"{image['image_id']}.json"
        reference = read_json(reference_path) if reference_path.exists() else None
        if reference:
            validate_reference(reference, image)
        for task in TASKS:
            row = {"image_id": image["image_id"], "relative_path": image["relative_path"], "language": image["language"],
                   "capture_type": image["capture_type"], "scene": reference["scene"] if reference else "unknown", "task": task,
                   "duplicate_of": image["duplicate_of"], "generation_status": "invalid_image" if image["status"] != "ok" else "missing",
                   "rules_status": "missing", "judge_status": "missing", "needs_human_review": False, "review_reasons": [],
                   "reference_status": reference["reference_status"] if reference else "missing", "text": None,
                   "generation_error": image.get("error")}
            candidate = load_candidate(run_dir, image, task) if image["status"] == "ok" else None
            if candidate:
                row.update({"generation_status": candidate["status"], "text": candidate.get("text"),
                            "generation_error": candidate.get("error"), "latency_seconds": candidate.get("latency_seconds"),
                            "output_tokens": candidate.get("output_tokens"), "finish_reason": candidate.get("finish_reason")})
            if candidate and candidate["status"] == "ok":
                if candidate.get("finish_reason") == "length":
                    row["review_reasons"].append("达到生成令牌上限，输出可能截断")
                rules_path = Path(run_dir) / "rules" / image["image_id"] / f"{task}.json"
                if rules_path.exists():
                    rules = read_json(rules_path)
                    if rules["candidate_sha256"] != text_sha256(candidate["text"]) or rules.get("reference_sha256") != (fingerprint(reference) if reference else None):
                        row["rules_status"] = "stale"
                        row["review_reasons"].append("规则结果过期，需要重跑规则测评")
                    else:
                        failed = [c["id"] for c in rules["checks"] if c["status"] == "fail"]
                        row.update({"rules_status": "ok", "hard_pass": rules["hard_pass"], "rule_review": rules["needs_review"],
                                    "failed_rule_ids": failed, "mechanical_format_pass": not failed})
                        if rules["needs_review"]:
                            row["review_reasons"].append("部分提示词约束需要语义复核")
                judge_path = Path(run_dir) / "judgements" / image["image_id"] / f"{task}.json"
                if judge_path.exists() and reference:
                    judgement = read_json(judge_path)
                    try:
                        if system_hash is None or model_identity is None:
                            raise ValueError("缺少评审提示词或模型的运行快照")
                        validate_judgement(judgement, reference, candidate, rubric, system_hash)
                        if judgement["judge"] != model_identity:
                            raise ValueError("评审模型与运行快照不同")
                    except ValueError as error:
                        row["judge_status"] = "stale"
                        row["review_reasons"].append(f"评审结果不可用：{error}")
                    else:
                        row["judge_status"] = "ok"
                        row.update(score_judgement(judgement, reference, rubric))
                        if judgement["needs_human_review"]:
                            row["review_reasons"].extend(judgement["review_reasons"])
                        if row["critical_error"]:
                            row["review_reasons"].append("存在关键事实严重错误")
                        elif row["major_error"]:
                            row["review_reasons"].append("存在价格、主体或其他重要事实错误")
                        if row["required_fact_count"] == 0:
                            row["review_reasons"].append("缺少本任务的高置信关键事实，覆盖率无法评定")
                        if reference["image_quality"]["readability"] != "clear":
                            row["review_reasons"].append("图片存在不可读或部分可读区域")
            row["needs_human_review"] = bool(row["review_reasons"])
            rows.append(row)
    groups = defaultdict(list)
    for row in rows:
        groups[(row["task"], row["language"], row["capture_type"], row["scene"])].append(row)
    grouped = [{"task": key[0], "language": key[1], "capture_type": key[2], "scene": key[3], **aggregate(values)}
               for key, values in sorted(groups.items())]
    tasks = {task: aggregate([row for row in rows if row["task"] == task]) for task in TASKS}
    unique_tasks = {task: aggregate([row for row in rows if row["task"] == task and row["duplicate_of"] is None]) for task in TASKS}
    by_language = [{"task": task, "language": language,
                    **aggregate([row for row in rows if row["task"] == task and row["language"] == language])}
                   for task in TASKS for language in sorted({row["language"] for row in rows})]
    macro_language = {task: {field: mean_metric([{field: group[field]["mean"]} for group in by_language if group["task"] == task], field)
                    for field in ["quality_score", "weighted_key_fact_coverage", "factual_precision"]} for task in TASKS}
    report = {"schema_version": 1, "run_id": read_json(Path(run_dir) / "run.json")["run_id"],
              "rubric_sha256": fingerprint(rubric), "tasks": tasks, "unique_tasks": unique_tasks,
              "judge": model_identity, "judge_prompt_sha256": system_hash,
              "groups": grouped, "by_language": by_language, "macro_language": macro_language,
              "notes": ["所有比例为 0～1；总分为 0～100；各维度分为 0～4。",
                        "未导入评审时语义指标为空，不能当作模型得零分。",
                        "模型参考标注属于草稿；人工核验比例单列，不能把单个评审模型当作客观真值。",
                        "总分按适用维度重新归一化，必须同时查看适用权重、维度分及严重错误。",
                        "主要结果包含全部图片；unique_tasks 剔除相同文件内容的后续副本。"]}
    destination = Path(run_dir) / "reports"
    atomic_json(destination / "aggregate.json", report)
    write_jsonl(destination / "per_sample.jsonl", rows)
    fields = ["image_id", "relative_path", "language", "capture_type", "scene", "task", "duplicate_of", "generation_status",
              "finish_reason", "generation_error", "rules_status", "judge_status", "reference_status", "quality_score",
              "applicable_weight", "factual_precision", "weighted_key_fact_coverage", "hallucination_rate", "numeric_claim_accuracy",
              "required_fact_count", "covered_fact_count", "numeric_claim_count", "claim_counts", "dimension_scores",
              "major_error", "major_error_claim_count", "critical_error", "needs_human_review", "review_reasons", "text"]
    write_csv(destination / "per_sample.csv", rows, fields)
    write_csv(destination / "review.csv", [row for row in rows if row["needs_human_review"] or row["generation_status"] != "ok"], fields)
    lines = ["# 图片文本测评报告", "", "当前只汇总已完成的环节。没有评审回包时，语义质量、事实准确性和覆盖率均显示为未测。", "",
             "| 任务 | 预期条数 | 生成成功 | 已评审 | 质量均分 | 严重错误条数 |", "|---|---:|---:|---:|---:|---:|"]
    for task, label in [("summary", "摘要"), ("interpretation", "解读")]:
        stats = tasks[task]
        score = stats["quality_score"]["mean"]
        display = f"{score:.2f}" if score is not None else "未测"
        lines.append(f"| {label} | {stats['expected_pairs']} | {stats['generation_ok']} | {stats['judge_evaluated']} | {display} | {stats['critical_error_count']} |")
    lines.extend(["", "完整分组、分母和各维度分见 aggregate.json；逐张结果见 per_sample.csv；复核清单见 review.csv。", "",
                  "模型生成的事实参考仍是草稿。价格、币种、过敏原、份量、否定条件等关键事实应优先核验。", ""])
    atomic_text(destination / "report.md", "\n".join(lines))
    return {"path": str(destination), "tasks": tasks}
