"""生成可离线查看的中文测评报告；仅依赖标准库，可嵌入单文件测评脚本。

输入均作为数据处理，模型正文和证据不会执行为 HTML、脚本或链接。
报告不重新计算语义评分，以调用方已校验的汇总和逐样本结果为准。
"""

from __future__ import annotations

import html
import json
import math
import re
from urllib.parse import quote, urlsplit


_TASK_NAMES = {"summary": "摘要", "interpretation": "解读", "recommendation": "点餐推荐"}
_DATASET_NAMES = {"primary": "公司图片测评集", "menu_reference": "独立菜单参考测试集"}
_NAMES = {
    **_TASK_NAMES,
    "menu": "餐厅菜单", "restaurant": "餐厅实景", "beauty": "美妆护肤品",
    "food": "食品饮品包装", "medicine": "药品保健品", "landmark": "地标景点地图",
    "media": "书籍、海报与影视", "document": "文档、通知与票据", "general": "通用",
    "screenshot": "截图", "photo": "实拍", "photograph": "实拍", "camera": "实拍",
    "unknown": "未知", "not_applicable": "不适用", "feasible": "可满足需求",
    "infeasible": "无法满足需求", "missing": "缺失", "error": "失败",
    "invalid_image": "图片不可用", "stale": "结果过期", "ok": "完成",
    "draft": "模型参考草稿", "human_verified": "人工核验参考",
    "high": "高", "medium": "中", "low": "低", "clear": "清晰",
    "partial": "部分可读", "unreadable": "不可读", "supported": "有依据",
    "contradicted": "与证据矛盾", "unsupported": "无图片依据", "unverifiable": "不可判定",
    "background": "背景补充", "critical": "严重", "major": "重要", "minor": "轻微",
    "pass": "通过", "fail": "失败", "review": "待复核", "met": "满足",
    "violated": "违反", "name": "名称", "category": "类别", "price": "价格",
    "currency": "币种", "quantity": "数量与份量", "ingredient": "食材与成分",
    "allergen": "过敏原", "time": "时间", "location": "地点", "restriction": "限制与条件",
    "other": "其他", "factual_accuracy": "事实准确性", "key_information": "关键信息覆盖",
    "translation": "翻译准确性", "numeric_accuracy": "数字与价格准确性",
    "non_hallucination": "无依据内容控制", "readability": "中文可读性",
    "conciseness": "简洁与信息密度", "prompt_following": "提示词遵循",
    "subject_identification": "主体识别", "title_description_complementarity": "标题描述互补",
    "scene_understanding": "场景理解", "structure": "解读结构", "usefulness": "解释与决策价值",
    "requirements": "需求满足", "requirement_satisfaction": "需求满足", "requirement_following": "需求符合程度",
    "budget": "预算与计算", "budget_accuracy": "预算与计算",
    "quality_score": "综合质量分", "factual_precision": "事实精确率",
    "conservative_factual_precision": "保守事实精确率", "hallucination_rate": "无依据率",
    "unsupported_rate": "无依据率", "contradiction_rate": "矛盾率",
    "unverifiable_rate": "不可判定率", "weighted_key_fact_coverage": "加权关键事实覆盖率",
    "numeric_claim_accuracy": "数字陈述准确率", "generation_success_rate": "生成成功率",
    "judge_coverage_of_generated": "已生成文本评审覆盖率", "full_hard_pass_rate": "硬约束通过率",
    "mechanical_format_pass_rate": "确定性格式通过率", "critical_error_rate": "严重错误样本率",
    "major_error_rate": "重要错误样本率", "hard_violation_rate": "明确硬需求违反率",
    "budget_overrun_rate": "确认预算超限率", "confirmed_recommend_success_rate": "确认推荐成功率",
    "confirmed_recommendation_success_rate": "确认推荐成功率", "latency_seconds": "生成用时（秒）",
    "recommendation_success_rate": "可确认推荐成功率", "hard_constraint_unknown_rate": "硬约束未知比例",
    "factual_error_rate": "事实错误率", "conservative_numeric_accuracy": "数字保守支持比例",
    "numeric_unverifiable_rate": "数字未知比例", "valid_input_generation_success_rate": "有效图片生成成功率",
}
_COUNT_NAMES = {
    "expected_pairs": "预期图片任务数", "generation_ok": "生成成功数",
    "generation_errors": "生成失败数", "generation_missing": "生成缺失数",
    "invalid_image_pairs": "不可用图片任务数", "judge_evaluated": "评审完成数",
    "judge_missing": "评审缺失数", "judge_stale": "评审过期数", "judge_errors": "评审失败数",
    "rules_evaluated": "规则检查完成数", "full_hard_pass_known_count": "硬约束可判定数",
    "rule_review_pairs": "规则待复核数", "critical_error_count": "严重错误样本数",
    "major_error_count": "重要错误样本数", "human_verified_reference_pairs": "人工核验参考样本数",
    "needs_human_review": "待人工复核数", "required_fact_count": "高置信必需事实数",
    "covered_fact_count": "已覆盖必需事实数", "numeric_claim_count": "数字陈述数",
    "hard_violation_count": "明确硬需求违反样本数", "budget_overrun_count": "确认预算超限样本数",
    "confirmed_recommend_success_count": "确认推荐成功数",
    "confirmed_recommendation_success_count": "确认推荐成功数",
    "recommendation_feasible_pairs": "参考可行样本数", "recommendation_known_feasibility": "参考可行性可判样本数",
}
_MAIN_METRICS = (
    "quality_score", "factual_precision", "conservative_factual_precision",
    "weighted_key_fact_coverage", "hallucination_rate", "unsupported_rate", "contradiction_rate",
    "unverifiable_rate", "numeric_claim_accuracy", "full_hard_pass_rate", "mechanical_format_pass_rate",
    "hard_violation_rate", "budget_overrun_rate", "confirmed_recommend_success_rate",
    "confirmed_recommendation_success_rate", "critical_error_rate", "major_error_rate",
)


def _escape(value) -> str:
    return html.escape("" if value is None else str(value), quote=True)


def _mapping(value) -> dict:
    return value if isinstance(value, dict) else {}


def _list(value) -> list:
    return value if isinstance(value, (list, tuple)) else []


def _label(value) -> str:
    return _NAMES.get(str(value), str(value)) if value is not None else "未提供"


def _number(value) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def _json_block(value, *, heading: str | None = None) -> str:
    if value is None:
        return '<p class="muted">未提供。</p>'
    body = _escape(json.dumps(value, ensure_ascii=False, indent=2, default=str))
    return (f"<h4>{_escape(heading)}</h4>" if heading else "") + f'<pre class="json">{body}</pre>'


def _badge(value, *, label: str | None = None) -> str:
    tone = "neutral"
    if value in {"ok", "pass", "met", "supported", "human_verified", "high"}:
        tone = "good"
    elif value in {"fail", "error", "violated", "contradicted", "unsupported", "critical", "invalid_image"}:
        tone = "bad"
    elif value in {"review", "draft", "missing", "stale", "unknown", "unverifiable", "major", "low"}:
        tone = "warning"
    return f'<span class="badge {tone}">{_escape(label if label is not None else _label(value))}</span>'


def _metric(value, field: str, *, show_n: bool = True) -> str:
    info = value if isinstance(value, dict) else {"mean": value}
    score = info.get("mean")
    if not _number(score):
        text = '<span class="unmeasured">未测／不可判定</span>'
    elif field == "quality_score":
        text = f"{score:.1f}"
    elif field == "latency_seconds":
        text = f"{score:.2f}"
    elif field.startswith("dimension:"):
        text = f"{score:.2f} / 4"
    else:
        text = f"{score * 100:.1f}%"
    if show_n and "n" in info:
        label = info.get("denominator_label", "有效样本")
        text += f'<small class="denominator">{_escape(label)} {_escape(info["n"])}</small>'
    return text


def _metric_input(stats: dict, field: str):
    value = stats.get(field)
    denominator = _mapping(_mapping(stats.get("metric_denominators")).get(field))
    if not isinstance(value, dict) and denominator:
        return {"mean": value, "n": denominator.get("n"), "denominator_label": denominator.get("label", "有效样本")}
    return value


def _table(headers: list[str], rows: list[list[str]], *, caption: str | None = None) -> str:
    if not rows:
        return '<p class="muted">暂无数据。</p>'
    heads = "".join(f'<th scope="col">{_escape(header)}</th>' for header in headers)
    body = "".join("<tr>" + "".join(f"<td>{cell}</td>" for cell in row) + "</tr>" for row in rows)
    title = f"<caption>{_escape(caption)}</caption>" if caption else ""
    return f'<div class="table-scroll"><table>{title}<thead><tr>{heads}</tr></thead><tbody>{body}</tbody></table></div>'


def _paragraphs(items, *, empty: str = "无。") -> str:
    values = _list(items)
    if not values:
        return f'<p class="muted">{_escape(empty)}</p>'
    return "<ul>" + "".join(f"<li>{_escape(value)}</li>" for value in values) + "</ul>"


def _task_rubric(report: dict, task: str) -> dict:
    rubric = _mapping(report.get("rubric"))
    tasks = _mapping(rubric.get("tasks", rubric))
    config = _mapping(tasks.get(task))
    return _mapping(config.get("dimensions", config))


def _dimension_name(report: dict, task: str, key: str) -> str:
    config = _mapping(_task_rubric(report, task).get(key))
    return str(config.get("name") or _label(key))


def _metric_name(report: dict, key: str) -> str:
    definition = _mapping(_mapping(report.get("metric_definitions")).get(key))
    return str(definition.get("name") or _label(key))


def _metric_keys(report: dict, stats: dict) -> list[str]:
    keys = [key for key in _MAIN_METRICS if key in stats]
    for key in _mapping(report.get("metric_definitions")):
        if key in stats and key not in keys and key not in _COUNT_NAMES:
            keys.append(key)
    return keys


def _image_html(sample: dict) -> str:
    """只预览静态 raster data URL；不请求远程图像或解释任意链接。"""
    data_url = sample.get("image_data_url")
    source = ""
    if isinstance(data_url, str) and re.fullmatch(
        r"data:image/(?:png|jpeg|webp);base64,[A-Za-z0-9+/]+={0,2}", data_url
    ):
        source = f'<img class="sample-image" src="{_escape(data_url)}" alt="原图：{_escape(sample.get("relative_path", ""))}" loading="lazy">'
    else:
        source = '<p class="muted">原图未嵌入；请在测评平台根据文件路径查看原图。</p>'
        if sample.get("image_path"):
            source += '<p class="muted">原图路径：' + _escape(sample["image_path"]) + '</p>'
    link = sample.get("image_link")
    if isinstance(link, str) and link and not re.search(r"[\x00-\x1f\\]", link):
        try:
            parsed = urlsplit(link)
        except ValueError:
            parsed = urlsplit("")
        if (not parsed.scheme and not parsed.netloc and not link.startswith(("/", "#"))
                and not parsed.query and not parsed.fragment
                and parsed.path.lower().endswith((".png", ".jpg", ".jpeg", ".webp"))):
            source += f'<p><a href="{_escape(quote(parsed.path, safe="/._-"))}">查看本地图像</a></p>'
    return f'<figure>{source}<figcaption>{_escape(sample.get("relative_path", "未提供图片路径"))}</figcaption></figure>'


def _summary_card(report: dict, task: str, stats: dict, samples: list[dict]) -> str:
    metrics = []
    for key in _metric_keys(report, stats):
        if key != "quality_score":
            metrics.append(f'<div><dt>{_escape(_metric_name(report, key))}</dt><dd>{_metric(_metric_input(stats, key), key)}</dd></div>')
    count_fields = [key for key in _COUNT_NAMES if key in stats]
    counts = "".join(f'<div><dt>{_escape(_COUNT_NAMES[key])}</dt><dd>{_escape(stats[key])}</dd></div>' for key in count_fields)
    weights = [row.get("applicable_weight") for row in samples if row.get("task") == task and _number(row.get("applicable_weight"))]
    if weights:
        weight_text = f"适用权重：{min(weights):g}～{max(weights):g}；有效样本 {len(weights)}。"
    else:
        weight_text = "适用权重尚未测量。"
    score = _metric(stats.get("quality_score"), "quality_score")
    numeric_score = _mapping(stats.get("quality_score")).get("mean")
    chart = ""
    if _number(numeric_score):
        width = max(0, min(100, numeric_score))
        chart = f'<div class="score-track" role="img" aria-label="质量均分 {_escape(numeric_score)}，满分100"><span style="width:{width:.3f}%"></span></div>'
    dimensions = []
    for key, info in _mapping(stats.get("dimensions")).items():
        dimensions.append([_escape(_dimension_name(report, task, key)), _metric(info, "dimension:" + key)])
    micro = _mapping(stats.get("micro_claim_metrics"))
    micro_html = ""
    if micro:
        micro_rows = [[_escape(_label(key)), _escape(value)] for key, value in _mapping(micro.get("counts")).items()]
        for key, value in micro.items():
            if key != "counts" and key in _NAMES:
                micro_rows.append([_escape(_label(key)), _metric(value, key, show_n=False)])
        micro_html = '<details><summary>合并陈述后的微观统计</summary>' + _table(["指标", "数值"], micro_rows) + '</details>'
    failures = _mapping(stats.get("failed_rules"))
    failure_html = ""
    if failures:
        failure_html = '<details><summary>格式规则失败分布</summary>' + _table(["规则标识", "失败样本数"], [[_escape(key), _escape(value)] for key, value in failures.items()]) + '</details>'
    return (f'<section class="task-card"><h3>{_escape(_label(task))}</h3>'
            f'<div class="score">{score}<span class="score-unit"> / 100</span></div>{chart}'
            f'<p class="muted">{_escape(weight_text)}</p><dl class="count-grid">{counts}</dl>'
            f'<dl class="metric-grid">{"".join(metrics)}</dl>'
            + _table(["评分维度", "平均分与有效样本"], dimensions)
            + micro_html + failure_html + '</section>')


def _group_table(report: dict, groups, *, detailed: bool) -> str:
    headers = ["任务", "语种"] + (["来源", "场景"] if detailed else [])
    headers += ["预期数", "生成成功", "已评审", "质量均分", "事实精确率", "关键事实覆盖", "严重错误数"]
    rows = []
    for group in _list(groups):
        if not isinstance(group, dict):
            continue
        row = [_escape(_label(group.get("task"))), _escape(group.get("language", "未知"))]
        if detailed:
            row += [_escape(_label(group.get("capture_type"))), _escape(_label(group.get("scene")))]
        row += [_escape(group.get("expected_pairs", "未提供")), _escape(group.get("generation_ok", "未提供")),
                _escape(group.get("judge_evaluated", "未提供")), _metric(group.get("quality_score"), "quality_score"),
                _metric(group.get("factual_precision"), "factual_precision"),
                _metric(group.get("weighted_key_fact_coverage"), "weighted_key_fact_coverage"),
                _escape(group.get("critical_error_count", "未提供"))]
        rows.append(row)
    return _table(headers, rows)


def _reference_html(sample: dict) -> str:
    reference = _mapping(sample.get("reference"))
    judgement = _mapping(sample.get("judgement"))
    if not reference:
        return '<p class="unmeasured">参考事实缺失，事实准确性与覆盖率尚不可判定。</p>'
    quality = _mapping(reference.get("image_quality"))
    intro = f'<p>参考状态：{_badge(reference.get("reference_status", sample.get("reference_status", "missing")))}'
    if quality:
        intro += f'　图像可读性：{_badge(quality.get("readability", "unknown"))}'
    intro += '</p>'
    if quality.get("issues"):
        intro += _paragraphs(quality["issues"])
    if reference.get("unreadable_regions"):
        intro += '<h4>不可读区域</h4>' + _paragraphs(reference["unreadable_regions"])
    rows = []
    covered = set(_list(judgement.get("covered_fact_ids")))
    judge_ok = sample.get("judge_status") == "ok"
    for fact in _list(reference.get("facts")):
        if not isinstance(fact, dict):
            continue
        fid = fact.get("fact_id", "")
        required = sample.get("task") in _list(fact.get("required_for")) and fact.get("confidence") == "high"
        coverage = "非本任务必需"
        if required:
            coverage = "已覆盖" if judge_ok and fid in covered else "遗漏" if judge_ok else "覆盖情况未测"
        evidence = _escape(fact.get("evidence", "未提供定位"))
        if fact.get("numeric") is not None:
            evidence += '<details><summary>数字与绑定条件</summary>' + _json_block(fact["numeric"]) + '</details>'
        rows.append([_escape(fid), _escape(_label(fact.get("category"))),
                     _escape(fact.get("normalized_zh", fact.get("text", ""))), _escape(fact.get("source_text", "")),
                     evidence, _badge(fact.get("confidence", "unknown")), _escape(fact.get("importance", "未提供")),
                     _escape("、".join(_label(task) for task in _list(fact.get("required_for")))), _escape(coverage)])
    intro += _table(["事实编号", "类别", "中文事实", "原文", "原图证据与定位", "置信度", "重要权重", "必需任务", "本任务覆盖"], rows)
    feasibility = reference.get("recommendation_feasibility")
    if feasibility is not None and sample.get("task") == "recommendation":
        intro += '<h4>需求可行性参考</h4>'
        intro += _json_block(feasibility) if isinstance(feasibility, (dict, list)) else _badge(feasibility)
    return intro


def _judgement_html(report: dict, sample: dict) -> str:
    judgement = _mapping(sample.get("judgement"))
    status = sample.get("judge_status", "missing")
    intro = f'<p>评审状态：{_badge(status)}</p>'
    if status != "ok":
        intro += '<p class="unmeasured">以下未完成、失败或过期的评审内容不进入有效语义指标。</p>'
    if not judgement:
        return intro + '<p class="muted">尚无评审回包。</p>'
    dimensions = []
    for key, value in _mapping(judgement.get("dimensions")).items():
        info = _mapping(value)
        score = info.get("score")
        score_text = f"{score} / 4" if _number(score) else "不适用／证据不足"
        dimensions.append([_escape(_dimension_name(report, str(sample.get("task")), key)), _escape(score_text),
                           _escape(info.get("reason", "未提供理由")),
                           _escape("、".join(str(fid) for fid in _list(info.get("evidence_fact_ids"))))])
    intro += '<h4>评分维度与理由</h4>' + _table(["维度", "得分", "评分理由", "证据事实编号"], dimensions)
    claims = []
    for claim in _list(judgement.get("claims")):
        if not isinstance(claim, dict):
            continue
        claims.append([_escape(claim.get("claim_id", "")), _escape(claim.get("text", "")),
                       _escape(_label(claim.get("category"))), _badge(claim.get("verdict", "unverifiable")),
                       _badge(claim.get("severity", "unknown")),
                       _escape("、".join(str(fid) for fid in _list(claim.get("fact_ids")))),
                       _escape(claim.get("explanation", "未提供理由"))])
    intro += '<h4>逐条陈述核对</h4>' + _table(["陈述编号", "模型陈述", "类别", "判定", "错误等级", "证据事实编号", "核对理由"], claims)
    if sample.get("task") == "recommendation":
        recommendation = _mapping(judgement.get("recommendation"))
        checks = []
        for check in _list(recommendation.get("requirement_checks", judgement.get("requirement_checks"))):
            if isinstance(check, dict):
                checks.append([_escape(check.get("constraint_id", "")), _badge(check.get("status", "unknown")),
                               _escape(check.get("reason", "未提供理由")),
                               _escape("、".join(str(fid) for fid in _list(check.get("evidence_fact_ids"))))])
        intro += '<h4>点餐需求逐项检查</h4>' + _table(["需求编号", "是否满足", "理由", "证据事实编号"], checks)
        intro += '<h4>推荐方案、数量与价格核算</h4>' + _json_block(recommendation.get("plan", judgement.get("recommendation_plan")))
        if sample.get("budget_check"):
            intro += '<h4>程序独立复核费用与预算</h4>' + _json_block(sample["budget_check"])
    if judgement.get("critical_errors"):
        intro += '<div class="error-box"><h4>已确认严重错误</h4>' + _paragraphs(judgement["critical_errors"]) + '</div>'
    if judgement.get("review_reasons"):
        intro += '<h4>评审提出的复核理由</h4>' + _paragraphs(judgement["review_reasons"])
    return intro


def _rules_html(sample: dict) -> str:
    rules = _mapping(sample.get("rules"))
    if not rules:
        return '<p class="unmeasured">规则检查尚未完成。</p>'
    hard_pass = rules.get("hard_pass")
    hard_label = "全部硬约束通过" if hard_pass is True else "存在硬约束失败" if hard_pass is False else "部分硬约束待语义复核"
    status = "pass" if hard_pass is True else "fail" if hard_pass is False else "review"
    intro = f'<p>{_badge(status, label=hard_label)}'
    if "char_count" in rules:
        intro += f'　有效字符数：{_escape(rules["char_count"])}'
    intro += '</p>'
    rows = []
    for check in _list(rules.get("checks")):
        if isinstance(check, dict):
            observed = json.dumps(check.get("observed"), ensure_ascii=False, default=str)
            expected = json.dumps(check.get("expected"), ensure_ascii=False, default=str)
            rows.append([_escape(check.get("name", check.get("id", ""))), _badge(check.get("status", "review")),
                         _escape(observed), _escape(expected), _escape(check.get("details", ""))])
    return intro + _table(["规则", "结果", "实际值", "要求", "说明"], rows)


def _sample_html(report: dict, sample: dict, index: int) -> str:
    task = str(sample.get("task", "unknown"))
    generation = sample.get("generation_status", "missing")
    judge = sample.get("judge_status", "missing")
    problem = bool(sample.get("needs_human_review") or sample.get("critical_error") or sample.get("major_error")
                   or generation != "ok" or judge != "ok" or sample.get("hard_pass") is False
                   or _mapping(sample.get("rules")).get("hard_pass") is False)
    attributes = (f'data-task="{_escape(task)}" data-language="{_escape(sample.get("language", "未知"))}" '
                  f'data-capture="{_escape(sample.get("capture_type", "unknown"))}" data-problem="{int(problem)}"')
    badges = _badge(generation, label="生成：" + _label(generation)) + _badge(judge, label="评审：" + _label(judge))
    if sample.get("critical_error"):
        badges += _badge("critical", label="严重错误")
    elif sample.get("major_error"):
        badges += _badge("major", label="重要错误")
    if sample.get("needs_human_review"):
        badges += _badge("review", label="待人工复核")
    subject = sample.get("relative_path") or sample.get("image_id") or f"样本 {index}"
    summary = (f'<summary><span class="sample-title">{index:03d} · {_escape(_label(task))} · {_escape(subject)}</span>'
               f'<span class="sample-score">{_metric(sample.get("quality_score"), "quality_score", show_n=False)}</span>'
               f'<span class="status-group">{badges}</span></summary>')
    meta = ["图片编号：" + str(sample.get("image_id", "未提供")),
            "语种：" + str(sample.get("language", "未知")), "来源：" + _label(sample.get("capture_type")),
            "场景：" + _label(sample.get("scene")), "参考：" + _label(sample.get("reference_status", "missing"))]
    prompt_version = sample.get("prompt_version", sample.get("candidate_prompt_version"))
    if prompt_version is not None:
        meta.append("提示词版本：" + str(prompt_version))
    if sample.get("duplicate_of"):
        meta.append("重复图片原件编号：" + str(sample["duplicate_of"]))
    text = sample.get("text")
    candidate = '<p class="unmeasured">生成正文缺失。</p>' if text is None else f'<pre class="candidate">{_escape(text)}</pre>'
    content = '<div class="sample-body"><p class="sample-meta">' + '　|　'.join(_escape(item) for item in meta) + '</p>'
    content += '<div class="image-text-grid">' + _image_html(sample) + '<section><h4>被测模型原始正文</h4>' + candidate + '</section></div>'
    if task == "recommendation":
        content += '<h4>本样本的点餐需求</h4>' + _json_block(sample.get("recommendation_requirement"))
    individual = []
    if "applicable_weight" in sample:
        individual.append(["适用维度权重", _escape(sample["applicable_weight"] if sample["applicable_weight"] is not None else "未测")])
    for key in _metric_keys(report, sample):
        if key != "quality_score":
            individual.append([_escape(_metric_name(report, key)), _metric(sample[key], key, show_n=False)])
    for key in ("required_fact_count", "covered_fact_count", "numeric_claim_count"):
        if key in sample:
            individual.append([_escape(_COUNT_NAMES[key]), _escape(sample[key])])
    if individual:
        content += '<h4>样本指标与分母</h4>' + _table(["指标", "数值"], individual)
    errors = [f"{label}：{sample[key]}" for key, label in (("generation_error", "生成失败原因"), ("judge_error", "评审失败原因"), ("reference_error", "参考失败原因")) if sample.get(key)]
    reasons = errors + _list(sample.get("review_reasons"))
    if reasons:
        content += '<div class="review-box"><h4>问题与复核事项</h4>' + _paragraphs(reasons) + '</div>'
    content += '<details class="subdetail" open><summary>独立参考事实及图像证据</summary>' + _reference_html(sample) + '</details>'
    content += '<details class="subdetail" open><summary>大模型评审详情</summary>' + _judgement_html(report, sample) + '</details>'
    content += '<details class="subdetail"><summary>确定性规则检查</summary>' + _rules_html(sample) + '</details></div>'
    return f'<details class="sample-card" {attributes} id="sample-{index}">{summary}{content}</details>'


def _definitions_html(report: dict) -> str:
    rows = []
    for key, value in _mapping(report.get("metric_definitions")).items():
        info = _mapping(value)
        description = info.get("definition", info.get("description", value if isinstance(value, str) else ""))
        if info.get("denominator"):
            description = str(description) + " 分母：" + str(info["denominator"])
        rows.append([_escape(info.get("name", _label(key))), _escape(description)])
    return _table(["指标", "定义与分母"], rows)


def _rubric_html(report: dict) -> str:
    rows = []
    tasks = _mapping(report.get("tasks"))
    for task in tasks:
        for key, value in _task_rubric(report, task).items():
            info = _mapping(value)
            rows.append([_escape(_label(task)), _escape(info.get("name", _label(key))),
                         _escape(info.get("weight", "未提供")),
                         _escape(info.get("description", info.get("definition", "")))])
    body = _table(["任务", "维度", "权重", "评分范围与判断依据"], rows)
    rubric = _mapping(report.get("rubric"))
    anchors = _mapping(_mapping(rubric.get("scale")).get("anchors"))
    if anchors:
        body += _table(["分数", "统一评分锚点"], [[_escape(score), _escape(text)] for score, text in anchors.items()])
    return body


_CSS = """
:root{color-scheme:light;--ink:#173042;--muted:#5a6c77;--line:#dbe4e9;--bg:#f4f7f9;--accent:#147a80;--bad:#a62536;--warn:#795400}
*{box-sizing:border-box}body{margin:0;color:var(--ink);background:var(--bg);font:15px/1.7 system-ui,-apple-system,"Segoe UI","Microsoft YaHei",sans-serif}main{max-width:1440px;margin:auto;padding:32px 28px 64px}h1{font-size:30px;line-height:1.35;margin:0 0 12px}h2{font-size:22px;margin:0 0 18px}h3{font-size:20px;margin:0 0 12px}h4{font-size:15px;margin:20px 0 8px}p{margin:8px 0 14px}a{color:var(--accent)}header{padding:12px 0 20px}.eyebrow{letter-spacing:.12em;color:var(--accent);font-weight:700;font-size:13px}.panel,.task-card{background:#fff;border:1px solid var(--line);border-radius:16px;padding:24px;margin-bottom:24px}.task-grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(min(100%,350px),1fr));gap:20px}.task-card{min-width:0}.score{font-size:38px;font-weight:750;line-height:1.4;margin-bottom:8px}.score-unit{font-size:14px;color:var(--muted);font-weight:400}.score .unmeasured{font-size:22px}.score-track{height:7px;border-radius:9px;background:#e7eef1;overflow:hidden}.score-track span{display:block;height:100%;background:var(--accent)}.muted,.denominator{color:var(--muted)}.denominator{display:block;font-size:12px;white-space:nowrap}.count-grid,.metric-grid{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:10px 16px;margin:18px 0}.count-grid div,.metric-grid div{padding:9px 0;border-bottom:1px solid #eaf0f3}dt{font-size:13px;color:var(--muted)}dd{margin:1px 0 0;font-size:17px;font-weight:650}.metric-grid dd{font-size:16px}.metric-grid .unmeasured{font-size:13px;font-weight:400}.unmeasured{color:var(--muted);font-weight:400}.table-scroll{overflow:auto;margin:12px 0;max-width:100%}table{border-collapse:collapse;width:100%;font-size:13px}th,td{text-align:left;vertical-align:top;padding:11px 12px;border-bottom:1px solid var(--line);min-width:70px;white-space:pre-wrap;overflow-wrap:anywhere}th{background:#edf3f5;font-weight:650;white-space:nowrap}caption{font-weight:650;text-align:left;padding:8px 0}.badge{display:inline-block;font-size:12px;font-weight:600;line-height:1.5;padding:3px 8px;border-radius:6px;background:#edf1f4;color:#4c616f;white-space:nowrap;margin:2px 3px 2px 0}.good{background:#e5f4ee;color:#196d4c}.bad{background:#ffe8eb;color:var(--bad)}.warning{background:#fff2d3;color:var(--warn)}.notice{padding:14px 18px;border-left:4px solid var(--accent);background:#eaf4f4;border-radius:6px}.review-box,.error-box{background:#fff4df;padding:12px 18px;border-radius:8px;margin:18px 0}.error-box{background:#fff0f2}.review-box h4,.error-box h4{margin-top:0}ul{padding-left:22px}details>summary{cursor:pointer;font-weight:650;padding:12px 0}details>summary:hover{color:var(--accent)}details>summary:focus-visible{outline:3px solid #94cbd0;outline-offset:4px}.filters{display:flex;flex-wrap:wrap;align-items:end;gap:14px;padding:18px;background:#edf3f5;border-radius:10px;margin:16px 0}.filter-field{display:flex;flex-direction:column;gap:4px;font-size:13px}.filter-field.search{flex:1;min-width:180px}select,input[type=search]{border:1px solid #b9c9d2;border-radius:6px;padding:9px 10px;font:inherit;color:var(--ink);background:#fff;min-height:40px}.filter-check{font-size:13px;display:flex;align-items:center;gap:7px;padding-bottom:9px}button{cursor:pointer;background:#fff;border:1px solid #b9c9d2;border-radius:6px;padding:9px 12px;color:var(--ink);font:inherit;min-height:40px}button:hover{border-color:var(--accent)}.sample-card{border:1px solid var(--line);border-radius:10px;background:white;margin-bottom:12px}.sample-card>summary{display:flex;flex-wrap:wrap;gap:10px 16px;align-items:center;padding:16px 18px}.sample-title{flex:1;min-width:240px;font-size:14px;overflow-wrap:anywhere}.sample-score{font-size:18px;font-weight:750;white-space:nowrap}.sample-score .unmeasured{font-size:13px}.status-group{display:flex;flex-wrap:wrap}.sample-body{padding:0 20px 24px}.sample-meta{font-size:12px;color:var(--muted);overflow-wrap:anywhere}.image-text-grid{display:grid;grid-template-columns:minmax(220px,.65fr) minmax(260px,1.35fr);gap:22px;align-items:start}.image-text-grid h4{margin-top:0}figure{margin:0}figcaption{font-size:12px;color:var(--muted);overflow-wrap:anywhere}.sample-image{width:100%;max-height:500px;object-fit:contain;border-radius:6px;background:#f1f4f5}pre{font:13px/1.8 "Microsoft YaHei",system-ui,sans-serif;white-space:pre-wrap;overflow-wrap:anywhere;background:#f5f8fa;border:1px solid var(--line);border-radius:8px;padding:16px;margin:8px 0}.candidate{font-size:14px;line-height:1.9}.json{font:12px/1.7 ui-monospace,"Cascadia Code",Consolas,monospace}.subdetail{border-top:1px solid var(--line);margin-top:20px}.filter-count{font-size:13px;color:var(--muted)}.hidden{display:none!important}.no-matches{padding:20px;text-align:center;color:var(--muted)}footer{font-size:12px;color:var(--muted);margin-top:24px}.meta-line{font-size:13px;color:var(--muted);overflow-wrap:anywhere}.stat-notes{font-size:13px}.methodology{font-size:14px}.dataset-badge{background:#dff0f1;color:#11626b;font-size:13px;padding:5px 10px;border-radius:8px;display:inline-block}
@media(max-width:760px){main{padding:20px 12px 36px}.panel,.task-card{padding:18px;border-radius:12px}h1{font-size:25px}.image-text-grid{grid-template-columns:1fr}.sample-body{padding:0 14px 20px}.sample-card>summary{padding:14px}.filters{padding:12px}.filter-field{flex:1}.sample-title{min-width:180px}}
@media print{body{background:white;font-size:11px}main{max-width:none;padding:0}.panel,.task-card,.sample-card{box-shadow:none;border-radius:0}.filters,.filter-count,.no-matches{display:none}.task-grid{display:block}.task-card{break-inside:avoid}.sample-card{break-before:auto}.sample-image{max-height:300px}summary{list-style:none}table{font-size:10px}th,td{padding:5px}.table-scroll{overflow:visible}pre{font-size:10px}h2{break-after:avoid}details::details-content{content-visibility:visible!important;display:block!important}.hidden{display:none!important}}
"""

_JS = """
(()=>{
 'use strict';
 const cards=Array.from(document.querySelectorAll('.sample-card'));
 const search=document.getElementById('filter-search');
 const task=document.getElementById('filter-task');
 const language=document.getElementById('filter-language');
 const capture=document.getElementById('filter-capture');
 const problems=document.getElementById('filter-problems');
 const count=document.getElementById('filter-count');
 const empty=document.getElementById('no-matches');
 function apply(){
  const needle=search.value.trim().toLocaleLowerCase();let visible=0;
  cards.forEach(card=>{
   const matches=(!task.value||card.dataset.task===task.value)&&(!language.value||card.dataset.language===language.value)&&(!capture.value||card.dataset.capture===capture.value)&&(!problems.checked||card.dataset.problem==='1')&&(!needle||card.textContent.toLocaleLowerCase().includes(needle));
   card.classList.toggle('hidden',!matches);if(matches)visible++;
  });
  count.textContent='显示 '+visible+' / '+cards.length+' 条图片任务结果';empty.classList.toggle('hidden',visible!==0);
 }
 [search,task,language,capture,problems].forEach(control=>control.addEventListener(control===search?'input':'change',apply));
 document.getElementById('filter-reset').addEventListener('click',()=>{search.value='';task.value='';language.value='';capture.value='';problems.checked=false;apply()});
 document.getElementById('expand-all').addEventListener('click',()=>cards.filter(card=>!card.classList.contains('hidden')).forEach(card=>{card.open=true}));
 document.getElementById('collapse-all').addEventListener('click',()=>cards.forEach(card=>{card.open=false}));
 let printState=[];
 window.addEventListener('beforeprint',()=>{printState=Array.from(document.querySelectorAll('details')).map(item=>[item,item.open]);printState.forEach(([item])=>{item.open=true})});
 window.addEventListener('afterprint',()=>{printState.forEach(([item,opened])=>{item.open=opened});printState=[]});
 apply();
})();
"""


def render_html(report: dict, samples: list[dict], *, title: str = "图片文本测评报告") -> str:
    """渲染独立数据集的汇总及逐图片、逐任务证据；不访问网络或本地文件。"""
    report = _mapping(report)
    samples = [sample for sample in _list(samples) if isinstance(sample, dict)]
    dataset_id = report.get("dataset_id", "未标记")
    dataset_name = report.get("dataset_name", _DATASET_NAMES.get(str(dataset_id), str(dataset_id)))
    tasks = _mapping(report.get("tasks"))
    task_order = [key for key in _TASK_NAMES if key in tasks] + [key for key in tasks if key not in _TASK_NAMES]
    cards = "".join(_summary_card(report, task, _mapping(tasks[task]), samples) for task in task_order)
    sections = [f'<header><p class="eyebrow">图像理解 · 文本质量测评</p><h1>{_escape(title)}</h1>'
                f'<span class="dataset-badge">{_escape(dataset_name)}</span>'
                f'<p class="meta-line">数据集标识：{_escape(dataset_id)}'
                f'　运行标识：{_escape(report.get("run_id", "未提供"))}'
                f'　报告时间：{_escape(report.get("created_at", report.get("generated_at", "未提供")))}</p>'
                '<p>摘要、解读与点餐推荐分别报告。此报告仅包含当前数据集，独立菜单参考测试集与公司图片测评集不混合汇总。</p></header>']
    sections.append('<section class="panel methodology"><h2>如何理解本次结果</h2>'
                    '<p>参考构造阶段独立查看原图，不读取被测文本；评分阶段逐条核对模型陈述、参考事实与原图证据。模型参考属于草稿，人工核验状态单独展示。</p>'
                    '<div class="notice">未测、失败、缺失、结果过期和不适用分别保留。语义指标的空值不等于零分，也不等于满分；每个均值旁边的有效样本数就是该指标的统计分母。</div>'
                    '<p>综合质量分按适用维度重新归一化：100 × Σ（权重 × 维度分 ÷ 4）÷ Σ适用权重。维度分为0～4；证据不足或客观不适用时不进入分母，必须说明原因。</p>'
                    '<p>均分同时查看适用权重、各维度、参考核验情况和严重错误。事实精确率、关键事实覆盖率及数字准确率分别反映正确性、遗漏和数字绑定；推荐成功还需要满足需求与预算条件，证据不足时保留未知状态。</p>'
                    + ('<div class="stat-notes">' + _paragraphs(report.get("notes")) + '</div>' if report.get("notes") else '')
                    + '</section>')
    sections.append('<h2>各任务质量与完成情况</h2><div class="task-grid">' + (cards or '<p class="muted">暂无任务汇总。</p>') + '</div>')
    sections.append('<section class="panel"><h2>指标定义与评分权重</h2><h3>指标口径</h3>'
                    + _definitions_html(report) + '<h3>任务维度权重</h3>' + _rubric_html(report) + '</section>')
    sections.append('<section class="panel"><h2>语种与来源分组</h2><h3>按任务和语种</h3>'
                    + _group_table(report, report.get("by_language"), detailed=False)
                    + '<details><summary>查看任务、语种、截图／实拍与场景的完整分组</summary>'
                    + _group_table(report, report.get("groups"), detailed=True) + '</details></section>')
    unique = _mapping(report.get("unique_tasks"))
    if unique:
        unique_rows = [[_escape(_label(task)), _escape(_mapping(stats).get("expected_pairs", "未提供")),
                        _metric(_mapping(stats).get("quality_score"), "quality_score"),
                        _metric(_mapping(stats).get("factual_precision"), "factual_precision")]
                       for task, stats in unique.items()]
        sections.append('<section class="panel"><h2>图片去重后的辅助统计</h2><p class="muted">主统计保留全部图片任务；下表仅剔除相同图片文件内容的后续副本。</p>'
                        + _table(["任务", "去重后任务数", "质量均分", "事实精确率"], unique_rows) + '</section>')
    options = lambda values, label: '<option value="">全部' + label + '</option>' + ''.join(
        f'<option value="{_escape(value)}">{_escape(_label(value))}</option>' for value in values)
    languages = sorted({str(sample.get("language", "未知")) for sample in samples})
    captures = sorted({str(sample.get("capture_type", "unknown")) for sample in samples})
    sample_tasks = [key for key in _TASK_NAMES if any(sample.get("task") == key for sample in samples)]
    sample_tasks += sorted({str(sample.get("task", "unknown")) for sample in samples} - set(sample_tasks))
    filters = ('<div class="filters"><label class="filter-field search" for="filter-search">搜索图片、正文或证据<input id="filter-search" type="search" placeholder="输入关键字"></label>'
               '<label class="filter-field" for="filter-task">任务<select id="filter-task">' + options(sample_tasks, '任务') + '</select></label>'
               '<label class="filter-field" for="filter-language">语种<select id="filter-language">' + options(languages, '语种') + '</select></label>'
               '<label class="filter-field" for="filter-capture">来源<select id="filter-capture">' + options(captures, '来源') + '</select></label>'
               '<label class="filter-check"><input id="filter-problems" type="checkbox">仅显示问题或未完成项</label>'
               '<button id="filter-reset" type="button">重置筛选</button><button id="expand-all" type="button">展开全部</button><button id="collapse-all" type="button">收起全部</button></div>')
    sample_cards = ''.join(_sample_html(report, sample, index) for index, sample in enumerate(samples, 1))
    sections.append('<section class="panel"><h2>逐图片与任务的完整结果</h2><p>展开样本查看被测正文、独立参考事实、评分依据、逐条陈述核对及规则检查。筛选仅影响本节，汇总统计保持不变。</p>'
                    + filters + f'<p id="filter-count" class="filter-count" role="status" aria-live="polite">共 {len(samples)} 条图片任务结果</p>'
                    + '<noscript><p class="notice">当前未启用脚本。汇总、表格和全部样本均可查看；请逐项展开样本，浏览器查找可检索正文。</p></noscript>'
                    + sample_cards + '<p id="no-matches" class="no-matches hidden">当前筛选条件下没有结果。</p></section>')
    sections.append('<footer>报告使用内嵌样式和脚本，可离线查看。正文、参考事实和评审证据均以纯文本展示；本页不请求外部接口、字体或图片。</footer>')
    csp = "default-src 'none'; img-src data:; style-src 'unsafe-inline'; script-src 'unsafe-inline'; connect-src 'none'; object-src 'none'; base-uri 'none'; form-action 'none'"
    return ('<!doctype html><html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">'
            f'<meta http-equiv="Content-Security-Policy" content="{_escape(csp)}"><title>{_escape(title)}</title>'
            f'<style>{_CSS}</style></head><body><main>{"".join(sections)}</main><script>{_JS}</script></body></html>')
