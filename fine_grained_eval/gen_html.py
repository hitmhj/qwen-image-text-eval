#!/usr/bin/env python3
"""Evaluation JSONL -> simple, self-contained HTML and optional aggregate JSON."""
from __future__ import annotations

import argparse
import json
import math
import random
from collections import defaultdict
from html import escape
from pathlib import Path
from statistics import mean

from common import atomic_text, char_count, ensure_distinct_paths, read_jsonl
from contracts import EVALUATION_SCHEMA, TASKS, check_prediction, validate
from gen_truth import check_truth
from metrics import calculate

NAMES = {"summary": "摘要有效信息", "interpretation": "AI解读", "recommendation": "组合约束满足", "phrase": "常用语成功"}
STATES = {"scored": "已评分", "not_applicable": "不适用", "reference_error": "参考生成/复核失败",
          "judge_error": "评审失败，待重试", "missing_prediction": "模型输出缺失，计0", "model_error": "模型失败/空输出，计0"}
FIELDS = {"meaning": "主要内容及含义", "conditions": "必要关系与条件", "boundaries": "重要边界及限制"}
SCENE_NAMES = {"menu": "菜单", "medicine": "药品", "beauty": "化妆品", "food": "食物标签", "road_sign": "路牌", "other": "其他", "reference_error": "参考失败，场景待定"}
LABELS = {"correct": "正确", "incorrect": "错误", "missing": "遗漏", "passed": "通过", "failed": "未通过",
          "unknown": "依据不足，未通过", "supported_relevant": "有据且相关", "supported_irrelevant": "有据但无关",
          "contradicted": "与证据矛盾", "unsupported": "无依据", "valid_optional": "正确的可选补充", "irrelevant": "无关"}
CHECKS = {"target_language": "当地语言适用", "intent_fidelity": "用户意图准确", "selection_consistency": "菜品数量对应",
          "communicative_validity": "可理解且语用正确", "items_exist": "菜品在菜单中", "positive_integer_quantities": "数量为正整数",
          "budget": "预算满足", "no_spicy": "不辣要求", "arithmetic_consistency": "金额核算一致"}


def e(value):
    return escape(str(value) if value is not None else "—", quote=True)


def pre(value):
    text = value if isinstance(value, str) else json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)
    return "<pre>" + e(text) + "</pre>"


def table(heads, rows):
    return '<div class="scroll"><table><thead><tr>' + ''.join('<th>'+e(x)+'</th>' for x in heads) + '</tr></thead><tbody>' + ''.join('<tr>'+''.join('<td>'+str(x)+'</td>' for x in row)+'</tr>' for row in rows) + '</tbody></table></div>'


def pct(x):
    return "N/A" if x is None else f"{x * 100:.1f}%"


def lines(values):
    return '<br>'.join(e(v) for v in values) or '—'


def score_details(r):
    d, task = r["details"], r["task"]
    if not d:
        return ''
    if task == "summary":
        return table(["计分项", "本图结果"], [
            ["精确率 P：说出的事实中正确且相关的比例", f'{d["valid_claims"]}/{d["claims"]} = {pct(d["precision"])}'],
            ["覆盖率 R：必需事实中正确覆盖的比例", f'{d["covered"]}/{d["required"]} = {pct(d["recall"])}'],
            ["信息 F 分数：2PR/(P+R)", pct(d["f_score"])],
            ["实际长度 / 参考长度（非空白字符）", f'{d["candidate_characters"]} / {d["reference_characters"]} = {d["length_ratio"]:.3f}'],
            ["长度系数：min(1,参考长度/实际长度)", f'{d["length_coefficient"]:.3f}'],
            ["最终：信息 F 分数 × 长度系数", pct(r["score"])]])
    if task == "interpretation":
        return table(["字段", "正确项 / 必需项", "字段正确率"],
            [[e(FIELDS[f["field"]]), f'{f["correct"]}/{f["required"]}', pct(f["score"])] for f in d["fields"]]) + \
            f'<p>必需项 K={d["required"]}；额外无效项 H={d["extra_invalid"]}；校正系数 K/(K+H)={d["extra_coefficient"]:.3f}。本图得分为字段正确率均值 × 校正系数。</p>'
    if task == "recommendation":
        return f'<p>通过 {d["passed"]}/{d["total_constraints"]} 项；程序核算总价：{e(d["computed_total_minor"])} 最小货币单位，币种 {e(d["computed_currency"])}。</p>' + \
            table(["约束", "结果", "核算依据"], [[e(CHECKS.get(c["id"], c["id"])), e(LABELS[c["status"]]), e(c["reason"])] for c in d["constraints"]])
    return table(["预定表达单元", "成功", "未通过的条件"],
        [[e(u["id"]), "是" if u["success"] else "否", lines([CHECKS[k] for k in u["failed_checks"]])] for u in d["units"]])


def reference_view(t, task):
    ref = t["reference"][task]
    if task in ("summary", "interpretation"):
        facts = [("摘要", x) for x in ref["required_facts"]] if task == "summary" else [
            (FIELDS[f["id"]], x) for f in ref["fields"] for x in f["items"]]
        view = table(["项 / 字段", "必需命题与限定", "接受答案 / 判错边界", "依据与必要性"], [
            [e(x["id"]+" / "+field), e(x["statement"])+ '<br>'+lines(x["qualifiers"]),
             '接受：'+lines(x["acceptable_answers"])+'<br>判错：'+lines(x["failure_conditions"]),
             e(x["applicability_reason"])+ '<br>'+e(x["reasoning"])+ '<br>证据：'+lines(x["evidence_ids"]) + ('<br>关键项' if x["critical"] else '')]
            for field, x in facts])
        if task == "summary":
            view += '<p>允许补充：'+lines(ref["allowed_supplement"])+'</p>'
        return view
    if task == "recommendation":
        req = ref["requirements"]
        requirements = table(["预算（最小货币单位）", "币种", "不辣要求", "过敏忌口"], [[e(req["budget_minor"]),e(req["currency"]),"是" if req["no_spicy"] else "未提出",lines(req["allergies"])]])
        return table(["菜品ID", "菜名 / 原文", "单价（最小货币单位）", "辣度 / 过敏原依据"], [
            [e(m["id"]), e(m["name"])+ '<br>'+e(m["original"]), e(m["price_minor"])+ ' '+e(m["currency"]),
             e({"yes":"辣", "no":"明确不辣", "unknown":"辣度未知"}[m["spicy"]])+ '<br>'+lines(m["allergens"])+ '<br>'+('配料完整' if m["ingredients_complete"] else '配料信息不完整')]
            for m in ref["menu"]]) + '<p>用户约束：</p>'+requirements + \
            (table(["额外约束ID", "用户要求"], [[e(c["id"]), e(c["statement"])] for c in ref["semantic_constraints"]]) if ref["semantic_constraints"] else '')
    return '<p>可接受语言：'+lines(ref["accepted_languages"])+ '</p>' + table(["单元", "沟通目的", "必需槽位", "参考模板"],
        [[e(u["id"]),e(u["purpose"]),lines(u["required_slots"]),e(u["reference_template"])] for u in ref["units"]])


def judgement_view(r):
    j = r["judgement"]
    if not j:
        return ''
    task = r["task"]
    if task == "recommendation":
        return table(["提取菜品ID", "数量", "模型原文"], [[e(s["id"]),e(s["quantity"]),e(s["quote"])] for s in j["selections"]])
    if task == "phrase":
        return table(["单元", "原文", "四项判断", "理由"], [[e(u["id"]),e(u["quote"]),
            lines([CHECKS[k]+('：通过' if u[k] else '：未通过') for k in ("target_language","intent_fidelity","selection_consistency","communicative_validity")]),e(u["reason"])] for u in j["units"]])
    items = j["claims"] if task == "summary" else j["items"] + j["extra_claims"]
    return table(["项", "模型原文", "判定", "证据 / 理由"], [[e(x["id"]),e(x["quote"]),
        e(LABELS[x.get("status", x.get("label"))]),lines(x["evidence_ids"])+ '<br>'+e(x["reason"])] for x in items])


def validate_result(r):
    validate(r, EVALUATION_SCHEMA)
    check_truth(r["truth"])
    if r["image_name"] != r["truth"]["image_name"] or r["image_sha256"] != r["truth"]["image_sha256"]:
        raise ValueError("result/reference image mismatch")
    state = r["evaluation_status"]
    if r["prediction"] is not None:
        p = r["prediction"]
        check_prediction(p)
        if any(p[k] != r[k] for k in ("image_name", "image_sha256", "task", "model_id", "run_id")):
            raise ValueError("evaluation/prediction identity mismatch")
        if p["prompt"] != r["truth"]["tasks"][r["task"]]["prompt"] or p["context"] != r["truth"]["context"]:
            raise ValueError("evaluation/prediction prompt mismatch")
    if state not in ("reference_error", "not_applicable") and (
        r["truth"]["status"] != "ok" or not r["truth"]["reference"]["applicability"][r["task"]]["applicable"]):
        raise ValueError("capability result needs applicable, accepted reference")
    if state == "not_applicable" and (r["truth"]["status"] != "ok" or r["truth"]["reference"]["applicability"][r["task"]]["applicable"]):
        raise ValueError("not_applicable disagrees with frozen reference")
    if state == "pending":
        raise ValueError("pending is not a finalized evaluation result")
    if state == "scored":
        if r["prediction"] is None or r["prediction"]["status"] != "ok" or r["truth"]["status"] != "ok":
            raise ValueError("scored result requires successful truth and prediction")
        expected = calculate(r["truth"], r["task"], r["judgement"], r["prediction"]["output"])
        if expected != {"score": r["score"], "details": r["details"]}:
            raise ValueError("stored score/details disagree with deterministic recalculation")
    elif state in ("model_error", "missing_prediction"):
        if r["score"] != 0:
            raise ValueError("model failure/missing must be zero")
        if state == "missing_prediction" and r["prediction"] is not None:
            raise ValueError("missing result must not contain a prediction")
        if state == "model_error" and (r["prediction"] is None or (r["prediction"]["status"] == "ok" and r["prediction"]["output"].strip())):
            raise ValueError("model_error disagrees with model output")
    elif r["score"] is not None:
        raise ValueError("inapplicable/reference/judge failure must not have a capability score")
    if r["score"] is not None and not 0 <= r["score"] <= 1:
        raise ValueError("score outside [0,1]")
    if state == "reference_error" and r["truth"]["status"] != "error":
        raise ValueError("reference_error disagrees with reference status")


def aggregate(rows):
    scored = [r for r in rows if r["score"] is not None]
    na = sum(r["evaluation_status"] == "not_applicable" for r in rows)
    reference_failed = sum(r["evaluation_status"] == "reference_error" for r in rows)
    judge_failed = sum(r["evaluation_status"] == "judge_error" for r in rows)
    avg = mean(r["score"] for r in scored) if scored else None
    # Resample independent image clusters, not claims or individual constraints.
    clusters = defaultdict(list)
    for row in scored:
        clusters[row["image_sha256"]].append(row["score"])
    ci = None
    if len(clusters) >= 2:
        values = [mean(v) for v in clusters.values()]
        rng = random.Random(20261022)
        boot = sorted(mean(rng.choices(values, k=len(values))) for _ in range(1000))
        ci = [boot[24], boot[974]]
    predictions = [r["prediction"] for r in rows if r["prediction"] and r["prediction"]["status"] == "ok"]
    characters = [char_count(p["output"]) for p in predictions]
    tokens = [p["generation"]["output_tokens"] for p in predictions if p["generation"]["output_tokens"] is not None]
    known_finish = sum(bool(p["generation"]["finish_reason"]) for p in predictions)
    truncated = sum(p["generation"]["finish_reason"] == "length" for p in predictions)
    return {"planned_records": len(rows), "scored_records": len(scored), "not_applicable": na,
            "reference_errors": reference_failed, "judge_errors": judge_failed,
            "mean": avg, "mean_is_partial": bool(reference_failed or judge_failed),
            "unique_images": len(clusters), "bootstrap_95_ci": ci,
            "model_failures": sum(r["evaluation_status"] in ("model_error", "missing_prediction") for r in rows),
            "generation": {"provided_ok_outputs": len(predictions), "mean_characters": mean(characters) if characters else None,
                "max_characters": max(characters) if characters else None, "known_output_tokens": len(tokens),
                "mean_output_tokens": mean(tokens) if tokens else None, "known_finish_reasons": known_finish,
                "truncated_outputs": truncated, "truncation_rate_among_known": truncated / known_finish if known_finish else None}}


def render(rows, title):
    grouped = defaultdict(list)
    for r in rows:
        scene = r["truth"]["reference"]["scene"] if r["truth"]["status"] == "ok" else "reference_error"
        grouped[(r["model_id"], r["run_id"], r["task"], scene)].append(r)
    totals = []
    report_groups = []
    for key, values in sorted(grouped.items()):
        a = aggregate(values)
        report_groups.append({"model_id": key[0], "run_id": key[1], "task": key[2], "scene": key[3], **a})
        interval = a["bootstrap_95_ci"]
        totals.append([e(key[0]+" / "+key[1]), e(NAMES[key[2]]), e(SCENE_NAMES[key[3]]), a["scored_records"],
                       pct(a["mean"])+("（未完成子集均值）" if a["mean_is_partial"] else ""),
                       "—" if interval is None else pct(interval[0])+"–"+pct(interval[1]),
                       a["not_applicable"], a["reference_errors"]+a["judge_errors"], a["model_failures"]])
    overview_groups = defaultdict(list)
    for r in rows:
        overview_groups[(r["model_id"], r["run_id"], r["task"])].append(r)
    overviews = [{"model_id": k[0], "run_id": k[1], "task": k[2], **aggregate(v)} for k, v in sorted(overview_groups.items())]
    body = '<h1>'+e(title)+'</h1><p>四项主指标分别计算，图片等权；组合分为约束满足比例，不是全部合规成功率。待评审或参考失败不冒充模型0分，其数量保留；有待评记录时总体结论尚不完整。</p>'
    body += '<p>参考由模型生成并机器复核，默认未经人工核验。表中区间仅反映已评分样本的抽样波动，不包含参考或评审器偏差；同模板与同产品变体仍需在采样时识别。</p>'
    body += table(["模型/批次", "指标", "全场景已评分数", "全场景均值", "待参考/评审"],
        [[e(a["model_id"]+" / "+a["run_id"]), e(NAMES[a["task"]]), a["scored_records"],
          pct(a["mean"])+("（部分结果）" if a["mean_is_partial"] else ""), a["reference_errors"]+a["judge_errors"]] for a in overviews])
    body += '<h2>输出长度与截断</h2><p>字符数按非空白字符计算。token与结束原因只使用推理端实际记录；未知不按未截断处理。截断比例的分母仅为结束原因已知的成功输出数，未记录者单列。</p>'
    length_rows = []
    for a in overviews:
        g = a["generation"]
        length_rows.append([e(a["model_id"]+" / "+a["run_id"]), e(NAMES[a["task"]]),
            "—" if g["mean_characters"] is None else f'{g["mean_characters"]:.1f} / {g["max_characters"]}',
            "—" if g["mean_output_tokens"] is None else f'{g["mean_output_tokens"]:.1f}（{g["known_output_tokens"]}条已知）',
            f'{g["truncated_outputs"]}/{g["known_finish_reasons"]}，'+pct(g["truncation_rate_among_known"]),
            g["provided_ok_outputs"]-g["known_finish_reasons"]])
    body += table(["模型/批次","任务","平均 / 最大字符数","平均输出token","截断数 / 结束原因已知数","结束原因未知数"],length_rows)
    body += '<h2>按任务与场景分类</h2>'+table(["模型/批次","指标","场景","计分数","均值","95%区间","N/A","待评","模型失败"],totals)
    for group_key, values in sorted(grouped.items()):
        body += '<h2>'+e(" / ".join([group_key[0],group_key[1],NAMES[group_key[2]],SCENE_NAMES[group_key[3]]]))+'</h2>'
        for r in sorted(values, key=lambda x:x["image_name"]):
            t, p = r["truth"], r["prediction"]
            body += '<article><h3>'+e(r["image_name"])+' · '+e(NAMES[r["task"]])+'</h3>'
            body += '<img loading="lazy" alt="原始输入图片" src="'+e(t["image_data_url"])+'">'
            body += '<h4>输入 Prompt</h4>'+pre((p or t["tasks"][r["task"]])["prompt"])+ '<p>共享上下文：</p>'+pre(t["context"])
            body += '<h4>模型原始输出</h4>'+pre(p["output"] if p else "未提供输出")
            body += '<h4>本图指标与具体计分</h4>'+table(["项目","结果"],[["状态",e(STATES[r["evaluation_status"]])],
                ["本图得分",pct(r["score"])],["参考目标", "满足全部参考项时为100%；不是另一模型的实测成绩"],
                ["人工核验状态", "已声明人工核验" if t["human_verified"] else "未人工核验（机器参考）"]])
            if r["details"]:
                body += score_details(r)
            if t["status"] == "ok":
                body += '<h4>参考答案与量表</h4>'
                if r["task"] == "summary":
                    body += '<p>参考长度：'+e(t["reference_length"])+'非空白字符；每图使用有效参考候选长度中位数。</p>'
                    first = True
                    for s, good in zip(t["summary_references"], t["verification"]["summary_valid"]):
                        if good and first:
                            body += '<p>有效参考摘要示例：</p>'+pre(s["text"])
                            first = False
                        else:
                            body += '<details><summary>其他参考摘要 · '+("核验接受" if good else "核验未接受")+'</summary>'+pre(s["text"])+'</details>'
                body += reference_view(t, r["task"])
                body += '<details><summary>参考证据原文与位置</summary>'+table(["证据ID", "来源 / 位置", "原文", "外部来源"],
                    [[e(x["id"]),e(x["source"])+ ' / '+e(x["location"]),e(x["text"]),e(x["url"])+ '<br>'+e(x["version"])] for x in t["reference"]["evidence"]])+'</details>'
            body += '<h4>问题与本例结论</h4>'
            if r["error"]:
                body += pre(r["error"])
            if r["judgement"]:
                issues = r["judgement"]["issues"]
                body += pre(issues if issues else ["评审未列出额外问题；完整性仍以逐项记录为准"])
                body += judgement_view(r)
                body += '<details><summary>原始评分记录 JSON</summary>'+pre(r["judgement"])+pre(r["details"])+'</details>'
            if r["score"] is not None:
                critical = r["details"].get("critical_failures", [])
                conclusion = "达到本量表全部检查项" if r["score"] == 1 else "存在未完成、错误或长度冗余，见计分记录"
                body += '<p>'+e(conclusion)+('；关键项未通过：'+e("、".join(critical)) if critical else '')+'。分数不等同上线认证。</p>'
            else:
                body += '<p>本例未形成可用的能力分数，不能据此推断模型能力。</p>'
            if p:
                body += '<details><summary>模型长度、token与运行记录</summary>'+pre(p["generation"])+pre(r["warnings"] or [])+'</details>'
            body += '</article>'
    style = 'body{font:15px/1.75 system-ui,"Microsoft YaHei",sans-serif;color:#111;margin:0}main{max-width:1100px;margin:auto;padding:24px}h1{font-size:26px}h2{font-size:21px;margin-top:32px}h3{font-size:18px}h4{font-size:15px;margin-bottom:8px}article{border-top:1px solid #999;margin-top:24px;padding-top:16px}table{border-collapse:collapse;width:100%;font-size:13px}th,td{border:1px solid #aaa;padding:8px;text-align:left;vertical-align:top}th{background:#f3f3f3}.scroll{overflow:auto}pre{white-space:pre-wrap;overflow-wrap:anywhere;background:#f6f6f6;padding:12px;font:13px/1.7 system-ui,sans-serif}img{max-width:100%;max-height:440px;object-fit:contain}summary{cursor:pointer}details{margin:12px 0}'
    html = '<!doctype html><html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>'+e(title)+'</title><style>'+style+'</style></head><body><main>'+body+'</main></body></html>'
    return html, {"groups": report_groups, "overall_by_model_run_task": overviews}


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("evaluations", type=Path)
    p.add_argument("--output", required=True, type=Path)
    p.add_argument("--summary-output", type=Path)
    p.add_argument("--title", default="图片模型细粒度评测报告")
    args = p.parse_args(argv)
    ensure_distinct_paths(args.evaluations, args.output, args.summary_output)
    rows = read_jsonl(args.evaluations)
    if not rows:
        raise ValueError("empty evaluation file")
    seen = set()
    for row in rows:
        validate_result(row)
        k = (row["image_name"], row["task"], row["model_id"], row["run_id"])
        if k in seen:
            raise ValueError("duplicate result key")
        seen.add(k)
    html, summary = render(rows, args.title)
    atomic_text(args.output, html)
    if args.summary_output:
        atomic_text(args.summary_output, json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False)+"\n")
    print(json.dumps({"html": str(args.output), "records": len(rows)}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (ValueError, OSError, KeyError, TypeError) as exc:
        raise SystemExit(str(exc))
