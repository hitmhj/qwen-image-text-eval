"""HTML 报告安全、空值口径与三任务证据展示的合成数据测试。"""

import copy
import unittest
from html.parser import HTMLParser

from qwen_text_eval.html_report import render_html


class ParsedHTML(HTMLParser):
    def __init__(self, document):
        super().__init__(convert_charrefs=True)
        self.nodes = []
        self.text = []
        self.scripts = []
        self.in_script = False
        self.feed(document)

    def handle_starttag(self, tag, attrs):
        self.nodes.append((tag, dict(attrs)))
        if tag == "script":
            self.in_script = True
            self.scripts.append("")

    def handle_endtag(self, tag):
        if tag == "script":
            self.in_script = False

    def handle_data(self, data):
        self.text.append(data)
        if self.in_script:
            self.scripts[-1] += data


def fixture():
    rubric = {"tasks": {task: {"dimensions": {"factual_accuracy": {
        "name": "事实准确性", "weight": 35, "description": "核对原图事实"}}}
        for task in ["summary", "interpretation", "recommendation"]},
        "scale": {"anchors": {"0": "任务失败", "4": "充分满足"}}}
    stats = {"expected_pairs": 2, "generation_ok": 1, "generation_missing": 1,
             "generation_errors": 0, "judge_evaluated": 1, "judge_missing": 0,
             "quality_score": {"mean": 75, "n": 1},
             "factual_precision": {"mean": 1, "n": 1},
             "hallucination_rate": {"mean": 0, "n": 1},
             "numeric_claim_accuracy": {"mean": None, "n": 0},
             "major_error_count": 0, "critical_error_count": 0,
             "dimensions": {"factual_accuracy": {"mean": 3, "n": 1}}}
    report = {"dataset_id": "menu_reference", "run_id": "合成运行", "rubric": rubric,
              "tasks": {"summary": copy.deepcopy(stats), "interpretation": copy.deepcopy(stats),
                        "recommendation": copy.deepcopy(stats)},
              "by_language": [{"task": "summary", "language": "日语", **copy.deepcopy(stats)}],
              "groups": [{"task": "summary", "language": "日语", "capture_type": "photo",
                          "scene": "menu", **copy.deepcopy(stats)}],
              "metric_definitions": {"hallucination_rate": {"name": "无依据率",
                  "definition": "无依据陈述除以可判定陈述", "denominator": "可判定陈述数"}}}
    reference = {"reference_status": "draft", "image_quality": {"readability": "clear", "issues": []},
                 "facts": [{"fact_id": "f001", "category": "price", "source_text": "ラーメン 800円",
                            "normalized_zh": "拉面每份800日元", "evidence": "菜单左上第一行",
                            "confidence": "high", "importance": 3,
                            "required_for": ["summary", "interpretation", "recommendation"],
                            "numeric": {"value": "800", "unit": "份", "currency": "JPY", "qualifier": "每份"}}],
                 "recommendation_feasibility": {"status": "feasible", "reason": "菜单能满足已知需求"}}
    judgement = {"dimensions": {"factual_accuracy": {"score": 3, "reason": "核对价格与主体一致",
                  "evidence_fact_ids": ["f001"]}},
                 "claims": [{"claim_id": "c001", "text": "拉面每份800日元", "category": "price",
                             "verdict": "supported", "severity": "minor", "fact_ids": ["f001"],
                             "explanation": "菜单价格一致"}], "covered_fact_ids": ["f001"],
                 "critical_errors": [], "review_reasons": [],
                 "requirement_checks": [{"constraint_id": "预算", "status": "met", "reason": "未超过1000日元",
                                         "evidence_fact_ids": ["f001"]}],
                 "recommendation_plan": {"items": [{"name": "拉面", "quantity": 1}],
                                         "pricing": {"total": "800", "currency": "JPY"}}}
    sample = {"image_id": "img001", "relative_path": "日语/实拍/菜单.png", "language": "日语",
              "capture_type": "photo", "scene": "menu", "task": "summary", "text": "拉面菜单\n每份800日元",
              "generation_status": "ok", "judge_status": "ok", "reference_status": "draft",
              "quality_score": 75, "applicable_weight": 90, "factual_precision": 1,
              "numeric_claim_accuracy": None, "required_fact_count": 1, "covered_fact_count": 1,
              "reference": reference, "judgement": judgement, "review_reasons": [],
              "needs_human_review": False, "prompt_version": "v04_summary",
              "rules": {"hard_pass": True, "char_count": 17,
                        "checks": [{"id": "two_lines", "name": "两行格式", "status": "pass",
                                    "observed": 2, "expected": 2, "details": "正文仅两行"}]}}
    return report, sample


class HtmlReportTests(unittest.TestCase):
    def test_three_tasks_and_independent_dataset_are_visible(self):
        report, sample = fixture()
        samples = [copy.deepcopy(sample) for _ in range(3)]
        for row, task in zip(samples, ["summary", "interpretation", "recommendation"]):
            row["task"] = task
        samples[2]["recommendation_requirement"] = {"id": "r01", "text": "单人预算1000日元",
                                                     "constraints": [{"id": "预算", "max_amount": 1000}]}
        document = render_html(report, samples)
        parsed = ParsedHTML(document)
        option_values = [attrs.get("value") for tag, attrs in parsed.nodes if tag == "option"]
        self.assertIn("recommendation", option_values)
        self.assertIn("独立菜单参考测试集", document)
        self.assertIn("不混合汇总", document)
        self.assertIn("单人预算1000日元", document)
        self.assertIn("未超过1000日元", document)
        self.assertIn("推荐方案、数量与价格核算", document)
        self.assertIn("菜单能满足已知需求", document)

    def test_missing_metrics_are_not_zero_but_actual_zero_is_visible(self):
        report, sample = fixture()
        document = render_html(report, [sample])
        self.assertIn("0.0%", document)
        self.assertIn("未测／不可判定", document)
        self.assertIn("有效样本 0", document)
        self.assertIn("有效样本 1", document)
        self.assertIn("生成缺失数", document)
        self.assertIn("3.00 / 4", document)
        self.assertIn("适用权重：90～90", document)
        self.assertIn("可判定陈述数", document)

    def test_all_model_text_and_metadata_are_escaped(self):
        report, sample = fixture()
        attack = '</script><script>alert("注入")</script><img src=x onerror=alert(1)>'
        report["run_id"] = attack
        report["dataset_name"] = attack
        report["metric_definitions"]["hallucination_rate"]["definition"] = attack
        report["rubric"]["tasks"]["summary"]["dimensions"]["factual_accuracy"]["name"] = attack
        sample["text"] = attack
        sample["relative_path"] = attack
        sample["language"] = attack
        sample["prompt_version"] = attack
        sample["reference"]["facts"][0]["normalized_zh"] = attack
        sample["reference"]["facts"][0]["source_text"] = attack
        sample["reference"]["facts"][0]["numeric"]["qualifier"] = attack
        sample["judgement"]["dimensions"]["factual_accuracy"]["reason"] = attack
        sample["judgement"]["claims"][0]["explanation"] = attack
        sample["rules"]["checks"][0]["details"] = attack
        sample["review_reasons"] = [attack]
        document = render_html(report, [sample], title=attack)
        parsed = ParsedHTML(document)
        scripts = [(tag, attrs) for tag, attrs in parsed.nodes if tag == "script"]
        self.assertEqual(len(scripts), 1)
        self.assertEqual([tag for tag, _ in parsed.nodes if tag == "img"], [])
        self.assertNotIn('alert("注入")', ''.join(parsed.scripts))
        self.assertTrue(any(attack in text for text in parsed.text))
        self.assertIn("&lt;/script&gt;", document)
        for _, attrs in parsed.nodes:
            self.assertFalse(any(key.startswith("on") for key in attrs))

    def test_only_safe_raster_data_urls_can_be_images(self):
        report, sample = fixture()
        safe_url = "data:image/png;base64,aGVsbG8="
        sample["image_data_url"] = safe_url
        parsed = ParsedHTML(render_html(report, [sample]))
        self.assertEqual([attrs["src"] for tag, attrs in parsed.nodes if tag == "img"], [safe_url])
        for url in ["https://host/image.png", "javascript:alert(1)", "data:image/svg+xml;base64,aGVsbG8=",
                    'data:image/png;base64,aGVsbG8=" onerror="alert(1)', "data:image/gif;base64,aGVsbG8=",
                    "data:image/png;base64,aGVsbG8=\n"]:
            with self.subTest(url=url):
                sample["image_data_url"] = url
                parsed = ParsedHTML(render_html(report, [sample]))
                self.assertFalse(any(tag == "img" for tag, _ in parsed.nodes))

    def test_local_links_are_restricted_to_relative_raster_paths(self):
        report, sample = fixture()
        sample["image_link"] = "../图片/一 张.png"
        parsed = ParsedHTML(render_html(report, [sample]))
        links = [attrs["href"] for tag, attrs in parsed.nodes if tag == "a"]
        self.assertEqual(links, ["../%E5%9B%BE%E7%89%87/%E4%B8%80%20%E5%BC%A0.png"])
        for link in ["javascript:alert(1).png", "https://host/image.png", "//host/image.png", "/abs/image.png",
                     "file:///tmp/image.png", "image.svg", "image.html", "image.png?execute=1",
                     "http://[invalid/image.png", "C:\\image.png", "image.png\n"]:
            with self.subTest(link=link):
                sample["image_link"] = link
                parsed = ParsedHTML(render_html(report, [sample]))
                self.assertFalse(any(tag == "a" for tag, _ in parsed.nodes))

    def test_missing_and_stale_results_are_classified_as_problems(self):
        report, sample = fixture()
        samples = [copy.deepcopy(sample) for _ in range(4)]
        samples[1].update(generation_status="missing", judge_status="missing", text=None, reference=None, judgement=None)
        samples[2].update(judge_status="stale", quality_score=None)
        samples[3].update(critical_error=True, needs_human_review=True, review_reasons=["过敏原错误需要复核"])
        document = render_html(report, samples)
        parsed = ParsedHTML(document)
        cards = [attrs for tag, attrs in parsed.nodes if tag == "details" and attrs.get("class") == "sample-card"]
        self.assertEqual([card["data-problem"] for card in cards], ["0", "1", "1", "1"])
        self.assertIn("生成正文缺失", document)
        self.assertIn("参考事实缺失", document)
        self.assertIn("结果过期", document)
        self.assertIn("过敏原错误需要复核", document)
        self.assertIn("不进入有效语义指标", document)

    def test_full_evidence_and_candidate_are_available_without_javascript(self):
        report, sample = fixture()
        document = render_html(report, [sample])
        parsed = ParsedHTML(document)
        visible_text = ''.join(parsed.text)
        for expected in [sample["text"], "菜单左上第一行", "ラーメン 800円", "核对价格与主体一致",
                         "菜单价格一致", "正文仅两行", "f001", "已覆盖", "v04_summary"]:
            self.assertIn(expected, visible_text)
        self.assertIn("<noscript>", document)
        self.assertIn("未启用脚本", document)
        self.assertEqual(len([1 for tag, attrs in parsed.nodes if tag == "details" and attrs.get("class") == "sample-card"]), 1)
        self.assertFalse(any(attrs.get("src", "").startswith("http") for _, attrs in parsed.nodes))

    def test_dimension_null_and_zero_are_distinct(self):
        report, sample = fixture()
        sample["judgement"]["dimensions"]["factual_accuracy"]["score"] = 0
        sample["judgement"]["dimensions"]["translation"] = {"score": None, "reason": "无翻译需求", "evidence_fact_ids": []}
        document = render_html(report, [sample])
        self.assertIn("0 / 4", document)
        self.assertIn("不适用／证据不足", document)
        self.assertIn("无翻译需求", document)

    def test_new_metric_and_dynamic_dimension_name_are_retained(self):
        report, sample = fixture()
        report["metric_definitions"]["new_metric"] = {"name": "新口径", "definition": "独立的自定义统计口径"}
        report["tasks"]["summary"]["new_metric"] = {"mean": None, "n": 0}
        report["rubric"]["tasks"]["summary"]["dimensions"]["new_dimension"] = {"name": "新维度", "weight": 10, "description": "自定义评分"}
        sample["judgement"]["dimensions"]["new_dimension"] = {"score": 4, "reason": "自定义证据匹配", "evidence_fact_ids": ["f001"]}
        document = render_html(report, [sample])
        self.assertIn("新口径", document)
        self.assertIn("独立的自定义统计口径", document)
        self.assertIn("新维度", document)
        self.assertIn("自定义证据匹配", document)

    def test_empty_report_is_valid_and_offline(self):
        document = render_html({}, [])
        parsed = ParsedHTML(document)
        self.assertIn("暂无任务汇总", document)
        self.assertIn("共 0 条图片任务结果", document)
        self.assertIn("lang=\"zh-CN\"", document)
        self.assertIn("Content-Security-Policy", document)
        self.assertEqual(len(parsed.scripts), 1)
        self.assertFalse(any(tag in {"iframe", "embed", "object", "link"} for tag, _ in parsed.nodes))
        self.assertFalse(any(tag == "script" and "src" in attrs for tag, attrs in parsed.nodes))


if __name__ == "__main__":
    unittest.main()
