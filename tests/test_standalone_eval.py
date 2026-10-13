"""单文件评测的协议、独立性、真实分母与离线报告测试。"""
import copy
import importlib.util
import json
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("standalone_evaluate", ROOT / "evaluate.py")
ev = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(ev)


def image_fixture(root):
    path = root / "pics" / "法语" / "实拍" / "menu.png"
    path.parent.mkdir(parents=True)
    path.write_bytes(b"standalone-image-fixture")
    digest = ev.sha256_file(path)
    texts = {"summary": "餐厅菜单\n列有牛肉面，单份价格二十元。",
        "interpretation": "菜单展示面食与价格。\n\n【菜单信息】\n- 牛肉面价格为二十元。\n- 菜单包含主食。",
        "recommendation": "一人点一份牛肉面，单价20元，总计20元，满足50元预算。"}
    prompts = {task: {"version": "vtest", "task": task, "system": "按原图提供中文正文", "user": "请分析图片"} for task in ev.TASKS}
    row = {"schema_version": 2, "dataset_id": "primary", "run_id": "test-run", "image_id": "img001",
        "relative_path": "法语/实拍/menu.png", "language": "法语", "capture_type": "实拍", "image_sha256": digest,
        "width": 100, "height": 100, "image_status": "ok", "image_error": None, "duplicate_of": None,
        "prompts": prompts, "recommendation_requirement": {"id": "req01", "text": "一人，预算50元。", "constraints": {
            "budget": {"amount": "50", "currency": "CNY", "scope": "total"}, "diners": 1,
            "hard_constraints": [], "preferences": [], "meal_requirements": {}}, "fx_context": None}}
    for task in ev.TASKS:
        row[task] = {"status": "ok", "text": texts[task], "raw_text": texts[task], "latency_seconds": 0.2,
            "image_sha256": digest, "prompt_sha256": ev.fingerprint(prompts[task]), "finish_reason": "stop", "error": None}
    reference = {"schema_version": 2, "dataset_id": "primary", "image_id": "img001", "image_sha256": digest,
        "scene": "menu", "image_quality": {"readability": "clear", "issues": []}, "unreadable_regions": [],
        "reference_status": "draft", "annotator": {"model": "test-gpt", "version": "test-v1"},
        "requirement_sha256": ev.fingerprint(row["recommendation_requirement"]),
        "recommendation_feasibility": {"requirement_id": "req01", "status": "feasible", "reason": "价格明确且未超预算", "evidence_fact_ids": ["f001", "f002"], "fees_status": "complete", "mandatory_fees": []},
        "facts": [{"fact_id": "f001", "category": "name", "source_text": "牛肉面", "normalized_zh": "牛肉面",
            "evidence": "第一行左侧菜名", "confidence": "high", "importance": 3, "required_for": ev.TASKS, "numeric": None},
            {"fact_id": "f002", "category": "price", "source_text": "20元", "normalized_zh": "牛肉面每份20元",
                "evidence": "菜名右侧价格", "confidence": "high", "importance": 2, "required_for": ["interpretation", "recommendation"],
                "numeric": {"value": "20", "unit": "份", "currency": "CNY", "qualifier": "每份"}}]}
    return row, reference


def score_fixture(image, reference, task):
    score = {"schema_version": 2, "dataset_id": image["dataset_id"], "image_id": image["image_id"], "task": task,
        "candidate_sha256": ev.text_sha256(image[task]["text"]), "reference_sha256": ev.fingerprint(reference),
        "judge_prompt_sha256": ev.text_sha256(ev.JUDGE_SYSTEM), "scene": "menu",
        "source_evidence_mode": "reference_only", "task_prompt_sha256": ev.fingerprint(image["prompts"][task]),
        "requirement_sha256": ev.fingerprint(image["recommendation_requirement"]) if task == "recommendation" else None,
        "dimensions": {name: {"score": 4, "reason": "已按图示核验", "evidence_fact_ids": ["f001"]} for name in ev.RUBRIC["tasks"][task]["dimensions"]},
        "claims": [{"claim_id": "c001", "text": "牛肉面", "verdict": "supported", "fact_ids": ["f001"], "severity": "minor", "explanation": "菜名可见", "category": "name"},
            {"claim_id": "c002", "text": "牛肉面20元", "verdict": "supported", "fact_ids": ["f002"], "severity": "major", "explanation": "实体与价格对应", "category": "price"}],
        "covered_fact_ids": ["f001"] if task == "summary" else ["f001", "f002"], "critical_errors": [],
        "needs_human_review": False, "review_reasons": [], "judge": {"model": "test-judge", "version": "v1"}, "recommendation": None}
    if task == "recommendation":
        score["recommendation"] = {"requirement_id": "req01", "disposition": "confirmed",
            "requirement_checks": [{"constraint_id": key, "status": "met", "reason": "符合需求", "evidence_fact_ids": ["f002"]} for key in ["budget", "diners"]],
            "plan": {"items": [{"name": "牛肉面", "quantity": "1", "unit_price": "20", "currency": "CNY", "price_fact_ids": ["f002"], "person": "一人"}],
                "subtotal": "20", "mandatory_fees": [], "total": "20", "currency": "CNY", "fx": None, "budget_total": "20", "budget_currency": "CNY"}}
    return score


class StandaloneEvaluationTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.image, self.reference = image_fixture(self.root)
        self.input = self.root / "generation_results.jsonl"
        ev.write_jsonl(self.input, [self.image])
        self.output = self.root / "eval"

    def tearDown(self):
        self.temporary.cleanup()

    def test_reference_is_blind_to_candidates_and_model(self):
        requests = ev.prepare_reference([self.image], self.output, self.root / "pics")
        self.assertEqual(len(requests), 1)
        payload = requests[0]["input"]
        self.assertNotIn("prompts", payload)
        self.assertNotIn("candidate_text", payload)
        self.assertNotIn("language", payload)
        self.assertNotIn(self.image["summary"]["text"], json.dumps(payload, ensure_ascii=False))

    def test_mixed_cohorts_and_runs_are_rejected(self):
        for field, value in [("dataset_id", "menu_reference"), ("run_id", "other-run")]:
            second = copy.deepcopy(self.image)
            second["image_id"] = "img002"
            second[field] = value
            ev.write_jsonl(self.input, [self.image, second])
            with self.assertRaises(ValueError):
                ev.load_generation(self.input)

    def test_image_and_prompt_mutations_are_rejected(self):
        (self.root / "pics" / self.image["relative_path"]).write_bytes(b"changed")
        with self.assertRaises(ValueError):
            ev.prepare_reference([self.image], self.output, self.root / "pics")
        self.image["prompts"]["summary"]["system"] = "changed"
        ev.write_jsonl(self.input, [self.image])
        with self.assertRaises(ValueError):
            ev.load_generation(self.input)

    def test_human_verified_requires_explicit_operator_flag(self):
        response = copy.deepcopy(self.reference)
        response["reference_status"] = "human_verified"
        target = self.root / "responses.jsonl"
        ev.write_jsonl(target, [response])
        with self.assertRaises(ValueError):
            ev.references_from_file(target, [self.image], importing=True)
        verified = ev.references_from_file(target, [self.image], human_verified=True, importing=True)
        self.assertEqual(verified["img001"]["reference_status"], "human_verified")

    def test_semantic_validation_rejects_unknown_citation_and_stale_hash(self):
        score = score_fixture(self.image, self.reference, "summary")
        for field, value in [("candidate_sha256", "0" * 64), ("reference_sha256", "0" * 64), ("judge_prompt_sha256", "0" * 64)]:
            bad = copy.deepcopy(score)
            bad[field] = value
            with self.assertRaises(ValueError):
                ev.validate_judgement(bad, self.reference, self.image, "summary")
        score["claims"][0]["fact_ids"] = ["missing"]
        with self.assertRaises(ValueError):
            ev.validate_judgement(score, self.reference, self.image, "summary")

    def test_metrics_separate_hallucination_contradiction_unknown(self):
        score = score_fixture(self.image, self.reference, "summary")
        extra = copy.deepcopy(score["claims"][1])
        for verdict, claim_id in [("contradicted", "c003"), ("unsupported", "c004"), ("unverifiable", "c005")]:
            claim = {**extra, "verdict": verdict, "claim_id": claim_id}
            score["claims"].append(claim)
        metrics = ev.score_judgement(score, self.reference)
        self.assertEqual(metrics["factual_error_rate"], 0.5)
        self.assertEqual(metrics["hallucination_rate"], 0.25)
        self.assertEqual(metrics["contradiction_rate"], 0.25)
        self.assertEqual(metrics["numeric_claim_accuracy"], 1 / 3)
        self.assertEqual(metrics["conservative_numeric_accuracy"], 1 / 4)
        self.assertEqual(metrics["numeric_unverifiable_rate"], 1 / 4)

    def test_null_dimensions_and_empty_denominators_are_unmeasured(self):
        score = score_fixture(self.image, self.reference, "summary")
        for dimension in score["dimensions"].values():
            dimension["score"] = None
        score["claims"] = []
        score["covered_fact_ids"] = []
        reference = copy.deepcopy(self.reference)
        reference["facts"] = []
        metrics = ev.score_judgement(score, reference)
        self.assertIsNone(metrics["quality_score"])
        self.assertIsNone(metrics["factual_precision"])
        self.assertIsNone(metrics["numeric_claim_accuracy"])
        self.assertIsNone(metrics["weighted_key_fact_coverage"])

    def test_recommendation_hard_constraints_and_decimal_cost(self):
        score = score_fixture(self.image, self.reference, "recommendation")
        ev.validate_judgement(score, self.reference, self.image, "recommendation")
        check = ev.check_recommendation_budget(score, self.image, self.reference)
        self.assertEqual(check["status"], "met")
        self.assertEqual(check["computed_total"], "20")
        score["recommendation"]["plan"]["total"] = "10"
        self.assertEqual(ev.check_recommendation_budget(score, self.image, self.reference)["status"], "violated")
        score["recommendation"]["requirement_checks"].pop()
        with self.assertRaises(ValueError):
            ev.validate_judgement(score, self.reference, self.image, "recommendation")

    def test_currency_conversion_requires_provided_fx(self):
        score = score_fixture(self.image, self.reference, "recommendation")
        self.image["recommendation_requirement"]["constraints"]["budget"]["currency"] = "USD"
        score["recommendation"]["plan"]["budget_currency"] = "USD"
        score["recommendation"]["plan"]["fx"] = {"from_currency": "CNY", "to_currency": "USD", "rate": "0.14", "source": "judge guessed"}
        self.assertEqual(ev.check_recommendation_budget(score, self.image, self.reference)["status"], "unknown")

    def test_total_budget_is_not_multiplied_by_diners(self):
        score = score_fixture(self.image, self.reference, "recommendation")
        self.image["recommendation_requirement"]["constraints"]["diners"] = 2
        self.image["recommendation_requirement"]["constraints"]["budget"]["amount"] = "30"
        plan = score["recommendation"]["plan"]
        plan["items"][0]["quantity"] = "2"
        plan["subtotal"] = plan["total"] = plan["budget_total"] = "40"
        self.assertEqual(ev.check_recommendation_budget(score, self.image, self.reference)["status"], "violated")
        self.image["recommendation_requirement"]["constraints"]["budget"]["scope"] = "per_person"
        self.assertEqual(ev.check_recommendation_budget(score, self.image, self.reference)["status"], "met")

    def test_known_fx_is_verified_and_low_confidence_prices_stay_unknown(self):
        score = score_fixture(self.image, self.reference, "recommendation")
        self.image["recommendation_requirement"]["constraints"]["budget"]["currency"] = "USD"
        fx = {"from_currency": "CNY", "to_currency": "USD", "rate": "0.14", "source": "实验配置的已核验来源"}
        self.image["recommendation_requirement"]["fx_context"] = fx
        plan = score["recommendation"]["plan"]
        plan["budget_currency"] = "USD"
        plan["budget_total"] = "2.8"
        plan["fx"] = fx
        self.assertEqual(ev.check_recommendation_budget(score, self.image, self.reference)["status"], "met")
        self.reference["facts"][1]["confidence"] = "medium"
        self.assertEqual(ev.check_recommendation_budget(score, self.image, self.reference)["status"], "unknown")

    def test_generation_nested_fx_context_is_supported(self):
        score = score_fixture(self.image, self.reference, "recommendation")
        self.reference["facts"][1]["numeric"]["currency"] = "EUR"
        plan = score["recommendation"]["plan"]
        plan["items"][0]["currency"] = plan["currency"] = "EUR"
        plan["fx"] = {"from_currency": "EUR", "to_currency": "CNY", "rate": "7.50", "source": "用户来源"}
        plan["budget_total"] = "150"
        self.image["recommendation_requirement"]["fx_context"] = {"as_of": "2026-10-13", "source": "用户来源", "rates": {"EUR": {"CNY": "7.5"}}}
        check = ev.check_recommendation_budget(score, self.image, self.reference)
        self.assertEqual(check["status"], "violated")
        self.assertEqual(check["converted_total"], "150.0")

    def test_omitted_ground_truth_fee_is_still_computed(self):
        fee = {"fact_id": "f003", "category": "price", "source_text": "服务费40元", "normalized_zh": "每单必付服务费40元",
            "evidence": "菜单底部明确收费", "confidence": "high", "importance": 3, "required_for": ["recommendation"],
            "numeric": {"value": "40", "unit": "单", "currency": "CNY", "qualifier": "必付服务费"}}
        self.reference["facts"].append(fee)
        self.reference["recommendation_feasibility"]["mandatory_fees"] = [{"fact_id": "f003", "name": "服务费", "kind": "flat", "amount": "40", "currency": "CNY", "scope": "per_order"}]
        score = score_fixture(self.image, self.reference, "recommendation")
        check = ev.check_recommendation_budget(score, self.image, self.reference)
        self.assertEqual(check["status"], "violated")
        self.assertEqual(check["computed_total"], "60")
        self.assertIs(check["budget_overrun"], True)
        report = ev.build_report([self.image], {"img001": self.reference}, {("img001", "recommendation"): score}, self.output)
        self.assertEqual(report["tasks"]["recommendation"]["recommendation_success_rate"], 0)

    def test_unmarked_fees_are_unknown_and_known_lower_bound_can_exceed_budget(self):
        self.reference["recommendation_feasibility"]["fees_status"] = "unmarked"
        score = score_fixture(self.image, self.reference, "recommendation")
        check = ev.check_recommendation_budget(score, self.image, self.reference)
        self.assertEqual(check["status"], "unknown")
        self.assertIsNone(check["budget_overrun"])
        self.image["recommendation_requirement"]["constraints"]["budget"]["amount"] = "10"
        self.reference["requirement_sha256"] = ev.fingerprint(self.image["recommendation_requirement"])
        score = score_fixture(self.image, self.reference, "recommendation")
        check = ev.check_recommendation_budget(score, self.image, self.reference)
        self.assertEqual(check["status"], "violated")
        self.assertIs(check["budget_overrun"], True)
        report = ev.build_report([self.image], {"img001": self.reference}, {("img001", "recommendation"): score}, self.output)
        stats = report["tasks"]["recommendation"]
        self.assertEqual(stats["budget_overrun_rate"], 1)
        self.assertEqual(stats["metric_denominators"]["budget_overrun_rate"]["n"], 1)

    def test_unstructured_visible_fees_do_not_silently_pass(self):
        self.reference["facts"].append({"fact_id": "f003", "category": "restriction", "source_text": "另付服务费",
            "normalized_zh": "服务费另付，金额未标注", "evidence": "菜单底部", "confidence": "high", "importance": 3,
            "required_for": [], "numeric": None})
        score = score_fixture(self.image, self.reference, "recommendation")
        self.assertEqual(ev.check_recommendation_budget(score, self.image, self.reference)["status"], "unknown")

    def test_price_confidence_and_actual_pixel_mode_are_enforced(self):
        score = score_fixture(self.image, self.reference, "summary")
        self.reference["facts"][1]["confidence"] = "low"
        score["reference_sha256"] = ev.fingerprint(self.reference)
        with self.assertRaises(ValueError):
            ev.validate_judgement(score, self.reference, self.image, "summary")
        self.reference["facts"][1]["confidence"] = "medium"
        score["reference_sha256"] = ev.fingerprint(self.reference)
        score["claims"][1]["verdict"] = "contradicted"
        with self.assertRaises(ValueError):
            ev.validate_judgement(score, self.reference, self.image, "summary")
        score["claims"][1]["verdict"] = "unsupported"
        score["claims"][1]["fact_ids"] = []
        with self.assertRaises(ValueError):
            ev.validate_judgement(score, self.reference, self.image, "summary")
        score["source_evidence_mode"] = "image_and_reference"
        with self.assertRaises(ValueError):
            ev.validate_judgement(score, self.reference, self.image, "summary")
        ev.validate_judgement(score, self.reference, self.image, "summary", "image_and_reference")

    def test_prompt_and_same_id_requirement_changes_expire_responses(self):
        score = score_fixture(self.image, self.reference, "recommendation")
        self.image["recommendation_requirement"]["constraints"]["budget"]["amount"] = "5"
        with self.assertRaises(ValueError):
            ev.validate_judgement(score, self.reference, self.image, "recommendation")
        with self.assertRaises(ValueError):
            ev.validate_reference(self.reference, self.image)
        score = score_fixture(self.image, self.reference, "summary")
        self.image["prompts"]["summary"]["system"] = "新的提示词"
        with self.assertRaises(ValueError):
            ev.validate_judgement(score, self.reference, self.image, "summary")

    def test_duplicate_constraint_ids_and_unsupported_diet_met_are_rejected(self):
        requirement = self.image["recommendation_requirement"]
        requirement["constraints"]["hard_constraints"] = [{"id": "budget"}]
        with self.assertRaises(ValueError):
            ev.constraint_ids(requirement)
        requirement["constraints"]["hard_constraints"] = [{"id": "no_nuts"}]
        score = score_fixture(self.image, self.reference, "recommendation")
        score["recommendation"]["requirement_checks"].append({"constraint_id": "no_nuts", "status": "met", "reason": "认为无坚果", "evidence_fact_ids": []})
        with self.assertRaises(ValueError):
            ev.validate_judgement(score, self.reference, self.image, "recommendation")

    def test_unknown_constraints_cannot_be_confirmed_success(self):
        score = score_fixture(self.image, self.reference, "recommendation")
        score["recommendation"]["requirement_checks"][0]["status"] = "unknown"
        report = ev.build_report([self.image], {"img001": self.reference}, {("img001", "recommendation"): score}, self.output)
        self.assertEqual(report["tasks"]["recommendation"]["recommendation_success_rate"], 0)
        self.assertEqual(report["tasks"]["recommendation"]["hard_constraint_unknown_rate"], 0.5)

    def test_stale_scores_are_visible_but_excluded(self):
        score = score_fixture(self.image, self.reference, "summary")
        score["candidate_sha256"] = "0" * 64
        report = ev.build_report([self.image], {"img001": self.reference}, {("img001", "summary"): score}, self.output)
        metrics = report["tasks"]["summary"]
        self.assertEqual(metrics["judge_stale"], 1)
        self.assertEqual(metrics["judge_evaluated"], 0)
        self.assertIsNone(metrics["quality_score"]["mean"])

    def test_invalid_image_without_hash_is_retained_in_report(self):
        invalid = copy.deepcopy(self.image)
        invalid.update({"image_id": "invalid001", "image_status": "error", "image_sha256": None, "image_error": "无法读取文件"})
        for task in ev.TASKS:
            invalid[task] = None
        ev.write_jsonl(self.input, [self.image, invalid])
        rows = ev.load_generation(self.input)
        report = ev.build_report(rows, {}, {}, self.output)
        self.assertEqual(report["tasks"]["summary"]["expected_pairs"], 2)
        self.assertEqual(report["tasks"]["summary"]["invalid_image_pairs"], 1)

    def test_full_offline_workflow_and_flat_outputs(self):
        reference_responses = self.root / "gt-responses.jsonl"
        ev.write_jsonl(reference_responses, [self.reference])
        result = ev.main(["reference", "--input", str(self.input), "--output-dir", str(self.output), "--image-root", str(self.root / "pics"), "--responses", str(reference_responses)])
        self.assertEqual(result["imported"], 1)
        score_responses = self.root / "score-responses.jsonl"
        ev.write_jsonl(score_responses, [score_fixture(self.image, self.reference, task) for task in ev.TASKS])
        result = ev.main(["score", "--input", str(self.input), "--ground-truth", str(self.output / "ground_truth.jsonl"), "--output-dir", str(self.output), "--responses", str(score_responses)])
        self.assertEqual(result["evaluated"], 3)
        self.assertEqual(result["generated"], 3)
        metrics = json.loads((self.output / "metrics.json").read_text(encoding="utf-8"))
        self.assertEqual(metrics["tasks"]["recommendation"]["recommendation_success_rate"], 1)
        self.assertTrue((self.output / "report.html").exists())
        self.assertFalse(any(path.is_dir() for path in self.output.iterdir()))

    def test_report_without_gpt_shows_unmeasured_and_escapes_payload(self):
        self.image["summary"]["text"] = "<script>alert('x')</script>"
        report = ev.build_report([self.image], {}, {}, self.output)
        self.assertEqual(report["tasks"]["summary"]["judge_evaluated"], 0)
        self.assertIsNone(report["tasks"]["summary"]["quality_score"]["mean"])
        rendered = (self.output / "report.html").read_text(encoding="utf-8")
        self.assertNotIn("<script>alert('x')</script>", rendered)
        self.assertIn("未测", rendered)

    def test_standalone_copy_runs_without_repo_or_site_packages(self):
        target = self.root / "isolated" / "evaluate.py"
        target.parent.mkdir()
        shutil.copy2(ROOT / "evaluate.py", target)
        command = [sys.executable, "-S", str(target), "report", "--input", str(self.input), "--output-dir", str(self.root / "isolated-output")]
        completed = subprocess.run(command, cwd=target.parent, capture_output=True, text=True, encoding="utf-8", env={**__import__("os").environ, "PYTHONIOENCODING": "utf-8"})
        self.assertEqual(completed.returncode, 0, completed.stderr)
        self.assertTrue((self.root / "isolated-output" / "report.html").exists())

    def test_gpt_stub_explicitly_fails_without_network(self):
        with self.assertRaises(NotImplementedError):
            ev.call_gpt({"stage": "reference"})


if __name__ == "__main__":
    unittest.main()
