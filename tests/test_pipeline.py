"""合成图和虚拟回包验证阶段隔离、校验及统计分母。"""

import copy
import tempfile
import unittest
from pathlib import Path

from PIL import Image

from qwen_text_eval.cli import assets_root, load_prompts
from qwen_text_eval.dataset import create_manifest, load_manifest
from qwen_text_eval.evaluation import (evaluate_run_rules, import_judgements, import_references,
                                       prepare_judge_requests, prepare_reference_requests)
from qwen_text_eval.reporting import build_report, score_judgement, write_csv
from qwen_text_eval.storage import atomic_json, fingerprint, read_json, read_jsonl, safe_image_path, text_sha256, write_jsonl
from qwen_text_eval.validation import (JUDGE_SCHEMA, REFERENCE_SCHEMA, validate_judgement,
                                      validate_reference, validate_rubric)


class PipelineTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        base = Path(self.temporary.name)
        self.images, self.run = base / "图片", base / "运行"
        one = self.images / "英语" / "截图" / "深层"
        one.mkdir(parents=True)
        Image.new("RGB", (20, 12), "red").save(one / "一.png")
        two = self.images / "日语" / "实拍"
        two.mkdir(parents=True)
        Image.new("RGB", (20, 12), "blue").save(two / "二.png")
        create_manifest(self.images, self.run)
        self.manifest = load_manifest(self.run)
        self.image = self.manifest[0]
        self.assets = assets_root()
        self.prompts = load_prompts()
        self.rubric = read_json(self.assets / "configs" / "rubric.json")
        self.reference = {"schema_version": 1, "image_id": self.image["image_id"], "image_sha256": self.image["sha256"],
                          "scene": "menu", "image_quality": {"readability": "clear", "issues": []},
                          "facts": [self.fact("f001", "name", "餐厅菜单", 3), self.fact("f002", "price", "拉面每份12元", 2)],
                          "unreadable_regions": [], "reference_status": "draft",
                          "annotator": {"model": "虚拟标注器", "version": "测试一"}}

    @staticmethod
    def fact(fid, category, text, importance):
        return {"fact_id": fid, "category": category, "source_text": text, "normalized_zh": text,
                "evidence": "合成标注的第一行", "confidence": "high", "importance": importance,
                "required_for": ["summary", "interpretation"], "numeric": None}

    def add_candidates(self):
        for task, text in [("summary", "餐厅菜单\n拉面每份12元"),
                           ("interpretation", "这是一张餐厅菜单。\n\n【菜品信息】\n- 有拉面。\n- 每份12元。")]:
            candidate = {"schema_version": 1, "run_id": self.run.name, "image_id": self.image["image_id"],
                         "task": task, "prompt_sha256": fingerprint(self.prompts[task]), "image_sha256": self.image["sha256"],
                         "status": "ok", "text": text, "raw_text": text, "latency_seconds": 1.25,
                         "output_tokens": 20, "finish_reason": "stop", "error": None}
            atomic_json(self.run / "samples" / self.image["image_id"] / f"{task}.json", candidate)
            atomic_json(self.run / "prompts" / f"{task}.json", self.prompts[task])

    def add_reference(self):
        path = Path(self.temporary.name) / "参考.jsonl"
        write_jsonl(path, [self.reference])
        import_references(self.run, path)
        prepare_judge_requests(self.run, self.assets / "prompts" / "judge_system.txt", self.rubric, self.prompts)

    def judgement(self, task="summary"):
        candidate = read_json(self.run / "samples" / self.image["image_id"] / f"{task}.json")
        return {"schema_version": 1, "image_id": self.image["image_id"], "task": task,
                "candidate_sha256": text_sha256(candidate["text"]), "reference_sha256": fingerprint(self.reference), "scene": "menu",
                "judge_prompt_sha256": text_sha256((self.assets / "prompts" / "judge_system.txt").read_text(encoding="utf-8")),
                "dimensions": {key: {"score": None if key == "translation" else 4,
                                      "reason": "无翻译需求" if key == "translation" else "合成证据完全支持",
                                      "evidence_fact_ids": [] if key == "translation" else ["f001"]}
                               for key in self.rubric["tasks"][task]["dimensions"]},
                "claims": [{"claim_id": "c001", "text": "餐厅菜单", "verdict": "supported", "fact_ids": ["f001"],
                            "severity": "minor", "explanation": "匹配主体事实", "category": "name"},
                           {"claim_id": "c002", "text": "拉面每份12元", "verdict": "supported", "fact_ids": ["f002"],
                            "severity": "major", "explanation": "匹配菜品、币种及份量", "category": "price"}],
                "covered_fact_ids": ["f001", "f002"], "critical_errors": [], "needs_human_review": False,
                "review_reasons": [], "judge": {"model": "虚拟评审器", "version": "测试一"}}

    def test_nested_folder_metadata_and_unknown_layout(self):
        self.assertEqual({row["language"] for row in self.manifest}, {"英语", "日语"})
        self.assertEqual({row["capture_type"] for row in self.manifest}, {"photo", "screenshot"})
        self.assertTrue(all(row["width"] == 20 and row["height"] == 12 for row in self.manifest))

    def test_duplicate_and_corrupt_images_remain_in_denominator(self):
        source = next(self.images.rglob("一.png"))
        (source.parent / "副本.png").write_bytes(source.read_bytes())
        (source.parent / "损坏.png").write_bytes(b"not an image")
        other = self.run.parent / "第二次"
        info = create_manifest(self.images, other)
        self.assertEqual((info["image_count"], info["duplicate_images"], info["invalid_images"]), (4, 1, 1))
        stats = build_report(other, self.rubric)["tasks"]["summary"]
        self.assertEqual(stats["expected_pairs"], 4)
        self.assertEqual(stats["invalid_image_pairs"], 1)
        self.assertEqual(stats["generation_missing"], 3)

    def test_changed_manifest_is_rejected(self):
        rows = read_jsonl(self.run / "manifest.jsonl")
        rows[0]["language"] = "改写"
        write_jsonl(self.run / "manifest.jsonl", rows)
        with self.assertRaisesRegex(ValueError, "清单"):
            load_manifest(self.run)

    def test_no_candidate_leakage_into_reference_requests(self):
        self.add_candidates()
        result = prepare_reference_requests(self.run, self.assets / "prompts" / "reference_system.txt")
        self.assertEqual(result["request_count"], 2)
        rows = read_jsonl(Path(result["path"]))
        self.assertEqual(set(rows[0]["input"]), {"image_id", "image_sha256"})
        self.assertNotIn("candidate_text", str(rows))
        self.assertNotIn("拉面每份12元", str(rows))

    def test_without_llm_reports_semantic_metrics_as_unmeasured(self):
        self.add_candidates()
        evaluate_run_rules(self.run)
        report = build_report(self.run, self.rubric)
        stats = report["tasks"]["summary"]
        self.assertEqual(stats["generation_ok"], 1)
        self.assertEqual(stats["generation_missing"], 1)
        self.assertEqual(stats["judge_evaluated"], 0)
        self.assertIsNone(stats["quality_score"]["mean"])
        self.assertIsNone(stats["critical_error_rate"])

    def test_complete_two_task_evaluation_and_null_renormalization(self):
        self.add_candidates()
        self.add_reference()
        requests = prepare_judge_requests(self.run, self.assets / "prompts" / "judge_system.txt", self.rubric, self.prompts)
        self.assertEqual(requests["request_count"], 2)
        self.assertEqual(requests["skipped"]["missing_reference"], 2)
        path = Path(self.temporary.name) / "评审.jsonl"
        write_jsonl(path, [self.judgement(task) for task in ["summary", "interpretation"]])
        self.assertEqual(import_judgements(self.run, path, self.rubric)["imported"], 2)
        evaluate_run_rules(self.run)
        report = build_report(self.run, self.rubric)
        for stats in report["tasks"].values():
            self.assertEqual(stats["quality_score"], {"mean": 100.0, "n": 1})
            self.assertEqual(stats["weighted_key_fact_coverage"]["mean"], 1.0)
            self.assertEqual(stats["dimensions"]["translation"]["n"], 0)
            self.assertEqual(stats["human_verified_reference_pairs"], 0)
        self.assertTrue((self.run / "reports" / "review.csv").exists())

    def test_reference_import_prevalidates_whole_batch(self):
        invalid = copy.deepcopy(self.reference)
        invalid["image_id"] = self.manifest[1]["image_id"]
        invalid["image_sha256"] = "0" * 64
        path = Path(self.temporary.name) / "无效参考.jsonl"
        write_jsonl(path, [self.reference, invalid])
        with self.assertRaises(ValueError):
            import_references(self.run, path)
        self.assertFalse((self.run / "references" / f"{self.image['image_id']}.json").exists())

    def test_model_cannot_claim_human_verified(self):
        self.reference["reference_status"] = "human_verified"
        path = Path(self.temporary.name) / "自封核验.jsonl"
        write_jsonl(path, [self.reference])
        with self.assertRaisesRegex(ValueError, "人工核验"):
            import_references(self.run, path)
        import_references(self.run, path, human_verified=True)
        self.assertEqual(read_json(self.run / "references" / f"{self.image['image_id']}.json")["reference_status"], "human_verified")

    def test_outdated_candidate_and_reference_are_rejected(self):
        self.add_candidates()
        candidate = read_json(self.run / "samples" / self.image["image_id"] / "summary.json")
        judge = self.judgement()
        candidate["text"] += "。"
        with self.assertRaisesRegex(ValueError, "候选文本"):
            validate_judgement(judge, self.reference, candidate, self.rubric)
        candidate["text"] = candidate["text"][:-1]
        reference = copy.deepcopy(self.reference)
        reference["facts"][1]["normalized_zh"] = "拉面每份13元"
        with self.assertRaisesRegex(ValueError, "参考标注"):
            validate_judgement(judge, reference, candidate, self.rubric)

    def test_stale_judgements_are_visible_and_excluded_from_means(self):
        self.add_candidates()
        self.add_reference()
        path = Path(self.temporary.name) / "评审.jsonl"
        write_jsonl(path, [self.judgement()])
        import_judgements(self.run, path, self.rubric)
        self.reference["facts"][1]["normalized_zh"] = "拉面每份13元"
        path = Path(self.temporary.name) / "修订.jsonl"
        write_jsonl(path, [self.reference])
        import_references(self.run, path, replace=True)
        report = build_report(self.run, self.rubric)
        self.assertEqual(report["tasks"]["summary"]["judge_stale"], 1)
        self.assertIsNone(report["tasks"]["summary"]["quality_score"]["mean"])

    def test_nonexistent_citations_and_unbacked_coverage_are_rejected(self):
        self.add_candidates()
        candidate = read_json(self.run / "samples" / self.image["image_id"] / "summary.json")
        judge = self.judgement()
        judge["claims"][0]["fact_ids"] = ["不存在"]
        with self.assertRaisesRegex(ValueError, "不存在"):
            validate_judgement(judge, self.reference, candidate, self.rubric)
        judge = self.judgement()
        judge["claims"][1]["verdict"] = "unverifiable"
        with self.assertRaisesRegex(ValueError, "覆盖事实"):
            validate_judgement(judge, self.reference, candidate, self.rubric)

    def test_critical_error_not_hidden_by_score_and_empty_denominators(self):
        self.add_candidates()
        judge = self.judgement()
        judge["claims"][1].update(verdict="unsupported", severity="critical", category="allergen")
        judge["covered_fact_ids"] = ["f001"]
        scored = score_judgement(judge, self.reference, self.rubric)
        self.assertTrue(scored["critical_error"])
        self.assertEqual(scored["quality_score"], 100)
        self.assertAlmostEqual(scored["weighted_key_fact_coverage"], 3 / 5)
        judge["claims"] = []
        judge["covered_fact_ids"] = []
        self.reference["facts"] = []
        scored = score_judgement(judge, self.reference, self.rubric)
        self.assertIsNone(scored["weighted_key_fact_coverage"])
        self.assertIsNone(scored["factual_precision"])
        self.assertIsNone(scored["numeric_claim_accuracy"])

    def test_missing_dimensions_and_fractional_scores_rejected(self):
        self.add_candidates()
        candidate = read_json(self.run / "samples" / self.image["image_id"] / "summary.json")
        judge = self.judgement()
        judge["dimensions"].pop("translation")
        with self.assertRaisesRegex(ValueError, "维度"):
            validate_judgement(judge, self.reference, candidate, self.rubric)
        judge = self.judgement()
        judge["dimensions"]["translation"]["score"] = 2.5
        with self.assertRaises(ValueError):
            validate_judgement(judge, self.reference, candidate, self.rubric)

    def test_unreadable_reference_cannot_demand_key_fact_recall(self):
        self.reference["image_quality"]["readability"] = "unreadable"
        with self.assertRaisesRegex(ValueError, "不可读"):
            validate_reference(self.reference, self.image)

    def test_medium_confidence_cannot_be_required_for_coverage(self):
        self.reference["facts"][0]["confidence"] = "medium"
        with self.assertRaisesRegex(ValueError, "高置信"):
            validate_reference(self.reference, self.image)

    def test_numeric_claim_category_is_required(self):
        self.add_candidates()
        candidate = read_json(self.run / "samples" / self.image["image_id"] / "summary.json")
        judge = self.judgement()
        judge["claims"][1].pop("category")
        with self.assertRaises(ValueError):
            validate_judgement(judge, self.reference, candidate, self.rubric)

    def test_judge_prompt_and_model_versions_cannot_be_mixed(self):
        self.add_candidates()
        self.add_reference()
        prompt = Path(self.temporary.name) / "不同评审.txt"
        prompt.write_text("不同评分口径", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "提示词"):
            prepare_judge_requests(self.run, prompt, self.rubric, self.prompts)
        path = Path(self.temporary.name) / "评审.jsonl"
        write_jsonl(path, [self.judgement()])
        import_judgements(self.run, path, self.rubric)
        changed = self.judgement("interpretation")
        changed["judge"]["version"] = "测试二"
        write_jsonl(path, [changed])
        with self.assertRaisesRegex(ValueError, "模型或版本"):
            import_judgements(self.run, path, self.rubric)
        self.assertFalse((self.run / "judgements" / self.image["image_id"] / "interpretation.json").exists())

    def test_rubric_weights_and_json_schemas_are_valid(self):
        from jsonschema import Draft202012Validator
        validate_rubric(self.rubric)
        Draft202012Validator.check_schema(REFERENCE_SCHEMA)
        Draft202012Validator.check_schema(JUDGE_SCHEMA)
        bad = copy.deepcopy(self.rubric)
        bad["tasks"]["summary"]["dimensions"]["factual_accuracy"]["weight"] = 1
        with self.assertRaisesRegex(ValueError, "100"):
            validate_rubric(bad)

    def test_path_escape_and_csv_formula_are_guarded(self):
        with self.assertRaises(ValueError):
            safe_image_path(self.images, "../外部文件.png")
        target = Path(self.temporary.name) / "输出.csv"
        write_csv(target, [{"text": "=1+1"}], ["text"])
        self.assertIn("'=1+1", target.read_text(encoding="utf-8-sig"))


if __name__ == "__main__":
    unittest.main()
