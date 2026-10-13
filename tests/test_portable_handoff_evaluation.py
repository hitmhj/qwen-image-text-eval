"""第二平台仅依赖一个交接文件：真实像素、盲参考、独立队列及两阶段恢复。"""
import base64
import copy
import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from tests.test_standalone_eval import ev, image_fixture, score_fixture


PNG = base64.b64decode("iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO+jL1kAAAAASUVORK5CYII=")


class PortableHandoffTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        row, reference = image_fixture(self.root)
        digest = hashlib.sha256(PNG).hexdigest()
        row["image_sha256"] = reference["image_sha256"] = digest
        row["relative_path"] = "已不存在的第一平台目录/菜单.png"
        row["image"] = {"mime_type": "image/png", "base64": base64.b64encode(PNG).decode("ascii"), "sha256": digest}
        for task in ev.TASKS:
            row[task]["image_sha256"] = digest
        self.datasets, self.references = [], {}
        for dataset_id in ev.DATASETS:
            image = copy.deepcopy(row)
            image["dataset_id"] = dataset_id
            image["run_id"] = dataset_id + "-run"
            ref = copy.deepcopy(reference)
            ref["dataset_id"] = dataset_id
            self.datasets.append({"dataset_id": dataset_id, "run_id": image["run_id"], "generation_results": [image]})
            self.references[dataset_id] = ref
        self.package = {"schema_version": 1, "artifact_type": "qwen_eval_handoff", "datasets": self.datasets}
        self.input = self.root / "portable" / "handoff.json"
        self.save()

    def tearDown(self):
        self.temp.cleanup()

    def save(self):
        ev.atomic_json(self.input, self.package)

    def results(self):
        return json.loads((self.input.parent / "results.json").read_text(encoding="utf-8"))

    def test_single_command_flat_outputs_and_actual_pixel_attachments(self):
        result = ev.main([str(self.input)])
        self.assertEqual(result["stage"], "awaiting_reference")
        self.assertEqual(result["network_calls"], 0)
        self.assertEqual(set(path.name for path in self.input.parent.iterdir()), {"handoff.json", "results.json", "report.html", "reference_requests.json"})
        requests = json.loads((self.input.parent / "reference_requests.json").read_text(encoding="utf-8"))
        self.assertEqual(len({request["request_id"] for request in requests}), 2)
        for request in requests:
            self.assertNotIn("candidate_text", request["input"])
            self.assertNotIn("prompts", request["input"])
            self.assertNotIn("local_path", request["attachments"][0])
            self.assertEqual(base64.b64decode(request["attachments"][0]["data_url"].split(",", 1)[1]), PNG)
        for report in self.results()["datasets"].values():
            self.assertEqual(set(report["tasks"]), set(ev.TASKS))
            for stats in report["tasks"].values():
                self.assertEqual(stats["judge_evaluated"], 0)
                self.assertIsNone(stats.get("accuracy", {}).get("mean"))
        document = (self.input.parent / "report.html").read_text(encoding="utf-8")
        self.assertIn("data:image/png;base64,", document)
        self.assertIn("两套数据分别统计，不计算混合总分", document)

    def test_image_corruption_format_and_encoding_fail_closed(self):
        for mutation in [
            {"base64": self.datasets[0]["generation_results"][0]["image"]["base64"] + "!"},
            {"sha256": "0" * 64},
            {"mime_type": "image/jpeg"},
            {"mime_type": "image/svg+xml"},
        ]:
            bad = copy.deepcopy(self.package)
            bad["datasets"][0]["generation_results"][0]["image"].update(mutation)
            ev.atomic_json(self.input, bad)
            with self.assertRaises(ValueError):
                ev.main([str(self.input)])
        self.assertFalse((self.input.parent / "results.json").exists())

    def test_cross_dataset_row_and_run_mismatch_are_rejected(self):
        for field, value in [("dataset_id", "menu_reference"), ("run_id", "wrong-run")]:
            bad = copy.deepcopy(self.package)
            bad["datasets"][0]["generation_results"][0][field] = value
            ev.atomic_json(self.input, bad)
            with self.assertRaises(ValueError):
                ev.load_handoff(self.input)

    def response_files(self):
        refs_path = self.root / "references.json"
        scores_path = self.root / "scores.json"
        ev.atomic_json(refs_path, list(self.references.values()))
        scores = []
        for dataset in self.datasets:
            dataset_id = dataset["dataset_id"]
            row = dataset["generation_results"][0]
            for task in ev.TASKS:
                score = score_fixture(row, self.references[dataset_id], task)
                score["source_evidence_mode"] = "image_and_reference"
                if dataset_id == "primary":
                    for claim in score["claims"]:
                        claim["verdict"] = "contradicted"
                    score["covered_fact_ids"] = []
                    for entity in score["entities"]:
                        entity["verdict"] = "contradicted"
                    for check in score["point_checks"]:
                        check.update(status="missed", accurate=False, claim_ids=[])
                scores.append(score)
        ev.atomic_json(scores_path, scores)
        return refs_path, scores_path

    def test_real_response_import_collision_isolated_scores_and_resume(self):
        refs, scores = self.response_files()
        ev.main([str(self.input), "--references", str(refs), "--stage", "reference"])
        first = self.results()
        self.assertEqual(first["status"], "reference_ready")
        self.assertFalse((self.input.parent / "score_requests.json").exists())
        result = ev.main([str(self.input), "--scores", str(scores), "--stage", "score"])
        self.assertEqual(result["stage"], "evaluated")
        actual = self.results()
        self.assertEqual(len(actual["references"]), 2)
        self.assertEqual(len(actual["scores"]), 6)
        self.assertEqual(actual["datasets"]["primary"]["tasks"]["summary"]["accuracy"]["mean"], 0)
        self.assertEqual(actual["datasets"]["menu_reference"]["tasks"]["summary"]["accuracy"]["mean"], 1)
        self.assertEqual(set(path.name for path in self.input.parent.iterdir()), {"handoff.json", "results.json", "report.html"})
        self.assertEqual(ev.main([str(self.input)])["stage"], "evaluated")

    def api_fixture(self, request, image_path=None, model=None, version=None):
        self.assertIsNone(image_path)
        self.assertTrue(request["attachments"][0]["data_url"].startswith("data:image/png;base64,"))
        dataset_id = request["input"]["dataset_id"]
        if request["stage"] == "reference":
            self.assertNotIn("candidate_text", request["input"])
            reference = copy.deepcopy(self.references[dataset_id])
            reference["annotator"] = request["input"]["annotator"]
            return reference
        row = next(dataset for dataset in self.datasets if dataset["dataset_id"] == dataset_id)["generation_results"][0]
        score = score_fixture(row, request["input"]["reference"], request["input"]["task"])
        score["source_evidence_mode"] = "image_and_reference"
        score["judge"] = request["input"]["judge"]
        return score

    def test_mock_api_reference_then_scores_independent_and_cached(self):
        with patch.object(ev, "call_gpt", side_effect=self.api_fixture) as adapter:
            result = ev.main([str(self.input), "--call-api", "--gpt-model", "fixture-model", "--gpt-version", "v1"])
            self.assertEqual(result["stage"], "evaluated")
            self.assertEqual(result["network_calls"], 8)
            self.assertEqual([call.args[0]["stage"] for call in adapter.call_args_list], ["reference", "reference"] + ["judge"] * 6)
        with patch.object(ev, "call_gpt", side_effect=AssertionError("不应重复调用")):
            self.assertEqual(ev.main([str(self.input), "--call-api", "--gpt-model", "fixture-model", "--gpt-version", "v1"])["network_calls"], 0)

    def test_two_api_stages_resume_without_other_input_files(self):
        options = ["--call-api", "--gpt-model", "fixture-model", "--gpt-version", "v1"]
        with patch.object(ev, "call_gpt", side_effect=self.api_fixture):
            result = ev.main([str(self.input), "--stage", "reference", *options])
            self.assertEqual(result["network_calls"], 2)
            self.assertEqual(result["stage"], "reference_ready")
            result = ev.main([str(self.input), "--stage", "score", *options])
            self.assertEqual(result["network_calls"], 6)
            self.assertEqual(result["stage"], "evaluated")

    def test_api_failure_checkpoints_and_resumes_only_missing_work(self):
        options = [str(self.input), "--call-api", "--gpt-model", "fixture-model", "--gpt-version", "v1"]
        seen = []
        def failing(request, **kwargs):
            seen.append(request["request_id"])
            if len(seen) == 4:
                raise OSError("模拟第四次请求网络中断")
            return self.api_fixture(request, **kwargs)
        with patch.object(ev, "call_gpt", side_effect=failing):
            with self.assertRaises(OSError):
                ev.main(options)
        saved = self.results()
        self.assertEqual(saved["status"], "in_progress")
        self.assertEqual(len(saved["references"]), 2)
        self.assertEqual(len(saved["scores"]), 1)
        with patch.object(ev, "call_gpt", side_effect=self.api_fixture) as adapter:
            result = ev.main(options)
            self.assertEqual(result["network_calls"], 5)
            self.assertEqual(result["stage"], "evaluated")
            self.assertTrue(all(call.args[0]["request_id"] not in seen[:3] for call in adapter.call_args_list))

    def test_api_cannot_spoof_model_metadata_or_human_verification(self):
        options = [str(self.input), "--call-api", "--gpt-model", "fixture-model", "--gpt-version", "v1"]
        def bad_reference(request, **kwargs):
            response = self.api_fixture(request, **kwargs)
            response["annotator"] = {"model": "invented", "version": "invented"}
            return response
        with patch.object(ev, "call_gpt", side_effect=bad_reference):
            with self.assertRaises(ValueError):
                ev.main(options)
        self.assertFalse((self.input.parent / "results.json").exists())

    def test_missing_reference_and_unimplemented_api_do_not_fake_scores(self):
        with self.assertRaises(ValueError):
            ev.main([str(self.input), "--stage", "score"])
        with self.assertRaises(NotImplementedError):
            ev.main([str(self.input), "--call-api", "--gpt-model", "actual-model", "--gpt-version", "v1"])
        self.assertFalse((self.input.parent / "results.json").exists())

    def test_changed_handoff_or_rubric_cannot_reuse_previous_scores(self):
        ev.main([str(self.input)])
        self.package["datasets"][0]["generation_results"][0]["summary"]["text"] += "内容变化"
        self.save()
        with self.assertRaises(ValueError):
            ev.main([str(self.input)])


if __name__ == "__main__":
    unittest.main()
