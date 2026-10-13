"""完整评估.py的SDK请求、配置、独立参考、评分与单文件交付。"""

import base64
import copy
import hashlib
import importlib.util
import json
import os
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from tests.test_standalone_eval import image_fixture, score_fixture

ROOT = Path(__file__).resolve().parents[1]
PNG = base64.b64decode("iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO+jL1kAAAAASUVORK5CYII=")


class HMCompleteEvaluatorTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.base = Path(self.temp.name)
        spec = importlib.util.spec_from_file_location("complete_hm_evaluator", ROOT / "评估.py")
        self.ev = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(self.ev)
        row, reference = image_fixture(self.base)
        digest = hashlib.sha256(PNG).hexdigest()
        row["image_sha256"] = reference["image_sha256"] = digest
        row["relative_path"] = "不可访问的第一平台目录/菜单.png"
        row["image"] = {"mime_type": "image/png", "base64": base64.b64encode(PNG).decode("ascii"), "sha256": digest}
        for task in self.ev.TASKS:
            row[task]["image_sha256"] = digest
        self.datasets, self.references = [], {}
        for dataset_id in self.ev.DATASETS:
            image = copy.deepcopy(row)
            image["dataset_id"], image["run_id"] = dataset_id, dataset_id + "-run"
            ref = copy.deepcopy(reference)
            ref["dataset_id"] = dataset_id
            self.datasets.append({"dataset_id": dataset_id, "run_id": image["run_id"], "generation_results": [image]})
            self.references[dataset_id] = ref
        self.input = self.base / "handoff.json"
        self.ev.atomic_json(self.input, {"schema_version": 1, "artifact_type": "qwen_eval_handoff", "datasets": self.datasets})
        self.calls = []

    def configure(self):
        self.ev.DEFAULT_ACCOUNT = "测试子账号"
        self.ev.DEFAULT_MODEL = "测试模型"
        self.ev.DEFAULT_MODEL_VERSION = "测试版本"
        self.ev.DEFAULT_MODEL_AGENT = "测试代理"
        self.ev.hm_aigc = SimpleNamespace(aigc_managed=self.sdk_response)

    def sdk_response(self, **kwargs):
        self.calls.append(kwargs)
        body = kwargs["req_data"]
        self.assertEqual(body["messages"][0]["role"], "system")
        content = body["messages"][1]["content"]
        self.assertEqual(base64.b64decode(content[1]["image_url"]["url"].split(",", 1)[1]), PNG)
        payload = json.loads(content[0]["text"])["input"]
        dataset_id = payload["dataset_id"]
        if "annotator" in payload:
            self.assertNotIn("candidate_text", payload)
            self.assertNotIn("prompts", payload)
            self.assertEqual(body["messages"][0]["content"], self.ev.REFERENCE_SYSTEM)
            response = copy.deepcopy(self.references[dataset_id])
            response["annotator"] = payload["annotator"]
        else:
            self.assertEqual(body["messages"][0]["content"], self.ev.JUDGE_SYSTEM)
            image = next(item for item in self.datasets if item["dataset_id"] == dataset_id)["generation_results"][0]
            response = score_fixture(image, payload["reference"], payload["task"])
            response["source_evidence_mode"] = "image_and_reference"
            response["judge"] = payload["judge"]
        return {"choices": [{"finish_reason": "stop", "message": {
            "content": "```json\n" + json.dumps(response, ensure_ascii=False) + "\n```"}}]}

    def test_configured_single_command_calls_actual_adapter_for_both_datasets(self):
        self.configure()
        result = self.ev.main([str(self.input)])
        self.assertEqual(result["stage"], "evaluated")
        self.assertEqual(len(self.calls), 8)
        for call in self.calls:
            self.assertEqual(call["sub_account_name"], "测试子账号")
            self.assertEqual(call["model_agent"], "测试代理")
            self.assertEqual(call["model"], "测试模型")
            self.assertEqual(call["timeout"], self.ev.DEFAULT_TIMEOUT)
        stages = ["reference" if "annotator" in json.loads(call["req_data"]["messages"][1]["content"][0]["text"])["input"] else "judge" for call in self.calls]
        self.assertEqual(stages, ["reference", "reference"] + ["judge"] * 6)
        saved = json.loads((self.base / "results.json").read_text(encoding="utf-8"))
        self.assertEqual(set(saved["datasets"]), {"primary", "menu_reference"})
        self.assertEqual(len(saved["references"]), 2)
        self.assertEqual(len(saved["scores"]), 6)
        self.assertTrue((self.base / "report.html").is_file())
        self.ev.main([str(self.input)])
        self.assertEqual(len(self.calls), 8)

    def test_two_stages_use_one_results_file(self):
        self.configure()
        self.assertEqual(self.ev.main([str(self.input), "--stage", "reference"])["stage"], "reference_ready")
        self.assertEqual(len(self.calls), 2)
        self.assertEqual(self.ev.main([str(self.input), "--stage", "score"])["stage"], "evaluated")
        self.assertEqual(len(self.calls), 8)

    def test_prepare_only_never_calls_sdk_even_with_configuration(self):
        self.configure()
        self.assertEqual(self.ev.main([str(self.input), "--prepare-only"])["network_calls"], 0)
        self.assertEqual(self.calls, [])

    def test_invalid_json_is_retried_then_validated(self):
        self.configure()
        request = self.ev.prepare_reference(self.datasets[0]["generation_results"])[0]
        responses = [{"choices": [{"message": {"content": "非法JSON"}}]}, self.sdk_response(
            req_data={"messages": [{"role": "system", "content": request["system"]},
                {"role": "user", "content": [{"type": "text", "text": json.dumps({"input": {
                    **request["input"], "annotator": {"model": "测试模型", "version": "测试版本"}}})},
                    {"type": "image_url", "image_url": {"url": request["attachments"][0]["data_url"]}}]}]})]
        self.ev.hm_aigc = SimpleNamespace(aigc_managed=lambda **kwargs: responses.pop(0))
        original_input = copy.deepcopy(request["input"])
        with patch.object(self.ev.time, "sleep") as sleep:
            answer = self.ev.call_gpt(request, model="测试模型", version="测试版本")
        self.assertEqual(answer["image_id"], request["input"]["image_id"])
        self.assertEqual(request["input"], original_input)
        sleep.assert_called_once_with(2)

    def test_sdk_import_configuration_and_missing_account_are_clear(self):
        self.configure()
        del self.ev.hm_aigc
        client = SimpleNamespace(aigc_managed=self.sdk_response)
        with patch.object(self.ev.importlib, "import_module", return_value=client):
            self.assertIs(self.ev.get_hm_client(), client)
        self.ev.DEFAULT_ACCOUNT = ""
        with self.assertRaisesRegex(RuntimeError, "DEFAULT_ACCOUNT"):
            self.ev.call_gpt({"stage": "reference"}, model="模型", version="版本")

    def test_complete_file_runs_alone_without_repository_or_site_packages(self):
        isolated = self.base / "独立平台"
        isolated.mkdir()
        shutil.copy2(ROOT / "评估.py", isolated / "评估.py")
        shutil.copy2(self.input, isolated / "handoff.json")
        environment = {key: value for key, value in os.environ.items() if key.upper() != "PYTHONPATH"}
        environment["PYTHONUTF8"] = "1"
        run = subprocess.run([sys.executable, "-S", "评估.py", "handoff.json", "--prepare-only"],
                             cwd=isolated, env=environment, capture_output=True, text=True,
                             encoding="utf-8", timeout=30)
        self.assertEqual(run.returncode, 0, run.stdout + run.stderr)
        self.assertTrue((isolated / "results.json").is_file())
        self.assertTrue((isolated / "report.html").is_file())
        self.assertFalse(any(item.is_dir() for item in isolated.iterdir()))


if __name__ == "__main__":
    unittest.main()
