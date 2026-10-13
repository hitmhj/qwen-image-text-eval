"""验证三任务独立输入、固定推荐要求和两数据集身份隔离。"""

import copy
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from PIL import Image

from qwen_text_eval.cli import assets_root, load_prompts
from qwen_text_eval.dataset import create_manifest
from qwen_text_eval.exporting import export_generation_results
from qwen_text_eval.generation import run_generation
from qwen_text_eval.requirements import select_requirement
from qwen_text_eval.storage import read_json, read_jsonl


class FakeRuntime:
    info = {"model_class": "三任务虚拟运行器"}

    def __init__(self):
        self.calls = []

    def generate(self, image_path, prompt, task, seed):
        self.calls.append((task, prompt, seed))
        return {"text": "原样测试正文", "raw_text": "原样测试正文", "latency_seconds": 0.1,
                "input_tokens": 10, "output_tokens": 8, "finish_reason": "stop"}

    def cleanup_after_error(self):
        pass


class ThreeTaskTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.base = Path(self.temporary.name)
        self.images = self.base / "图像"
        folder = self.images / "英语" / "实拍"
        folder.mkdir(parents=True)
        Image.new("RGB", (50, 40), "white").save(folder / "菜单.png")
        self.model = self.base / "模型"
        self.model.mkdir()
        (self.model / "config.json").write_text('{"model_type":"qwen3_5_moe"}', encoding="utf-8")
        self.prompts = load_prompts()
        self.pool = read_json(assets_root() / "configs" / "requirements_pool.json")

    def generate(self, dataset_id):
        run = self.base / dataset_id
        create_manifest(self.images, run, dataset_id)
        for task, prompt in self.prompts.items():
            folder = run / "prompts"
            folder.mkdir(exist_ok=True)
            (folder / f"{task}.json").write_text(json.dumps(prompt, ensure_ascii=False), encoding="utf-8")
        runtime = FakeRuntime()
        with patch("qwen_text_eval.generation._load_runtime", return_value=runtime):
            stats = run_generation(run, self.model, self.images, {}, self.prompts, requirements_pool=self.pool)
        return run, runtime, stats

    def test_three_independent_calls_and_requirement_only_in_recommendation(self):
        run, runtime, stats = self.generate("primary")
        self.assertEqual(stats["generated_ok"], 3)
        self.assertEqual([task for task, _, _ in runtime.calls], ["summary", "interpretation", "recommendation"])
        self.assertEqual(runtime.calls[0][1]["user"], self.prompts["summary"]["user"])
        self.assertEqual(runtime.calls[1][1]["user"], self.prompts["interpretation"]["user"])
        self.assertIn("点餐要求：", runtime.calls[2][1]["user"])
        self.assertNotIn("{requirement}", runtime.calls[2][1]["user"])
        exported = export_generation_results(run)
        rows = read_jsonl(Path(exported["path"]))
        self.assertEqual(rows[0]["dataset_id"], "primary")
        self.assertIsNotNone(rows[0]["recommendation_requirement"])
        self.assertEqual(rows[0]["recommendation_requirement"], rows[0]["recommendation"]["recommendation_requirement"])
        self.assertEqual(exported["expected_task_count"], 3)

    def test_cohorts_are_separate_and_same_image_keeps_same_requirement(self):
        primary, _, _ = self.generate("primary")
        auxiliary, _, _ = self.generate("menu_reference")
        one = read_jsonl(Path(export_generation_results(primary)["path"]))[0]
        two = read_jsonl(Path(export_generation_results(auxiliary)["path"]))[0]
        self.assertNotEqual(one["dataset_id"], two["dataset_id"])
        self.assertEqual(one["recommendation_requirement"], two["recommendation_requirement"])
        self.assertNotEqual(one["recommendation"]["settings_fingerprint"], two["recommendation"]["settings_fingerprint"])

    def test_pool_or_fx_change_cannot_resume_old_run(self):
        run, _, _ = self.generate("primary")
        changed = copy.deepcopy(self.pool)
        changed["requirements"][0]["text"] += "请注意。"
        with self.assertRaisesRegex(ValueError, "已变化"):
            run_generation(run, self.model, self.images, {}, self.prompts, requirements_pool=changed)

    def test_selection_is_stable_and_no_reroll_for_hard_requirements(self):
        first = select_requirement("1" * 64, self.pool, 7)
        self.assertEqual(first, select_requirement("1" * 64, self.pool, 7))
        ids = {select_requirement(f"{number:064x}", self.pool, 7)["id"] for number in range(500)}
        self.assertEqual(ids, {row["id"] for row in self.pool["requirements"]})

    def test_bundled_menus_have_only_twenty_unmodified_images(self):
        root = assets_root() / "datasets" / "menu_reference"
        metadata = read_json(root / "图片清单.json")
        from qwen_text_eval.storage import sha256_file
        self.assertEqual(len(metadata), 20)
        for row in metadata:
            self.assertFalse(row["ai_generated"])
            self.assertFalse(row["contains_pointing_hand"])
            self.assertEqual(sha256_file(root / row["file"]), row["sha256"])
        self.assertEqual(len(list(root.rglob("*.jpg"))), 20)


if __name__ == "__main__":
    unittest.main()
