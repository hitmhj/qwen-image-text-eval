"""从真实生成入口的虚拟运行到独立脚本，验证交接文件与20图分组。"""

import importlib.util
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from PIL import Image

from qwen_text_eval.cli import main as generate_main
from qwen_text_eval.storage import read_json, read_jsonl

ROOT = Path(__file__).resolve().parents[1]


def load_script(name, filename):
    spec = importlib.util.spec_from_file_location(name, ROOT / filename)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class FakeRuntime:
    info = {"model_class": "阶段交接测试运行器"}

    def generate(self, image_path, prompt, task, seed):
        return {"text": "图片中的菜单\n菜单文字可读。" if task == "summary" else "图中未提供充分费用信息，暂无法确认符合要求。",
                "raw_text": "测试原始输出", "latency_seconds": 0.01, "input_tokens": 20, "output_tokens": 15,
                "finish_reason": "stop"}

    def cleanup_after_error(self):
        pass


class StageHandoffTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.base = Path(self.temporary.name)
        self.model = self.base / "模型"
        self.model.mkdir()
        (self.model / "config.json").write_text('{"model_type":"qwen3_5_moe"}', encoding="utf-8")
        self.primary = self.base / "正式图片"
        folder = self.primary / "英语" / "截图"
        folder.mkdir(parents=True)
        Image.new("RGB", (45, 35), "white").save(folder / "测试.png")
        self.gen = load_script("handoff_generate", "generate.py")
        self.ev = load_script("handoff_evaluate", "evaluate.py")

    def test_both_datasets_generate_and_prepare_independent_flat_reports(self):
        output = self.base / "生成"
        with patch("qwen_text_eval.generation._load_runtime", side_effect=lambda *args: FakeRuntime()):
            result = self.gen.main(["--model-dir", str(self.model), "--primary-images", str(self.primary),
                                    "--output-root", str(output)])
        self.assertEqual(result, 0)
        for dataset_id, images, count in [("primary", self.primary, 1), ("menu_reference", ROOT / "datasets" / "menu_reference", 20)]:
            generation = output / dataset_id / "generation_results.jsonl"
            rows = self.ev.load_generation(generation)
            self.assertEqual(len(rows), count)
            self.assertEqual({row["dataset_id"] for row in rows}, {dataset_id})
            evaluation = self.base / "评测" / dataset_id
            prepared = self.ev.main(["reference", "--input", str(generation), "--image-root", str(images),
                                      "--output-dir", str(evaluation)])
            self.assertEqual(prepared["prepared"], count)
            rendered = self.ev.main(["report", "--input", str(generation), "--image-root", str(images),
                                     "--output-dir", str(evaluation)])
            self.assertEqual(rendered["evaluated"], 0)
            self.assertEqual(rendered["generated"], count * 3)
            stats = read_json(evaluation / "metrics.json")
            self.assertEqual(stats["dataset_id"], dataset_id)
            for task in ["summary", "interpretation", "recommendation"]:
                self.assertEqual(stats["tasks"][task]["expected_pairs"], count)
                self.assertEqual(stats["tasks"][task]["generation_ok"], count)
                self.assertIsNone(stats["tasks"][task]["quality_score"]["mean"])
            self.assertTrue((evaluation / "report.html").is_file())
            self.assertIn("data:image/", (evaluation / "report.html").read_text(encoding="utf-8"))
            self.assertFalse(any(path.is_dir() for path in evaluation.iterdir()))

    def test_data_roots_cannot_overlap(self):
        with self.assertRaises(SystemExit):
            self.gen.main(["--model-dir", str(self.model), "--primary-images", str(self.primary),
                           "--menu-images", str(self.primary), "--output-root", str(self.base / "输出")])

    def test_existing_run_cannot_be_relabelled_to_other_dataset(self):
        run = self.base / "primary"
        self.assertEqual(generate_main(["scan", "--image-root", str(self.primary), "--run-dir", str(run),
                                       "--dataset-id", "primary"]), 0)
        with patch("qwen_text_eval.generation._load_runtime", side_effect=AssertionError("不能加载模型")):
            self.assertEqual(generate_main(["generate", "--model-dir", str(self.model), "--image-root", str(self.primary),
                                            "--run-dir", str(run), "--dataset-id", "menu_reference"]), 2)


if __name__ == "__main__":
    unittest.main()
