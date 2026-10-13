"""实际入口跨平台交接：旧路径失效后，只复制两个文件仍可运行。"""

import base64
import importlib.util
import json
import os
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from PIL import Image

ROOT = Path(__file__).resolve().parents[1]


def load_file(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class FakeQwen:
    info = {"model_class": "跨平台交接测试"}

    def generate(self, image_path, prompt, task, seed):
        text = "餐厅菜单\n列有餐品与价格。" if task == "summary" else "图中费用信息不足，暂无法确认符合要求。"
        return {"text": text, "raw_text": text, "latency_seconds": 0.01,
                "input_tokens": 20, "output_tokens": 15, "finish_reason": "stop"}

    def cleanup_after_error(self):
        pass


class HandoffEndToEndTests(unittest.TestCase):
    def test_two_file_copy_runs_with_all_original_paths_unavailable(self):
        with tempfile.TemporaryDirectory() as temporary:
            base = Path(temporary)
            platform_one = base / "第一平台"
            platform_two = base / "第二平台"
            platform_two.mkdir()
            model = platform_one / "模型"
            model.mkdir(parents=True)
            (model / "config.json").write_text('{"model_type":"qwen3_5_moe"}', encoding="utf-8")
            image_roots = [platform_one / "正式图片", platform_one / "辅助菜单"]
            for image_root in image_roots:
                folder = image_root / "英语" / "截图"
                folder.mkdir(parents=True)
                # 两套数据故意使用相同相对路径和像素，检查跨集合身份隔离。
                Image.new("RGB", (48, 32), "white").save(folder / "菜单.png")
            generate = load_file("portable_roundtrip_generate", ROOT / "generate.py")
            output = platform_one / "生成结果"
            with patch("qwen_text_eval.generation._load_runtime", side_effect=lambda *args: FakeQwen()):
                status = generate.main(["--model-dir", str(model), "--primary-images", str(image_roots[0]),
                                        "--menu-images", str(image_roots[1]), "--output-root", str(output)])
            self.assertEqual(status, 0)
            bundle = json.loads((output / "handoff.json").read_text(encoding="utf-8"))
            self.assertEqual([group["dataset_id"] for group in bundle["datasets"]], ["primary", "menu_reference"])
            for group in bundle["datasets"]:
                self.assertEqual(len(group["generation_results"]), 1)
                row = group["generation_results"][0]
                self.assertEqual(row["image"]["mime_type"], "image/png")
                self.assertEqual(base64.b64decode(row["image"]["base64"]),
                                 (image_roots[0] / row["relative_path"]).read_bytes())
            self.assertEqual(bundle["datasets"][0]["generation_results"][0]["image_id"],
                             bundle["datasets"][1]["generation_results"][0]["image_id"])
            shutil.copy2(ROOT / "evaluate.py", platform_two / "evaluate.py")
            shutil.copy2(output / "handoff.json", platform_two / "handoff.json")
            platform_one.rename(base / "第一平台已离线")
            self.assertFalse(platform_one.exists())
            isolated_env = {key: value for key, value in os.environ.items() if key.upper() != "PYTHONPATH"}
            isolated_env["PYTHONUTF8"] = "1"
            executed = subprocess.run([sys.executable, "-S", "evaluate.py", "handoff.json"],
                                      cwd=platform_two, capture_output=True, text=True,
                                      encoding="utf-8", timeout=30,
                                      env=isolated_env)
            self.assertEqual(executed.returncode, 0, executed.stdout + executed.stderr)
            self.assertTrue((platform_two / "report.html").is_file())
            self.assertTrue((platform_two / "results.json").is_file())
            self.assertFalse(any(path.is_dir() for path in platform_two.iterdir()))
            html = (platform_two / "report.html").read_text(encoding="utf-8")
            self.assertIn("primary", html)
            self.assertIn("menu_reference", html)
            self.assertIn("data:image/png;base64,", html)
            self.assertNotIn(str(platform_one), html)
            results = json.loads((platform_two / "results.json").read_text(encoding="utf-8"))
            self.assertEqual(set(results["datasets"]), {"primary", "menu_reference"})
            self.assertEqual(results["pending_reference_count"], 2)
            self.assertEqual(results["references"], [])
            self.assertEqual(results["scores"], [])
            for dataset_id, report in results["datasets"].items():
                self.assertEqual(report["dataset_id"], dataset_id)
                for task in ["summary", "interpretation", "recommendation"]:
                    self.assertEqual(report["tasks"][task]["total_images"], 1)
                    self.assertEqual(report["tasks"][task]["generation_ok"], 1)
                    self.assertIsNone(report["tasks"][task].get("accuracy", {}).get("mean"))


if __name__ == "__main__":
    unittest.main()
