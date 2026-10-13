"""交接包复制为一个文件后，原图、任务与数据集身份仍完整可核对。"""

import base64
import hashlib
import importlib.util
import shutil
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from PIL import Image

from qwen_text_eval.cli import load_prompts
from qwen_text_eval.dataset import create_manifest, load_manifest
from qwen_text_eval.exporting import export_generation_results, export_handoff
from qwen_text_eval.storage import atomic_json, fingerprint, read_json, read_jsonl


ROOT = Path(__file__).resolve().parents[1]


class SelfContainedHandoffTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.base = Path(self.temporary.name)
        self.prompts = load_prompts()

    def make_dataset(self, dataset_id, invalid=False, disguised=False):
        images = self.base / f"原图_{dataset_id}"
        folder = images / "英语" / "实拍"
        folder.mkdir(parents=True)
        picture = folder / ("菜单.jpg" if disguised else "菜单.png")
        Image.new("RGB", (47, 31), "white").save(picture, format="PNG")
        if invalid:
            (folder / "无法读取.png").write_bytes(b"not an image")
        run = self.base / "第一平台" / dataset_id
        metadata = create_manifest(images, run, dataset_id)
        for task, prompt in self.prompts.items():
            atomic_json(run / "prompts" / f"{task}.json", prompt)
        for image in load_manifest(run):
            if image["status"] != "ok":
                continue
            for task, prompt in self.prompts.items():
                candidate = {"schema_version": 2, "run_id": metadata["run_id"],
                             "dataset_id": dataset_id, "image_id": image["image_id"],
                             "image_sha256": image["sha256"], "task": task, "status": "ok",
                             "text": "测试生成正文", "raw_text": "测试生成原文",
                             "prompt_sha256": fingerprint(prompt)}
                if task == "recommendation":
                    candidate["recommendation_requirement"] = {"id": "req_test", "text": "总预算50元，不吃辣"}
                atomic_json(run / "samples" / image["image_id"] / f"{task}.json", candidate)
        return dataset_id, run, images

    def test_copy_only_one_file_preserves_original_bytes_and_all_three_results(self):
        datasets = [self.make_dataset("primary"), self.make_dataset("menu_reference")]
        before = {}
        expected_bytes = {}
        for dataset_id, run, root in datasets:
            before[dataset_id] = read_jsonl(Path(export_generation_results(run)["path"]))
            expected_bytes[dataset_id] = {(row["image_id"]): (root / row["relative_path"]).read_bytes()
                                          for row in before[dataset_id]}
        output = self.base / "第一平台" / "handoff.json"
        stats = export_handoff(datasets, output)
        second = self.base / "另外一台机器"
        second.mkdir()
        shutil.copy2(output, second / "handoff.json")
        shutil.rmtree(self.base / "第一平台")
        for _, _, images in datasets:
            shutil.rmtree(images)
        artifact = read_json(second / "handoff.json")
        self.assertEqual(artifact["schema_version"], 1)
        self.assertEqual(artifact["artifact_type"], "qwen_eval_handoff")
        self.assertEqual(stats["embedded_image_count"], 2)
        self.assertEqual(stats["expected_task_count"], 6)
        self.assertEqual({ds["dataset_id"] for ds in artifact["datasets"]}, {"primary", "menu_reference"})
        for ds in artifact["datasets"]:
            for index, row in enumerate(ds["generation_results"]):
                payload = row.pop("image")
                data = base64.b64decode(payload["base64"], validate=True)
                self.assertEqual(data, expected_bytes[ds["dataset_id"]][row["image_id"]])
                self.assertEqual(hashlib.sha256(data).hexdigest(), payload["sha256"])
                self.assertEqual(payload["sha256"], row["image_sha256"])
                self.assertEqual(payload["mime_type"], "image/png")
                self.assertEqual(row, before[ds["dataset_id"]][index])
                for task in ["summary", "interpretation", "recommendation"]:
                    self.assertEqual(row[task]["image_sha256"], payload["sha256"])
                    self.assertEqual(row[task]["prompt_sha256"], fingerprint(row["prompts"][task]))
        self.assertEqual([p.name for p in second.iterdir()], ["handoff.json"])

    def test_invalid_picture_has_no_payload_and_keeps_diagnostic(self):
        dataset = self.make_dataset("primary", invalid=True)
        output = self.base / "handoff.json"
        stats = export_handoff([dataset], output)
        rows = read_json(output)["datasets"][0]["generation_results"]
        self.assertEqual(stats["image_count"], 2)
        self.assertEqual(stats["embedded_image_count"], 1)
        invalid = next(row for row in rows if row["image_status"] == "error")
        self.assertIsNone(invalid["image"])
        self.assertTrue(invalid["image_error"])
        self.assertTrue(all(invalid[task] is None for task in ["summary", "interpretation", "recommendation"]))

    def test_changed_original_cannot_replace_existing_handoff(self):
        dataset = self.make_dataset("primary")
        output = self.base / "handoff.json"
        export_handoff([dataset], output)
        previous = output.read_bytes()
        picture = next(dataset[2].rglob("*.png"))
        Image.new("RGB", (47, 31), "red").save(picture)
        with self.assertRaisesRegex(ValueError, "哈希不一致"):
            export_handoff([dataset], output)
        self.assertEqual(output.read_bytes(), previous)
        self.assertEqual(list(output.parent.glob(".handoff.json.*")), [])

    def test_mime_comes_from_image_bytes_and_dataset_ids_cannot_mix(self):
        dataset = self.make_dataset("primary", disguised=True)
        output = self.base / "handoff.json"
        export_handoff([dataset], output)
        row = read_json(output)["datasets"][0]["generation_results"][0]
        self.assertEqual(row["image"]["mime_type"], "image/png")
        with self.assertRaisesRegex(ValueError, "标识与运行记录不一致"):
            export_handoff([("menu_reference", dataset[1], dataset[2])], output)
        with self.assertRaisesRegex(ValueError, "标识不能重复"):
            export_handoff([dataset, dataset], output)

    def test_top_level_generate_automatically_exports_selected_datasets(self):
        primary = self.make_dataset("primary")
        menu = self.make_dataset("menu_reference")
        spec = importlib.util.spec_from_file_location("self_contained_generate", ROOT / "generate.py")
        generate = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(generate)
        with patch.object(generate, "generate_main", return_value=0):
            result = generate.main(["--model-dir", str(self.base / "模型"),
                                    "--primary-images", str(primary[2]), "--menu-images", str(menu[2]),
                                    "--output-root", str(primary[1].parent)])
        self.assertEqual(result, 0)
        artifact = read_json(primary[1].parent / "handoff.json")
        self.assertEqual(len(artifact["datasets"]), 2)


if __name__ == "__main__":
    unittest.main()
