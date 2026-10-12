"""使用本地假运行器验证断点和原始输出，测试不载入 GPU 依赖。"""

from __future__ import annotations

import contextlib
import json
import os
import tempfile
import unittest
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import Mock, patch

from PIL import Image

from qwen_text_eval.generation import DEFAULT_CONFIG, _QwenRuntime, run_generation
from qwen_text_eval.storage import sha256_file


class FakeRuntime:
    info = {"model_class": "测试运行器"}

    def __init__(self, fail_summary: bool = False):
        self.calls = []
        self.fail_summary = fail_summary

    def generate(self, image_path, prompt, task, seed):
        self.calls.append((image_path, task, seed))
        if task == "summary" and self.fail_summary:
            raise RuntimeError("模拟显存不足")
        return {
            "text": "<think>内部文本</think>\n\n菜单标题\n  描述 \n",
            "raw_text": "<think>内部文本</think>\n\n菜单标题\n  描述 \n<im_end>",
            "latency_seconds": 0.01, "input_tokens": 8, "output_tokens": 16,
            "finish_reason": "stop",
        }

    def cleanup_after_error(self):
        pass


class GenerationTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.run_dir = self.root / "输出"
        self.run_dir.mkdir()
        self.model_dir = self.root / "模型"
        self.model_dir.mkdir()
        (self.model_dir / "config.json").write_text('{"model_type":"qwen3_5_moe"}', encoding="utf-8")
        self.image_root = self.root / "图片"
        self.image_root.mkdir()
        self.image = self.image_root / "菜单.png"
        Image.new("RGB", (32, 32), "white").save(self.image)
        self.record = {
            "image_id": "a1", "relative_path": "菜单.png", "status": "ok",
            "sha256": sha256_file(self.image), "language": "日语", "capture_type": "实拍",
        }
        (self.run_dir / "run.json").write_text('{"run_id":"r1"}', encoding="utf-8")
        self.write_manifest([self.record])
        self.prompts = {
            task: {"version": f"v04_{task}", "task": task, "system": "系统指令", "user": "识别图片"}
            for task in ("summary", "interpretation")
        }

    def write_manifest(self, records):
        (self.run_dir / "manifest.jsonl").write_text(
            "\n".join(json.dumps(record, ensure_ascii=False) for record in records) + "\n", encoding="utf-8"
        )

    def result(self, task):
        return json.loads((self.run_dir / "samples" / "a1" / f"{task}.json").read_text(encoding="utf-8"))

    def run_fake(self, runtime, config=None, resume=True):
        with patch("qwen_text_eval.generation._load_runtime", return_value=runtime):
            return run_generation(self.run_dir, self.model_dir, self.image_root, config or {}, self.prompts, resume)

    def test_two_tasks_preserve_output_and_skip_completed_without_loading_model(self):
        runtime = FakeRuntime()
        stats = self.run_fake(runtime)
        self.assertEqual(stats["generated_ok"], 2)
        self.assertEqual([call[1] for call in runtime.calls], ["summary", "interpretation"])
        self.assertEqual(self.result("summary")["text"], "<think>内部文本</think>\n\n菜单标题\n  描述 \n")
        self.assertTrue(self.result("summary")["raw_text"].endswith("<im_end>"))
        self.assertEqual(os.environ["HF_HUB_OFFLINE"], "1")
        self.assertEqual(os.environ["TRANSFORMERS_OFFLINE"], "1")
        with patch("qwen_text_eval.generation._load_runtime", side_effect=AssertionError("不应加载模型")):
            stats = run_generation(self.run_dir, self.model_dir, self.image_root, {}, self.prompts)
        self.assertEqual(stats["resumed_ok"], 2)
        self.assertEqual(stats["generated_ok"], 0)

    def test_failed_summary_does_not_lose_interpretation_and_is_retried(self):
        stats = self.run_fake(FakeRuntime(fail_summary=True))
        self.assertEqual(stats["generated_error"], 1)
        self.assertEqual(self.result("summary")["finish_reason"], "error")
        self.assertEqual(self.result("interpretation")["status"], "ok")
        retry = FakeRuntime()
        stats = self.run_fake(retry)
        self.assertEqual(stats["resumed_ok"], 1)
        self.assertEqual(stats["generated_ok"], 1)
        self.assertEqual([call[1] for call in retry.calls], ["summary"])

    def test_changed_config_rejected_even_if_resume_is_disabled(self):
        self.run_fake(FakeRuntime())
        with patch("qwen_text_eval.generation._load_runtime", side_effect=AssertionError("不应加载模型")):
            with self.assertRaisesRegex(ValueError, "已变化"):
                run_generation(self.run_dir, self.model_dir, self.image_root, {"seed": 7}, self.prompts, resume=False)

    def test_changed_image_detected_before_resuming_cached_result(self):
        self.run_fake(FakeRuntime())
        Image.new("RGB", (32, 32), "black").save(self.image)
        with self.assertRaisesRegex(ValueError, "SHA256"):
            self.run_fake(FakeRuntime())

    def test_path_escape_is_recorded_as_error_for_both_tasks(self):
        self.record["relative_path"] = "../公司外部.png"
        self.write_manifest([self.record])
        runtime = FakeRuntime()
        stats = self.run_fake(runtime)
        self.assertEqual(stats["generated_error"], 2)
        self.assertEqual(runtime.calls, [])
        self.assertIn("相对路径", self.result("summary")["error"]["message"])

    def test_model_metadata_change_rejects_cached_results(self):
        self.run_fake(FakeRuntime())
        (self.model_dir / "tokenizer_config.json").write_text('{"new":true}', encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "已变化"):
            self.run_fake(FakeRuntime())

    def test_invalid_images_skipped_and_duplicate_ids_rejected(self):
        self.write_manifest([self.record, {"image_id": "bad", "status": "error"}])
        stats = self.run_fake(FakeRuntime())
        self.assertEqual(stats["skipped_invalid_images"], 1)
        self.write_manifest([self.record, self.record])
        with self.assertRaisesRegex(ValueError, "重复 image_id"):
            self.run_fake(FakeRuntime())

    def test_seed_is_stable_across_retry(self):
        first = FakeRuntime(fail_summary=True)
        self.run_fake(first)
        retry = FakeRuntime()
        self.run_fake(retry)
        self.assertEqual(first.calls[0][2], retry.calls[0][2])

    def test_invalid_config_is_rejected_before_model_load(self):
        for config in ({"image_max_pixels": 0}, {"enable_thinking": "false"}, {"max_new_tokens": {"summary": 0}}):
            with self.subTest(config=config), self.assertRaises(ValueError):
                self.run_fake(FakeRuntime(), config)


class FakeIds(list):
    def __getitem__(self, item):
        value = super().__getitem__(item)
        return FakeIds(value) if isinstance(item, slice) else value

    def detach(self):
        return self

    def cpu(self):
        return self

    def tolist(self):
        return list(self)


class FakeInputs(dict):
    def to(self, device):
        return self


class DecodeTests(unittest.TestCase):
    def test_loader_forces_local_files_and_disallows_remote_code(self):
        fake_torch = ModuleType("torch")
        fake_torch.__version__ = "测试版本"
        fake_torch.bfloat16 = "测试精度"
        fake_torch.version = SimpleNamespace(cuda="测试 CUDA")
        fake_torch.cuda = SimpleNamespace(
            is_available=lambda: True, device_count=lambda: 1,
            get_device_capability=lambda i: (8, 0), get_device_name=lambda i: "测试 GPU",
        )
        fake_transformers = ModuleType("transformers")
        fake_transformers.__version__ = "测试版本"
        fake_transformers.AutoProcessor = SimpleNamespace(from_pretrained=Mock())
        fake_transformers.Qwen3_5MoeForConditionalGeneration = SimpleNamespace(
            from_pretrained=Mock(return_value=SimpleNamespace(eval=Mock(), hf_device_map={"层": "cuda:0"}))
        )
        fake_transformers.set_seed = Mock()
        with patch.dict("sys.modules", {"torch": fake_torch, "transformers": fake_transformers}):
            _QwenRuntime(Path("/本地模型"), DEFAULT_CONFIG)
        processor_kwargs = fake_transformers.AutoProcessor.from_pretrained.call_args.kwargs
        model_kwargs = fake_transformers.Qwen3_5MoeForConditionalGeneration.from_pretrained.call_args.kwargs
        for kwargs in (processor_kwargs, model_kwargs):
            self.assertIs(kwargs["local_files_only"], True)
            self.assertIs(kwargs["trust_remote_code"], False)
        self.assertEqual(model_kwargs["dtype"], "测试精度")
        self.assertEqual(model_kwargs["device_map"], "auto")

    def test_only_terminal_protocol_token_is_removed_and_limit_with_eos_is_stop(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "image.png"
            Image.new("RGB", (32, 32), "white").save(path)
            runtime = _QwenRuntime.__new__(_QwenRuntime)
            runtime.Image = Image
            from PIL import ImageOps
            runtime.ImageOps = ImageOps
            runtime.config = {**DEFAULT_CONFIG, "max_new_tokens": {"summary": 4, "interpretation": 4}}
            runtime.set_seed = lambda seed: None
            runtime._synchronize = lambda: None
            runtime.torch = SimpleNamespace(inference_mode=contextlib.nullcontext)
            captured = {}

            def chat_template(messages, **kwargs):
                self.assertEqual(messages[0]["content"], [{"type": "text", "text": "系统"}])
                captured.update(kwargs)
                return FakeInputs(input_ids=SimpleNamespace(shape=(1, 2)))

            spellings = {11: "<think>", 12: "  内容\n", 13: "</think>\n\n", 99: "<im_end>"}
            runtime.processor = SimpleNamespace(
                apply_chat_template=chat_template,
                decode=lambda ids, **kwargs: "".join(spellings[token] for token in ids),
                tokenizer=SimpleNamespace(pad_token_id=0),
            )
            runtime.model = SimpleNamespace(
                device="cuda:0", generation_config=SimpleNamespace(eos_token_id=99),
                generate=lambda **kwargs: [FakeIds([1, 2, 11, 12, 13, 99])],
            )
            result = runtime.generate(path, {"system": "系统", "user": "图片"}, "summary", 1)
            self.assertEqual(result["text"], "<think>  内容\n</think>\n\n")
            self.assertEqual(result["raw_text"], "<think>  内容\n</think>\n\n<im_end>")
            self.assertEqual(result["finish_reason"], "stop")
            self.assertEqual(result["raw_output_token_ids"], [11, 12, 13, 99])
            self.assertIs(captured["enable_thinking"], False)
            self.assertEqual(captured["images_kwargs"]["size"]["longest_edge"], DEFAULT_CONFIG["image_max_pixels"])
            runtime.model.generate = lambda **kwargs: [FakeIds([1, 2, 11, 12, 13, 12])]
            result = runtime.generate(path, {"system": "系统", "user": "图片"}, "summary", 1)
            self.assertEqual(result["finish_reason"], "length")


if __name__ == "__main__":
    unittest.main()
