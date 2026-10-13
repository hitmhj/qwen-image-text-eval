"""生成入口无需项目构建或评审依赖，旧解释器先给出明确错误。"""

import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


class MinimalGenerationTests(unittest.TestCase):
    def child(self, code, *arguments):
        environment = {key: value for key, value in os.environ.items() if key.upper() != "PYTHONPATH"}
        environment["PYTHONUTF8"] = "1"
        return subprocess.run([sys.executable, "-c", code, *map(str, arguments)],
                              capture_output=True, text=True, encoding="utf-8",
                              env=environment, timeout=30)

    def test_python39_is_rejected_before_project_dependency_imports(self):
        code = "import sys, runpy; sys.version_info=(3,9,1); runpy.run_path(sys.argv[1], run_name='__main__')"
        result = self.child(code, ROOT / "generate.py")
        self.assertEqual(result.returncode, 2)
        self.assertIn("Python 3.10", result.stderr)
        self.assertNotIn("Traceback", result.stderr)

    def test_direct_generation_and_handoff_work_without_review_dependencies(self):
        code = '''
import importlib.abc
import json
import sys
from pathlib import Path
from unittest.mock import patch

class MissingReviewPackages(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in {'jsonschema', 'opencc'}:
            raise ModuleNotFoundError('评审依赖在生成环境中未安装')

sys.meta_path.insert(0, MissingReviewPackages())
repository, base = map(Path, sys.argv[1:3])
sys.path.insert(0, str(repository))
import generate
from PIL import Image
assert 'qwen_text_eval.validation' not in sys.modules
assert 'qwen_text_eval.evaluation' not in sys.modules
model = base / '模型'
model.mkdir()
(model / 'config.json').write_text('{"model_type":"qwen3_5_moe"}', encoding='utf-8')
images = base / '原图' / '英语' / '实拍'
images.mkdir(parents=True)
Image.new('RGB', (35, 27), 'white').save(images / '菜单.png')

class FakeRuntime:
    info = {'model_class': '最小环境测试'}
    def generate(self, image_path, prompt, task, seed):
        return {'text':'菜单测试正文', 'raw_text':'菜单测试正文', 'latency_seconds':0.01,
                'input_tokens':12, 'output_tokens':8, 'finish_reason':'stop'}
    def cleanup_after_error(self):
        pass

output = base / '结果'
with patch('qwen_text_eval.generation._load_runtime', return_value=FakeRuntime()):
    assert generate.main(['--model-dir', str(model), '--primary-images', str(base / '原图'),
                          '--output-root', str(output), '--dataset', 'primary']) == 0
handoff = json.loads((output / 'handoff.json').read_text(encoding='utf-8'))
row = handoff['datasets'][0]['generation_results'][0]
assert row['image']['mime_type'] == 'image/png'
assert all(row[task]['status'] == 'ok' for task in ['summary', 'interpretation', 'recommendation'])
assert 'qwen_text_eval.validation' not in sys.modules
assert 'qwen_text_eval.evaluation' not in sys.modules
print('最小生成环境和单文件交接通过')
'''
        with tempfile.TemporaryDirectory() as temporary:
            result = self.child(code, ROOT, temporary)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertIn("最小生成环境和单文件交接通过", result.stdout)


if __name__ == "__main__":
    unittest.main()
