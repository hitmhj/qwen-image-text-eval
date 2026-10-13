# 三任务图片文本测评

第一平台只调用本地 Qwen；第二平台负责 GPT 构造独立事实参考、GPT 评分和 HTML。两个平台通过**一个 handoff.json 文件**交接，包含原图完整字节、图片哈希、三任务答案、推荐要求和实际提示词。第二平台无需第一平台的文件路径、图片目录、断点目录或项目依赖。

## 已集成平台接口的完整评估文件

[评估.py](评估.py) 是独立完整脚本，已集成您提供的 `hm_aigc.aigc_managed` 调用。第二平台只复制该文件和 `handoff.json`。在文件顶部填写 `DEFAULT_ACCOUNT`、`DEFAULT_MODEL`、`DEFAULT_MODEL_VERSION`；`DEFAULT_MODEL_AGENT` 沿用 `yibu`，SDK模块名按实际填写。如果平台需要先创建客户端，在顶部放入您已有的导入和初始化代码，得到 `hm_aigc` 对象。

配置完成后，参考和评分可独立运行，结果保存在同一文件：

```bash
python 评估.py handoff.json --stage reference
python 评估.py handoff.json --stage score
```

也可以运行 `python 评估.py handoff.json` 顺序完成全部阶段。输出只有平铺的 `results.json` 和 `report.html`；两套数据在文件内独立统计。`--prepare-only` 只生成请求与预览。原图直接读取交接包，系统提示词保留，接口失败及非法JSON会重试；成功参考和评分立即保存，重跑复用已完成记录。完整文件已通过虚拟SDK及独立目录测试，真实子账号与实际SDK服务需在目标平台验证。

以下通用 `evaluate.py` 保留原有请求准备与真实回包导入流程；使用平台接口时直接采用上面的完整 `评估.py`。

## 最短使用流程

1. 第一平台运行 scripts/generate.sh，自动得到 OUTPUT_ROOT/handoff.json。
2. 按您的跨平台复制流程，把 **handoff.json 和 evaluate.py 两个文件**放到第二平台同一目录。
3. 在第二平台运行：

```bash
python evaluate.py handoff.json
```

输出为 report.html 和 results.json，默认写到 handoff.json 所在目录。原图已内嵌，无需另复制图片文件夹。GPT 接口明天提供后集中接入 evaluate.py 的 call_gpt()；目前运行会准备参考请求并生成预览报告，语义评分显示“未测”。

## 获取代码

私有仓库使用有读取权限的 GitHub 账号：

```bash
git clone https://github.com/hitmhj/qwen-image-text-eval.git
cd qwen-image-text-eval
```

已有旧版执行 git pull --ff-only。

## 第一平台：配置模型并生成

原始图片按语言分组，截图和实拍一起处理，来源字段保留：

```text
原始图片根目录/
├── 英语/
│   ├── 截图/图片一.png
│   └── 实拍/图片二.jpg
└── 日语/……
```

第一平台需要 Python 3.10 以上、NVIDIA CUDA，以及完整的本地 Qwen3.5-35B-A3B 多模态模型。模型位置已在 .sh 顶部预留。安装生成依赖：

```bash
python -m pip install -e '.[generation]'
```

华为 A800、驱动 535、现有 Python 3.9 环境请采用[指定目录的最小安装步骤](docs/模型运行.md#华为a800驱动535的最小安装)：单独准备 Python 3.10，安装 CUDA 11.8 版 torch 2.6.0 / torchvision 0.21.0，再执行 `python -m pip install -r requirements-generation.txt`。生成入口直接运行仓库源码，无需安装本项目，生成平台也无需 jsonschema 或 OpenCC。要求池、三任务提示词和交接格式保持一致。

填写目录后运行：

```bash
MODEL_DIR=/模型目录/Qwen3.5-35B-A3B \
PRIMARY_IMAGE_ROOT=/原始图片目录 \
OUTPUT_ROOT=/输出目录/三任务基线 \
CUDA_VISIBLE_DEVICES=0,1 \
bash scripts/generate.sh
```

默认同时运行原始图片集 primary 和仓库内的 20 张清晰、无手菜单集 menu_reference。设置 DATASET=primary 或 DATASET=menu_reference 可只运行一套；辅助菜单路径可用 MENU_IMAGE_ROOT 覆盖。

每张图分别生成摘要、解读、菜品推荐，三个任务使用独立对话。推荐从 16 条要求池中固定抽取一条，由图片内容哈希、要求池指纹和种子确定，复测不会重新挑要求。预算、忌口、人数、偏好和实际提示词一起写入交接文件。

外币菜单需要预算换算时，用 FX_CONTEXT=/汇率配置.json 显式输入汇率；未提供时不猜汇率。模型只从本地加载，生成脚本不调用评审 API。模型依赖、配置和资源说明见 [模型运行说明](docs/模型运行.md)。

运行后只需复制 OUTPUT_ROOT/handoff.json。primary/、menu_reference/ 等工作目录用于断点和追溯，可留在第一平台。原图按原文件字节内嵌，未再次压缩、缩放或裁剪。

## 第二平台：一个 Python 文件、一条命令

第二平台只准备以下两个文件，Python 3.10 以上标准库可运行：

```text
第二平台任意目录/
├── evaluate.py
└── handoff.json
```

```bash
python evaluate.py handoff.json
```

输出默认写到 handoff.json 所在目录：

```text
第二平台任意目录/
├── evaluate.py
├── handoff.json
├── results.json             # 参考、评分、指标和运行状态统一保存
├── report.html              # 原图、三任务原文、证据、理由和详细结果
└── reference_requests.json  # 接口未接入时准备的参考请求
```

原图从 handoff.json 读取并核验 SHA256。第二平台不安装 Qwen、torch、Transformers 或本项目包，不复制提示词、配置、图片目录，也不填写第一平台路径。评审提示词、量表、协议、规则和报告生成器都在 evaluate.py 中。

参考构造与评分是两次独立调用：参考只接收原图与固定要求，不接收 Qwen 答案；评分接收原图、独立参考和该任务答案。接口接入后由同一个入口按这个顺序执行，最后自动生成 HTML。

如需导入已经取得的真实回包，在同一入口增加 --references 参考回包.json、--scores 评分回包.json，支持 JSON 数组或 JSONL。--output-dir 可改输出位置。未来调用接口使用 --call-api --gpt-model 实际模型 --gpt-version 实际版本，接口尚未提供时不会完成真实评分。

需要分开执行参考和评分时，可选 --stage reference 或 --stage score；参考保存在同一个 results.json，后续评分自动读取。通常使用默认的一条命令即可。

## 两套数据独立测评

| 标识 | 图片 | 报告方式 |
| --- | --- | --- |
| primary | 原来的语种目录、截图及实拍 | 自己的三任务评分、分组和分母 |
| menu_reference | 本次整理的 20 张清晰、无手菜单原图 | 自己的三任务评分、分组和分母 |

两套数据可以装在同一个 handoff.json 中，但始终保留独立 dataset_id。results.json 分数据集保存参考、评分和指标，HTML 分开呈现两套数据，不合并均分、排名或成功率。辅助菜单来源、作者、许可和清晰度说明见 datasets/menu_reference/图片清单.json。部分原菜单没有价格，未知项不会被当作已满足条件。

## 三个任务的提示词与指标

[三个任务提示词完整原文](docs/三个任务提示词原文.md) 包含三组完整的系统提示词和用户提示词，与实际 JSON 逐字一致：

- 摘要：prompts/v04_summary.json，保留您提供的原文。
- 解读：prompts/v04_interpret.json，保留您提供的原文。
- 菜品推荐：prompts/v01_recommendation.json，约束预算、忌口、证据、未知信息和输出。

| 任务 | 评分维度与权重 |
| --- | --- |
| 摘要 | 事实 35%、重点 25%、翻译 10%、数字 10%、表达 10%、提示词遵循 10% |
| 解读 | 事实 35%、重点 25%、翻译 10%、数字 10%、结构 10%、实用性 5%、提示词遵循 5% |
| 推荐 | 事实 25%、需求符合 30%、价格预算 20%、合理性 15%、表达 10% |

维度按 0～4 分计算，程序换算为 0～100 分。同时报告事实准确率、无依据率、矛盾率、未知率、重点覆盖率、数字正确率、硬约束违规及可确认推荐成功率，并显示各自分母。预算使用十进制复算，未知辣度、食材、费用、过敏原或汇率不算满足；缺少参考或评分的语义指标为空。详细口径见 [测评设计](docs/测评设计.md)。

## 代码与第一平台工作目录

```text
qwen-image-text-eval/
├── generate.py                    # 第一平台：两套数据三任务生成及交接导出
├── evaluate.py                    # 第二平台：单文件参考、评分和HTML
├── scripts/generate.sh            # 预留模型、图片、输出和GPU配置
├── prompts/                       # 三份生成提示词、两份评审提示词
├── configs/                       # 生成设置、要求池、指标权重
├── datasets/menu_reference/       # 20张菜单及来源清单
├── src/qwen_text_eval/             # 扫描、抽样、模型生成、断点与导出
├── docs/
└── tests/

第一平台OUTPUT_ROOT/
├── handoff.json                   # 唯一需要跨平台复制的生成产物
├── primary/                      # 第一平台工作目录，保留断点
│   ├── generation_results.jsonl
│   ├── run.json、manifest.jsonl、generation_meta.json
│   ├── prompts/
│   └── samples/图片ID/三任务.json
└── menu_reference/               # 辅助集独立工作目录
```

旧版 reference、score、report 子命令保留用于兼容已有 JSONL 流程；新运行统一使用 handoff.json 单文件入口。

测试命令：python -m unittest discover -s tests -v。测试使用虚拟模型及合成回包，实际 35B GPU 推理和 GPT 接口尚未运行。
