# 三任务图片文本测评

本项目将本地 Qwen 生成、GPT 事实参考构造、GPT 评分分成三个独立阶段。第一阶段在 NVIDIA 平台运行；第二、第三阶段在另一平台仅需要一个独立的 `evaluate.py`。GPT 接口未提供，当前只准备请求、导入回包和生成报告，默认不联网。

## 两套数据分别测评

| 数据集标识 | 图片来源 | 用途 |
|---|---|---|
| `primary` | 您原来的图片大目录 | 正式图片集，包含各语种截图与实拍 |
| `menu_reference` | 本次整理的20张清晰、无手菜单原图 | 独立辅助集，随代码提供 |

两套数据各自生成、构造参考、评分、汇总和输出HTML，不能合并成一个总体分数。脚本拒绝两个输入根目录相同或互相包含；评审器拒绝将不同数据集或运行的结果混在同一输入中。

辅助集保存原始像素和文件内容，没有再次压缩、缩放、裁剪或去手。它们是之前公开来源测试素材中的**原始实拍菜单**，来源、作者、许可、文件摘要及质量说明记录于 `datasets/menu_reference/图片清单.json` 和CSV。部分原菜单没有价格，已注明，测评不能要求恢复未标注信息。它与本目录旁的 `gen_image` 中20张图片内容完全一致。

公司图片、模型、运行结果和密钥不会进入 Git。仓库公开来源菜单是唯一允许随代码提交的图片目录。

## 三份生成提示词原文

[三个任务提示词原文](docs/三个任务提示词原文.md) 包含摘要、解读、菜品推荐的完整系统提示词和用户提示词，六个正文块与实际JSON逐字一致。

- `prompts/v04_summary.json`：摘要，保留您提供的原版。
- `prompts/v04_interpret.json`：解读，保留您提供的原版。
- `prompts/v01_recommendation.json`：菜品推荐，采用讨论确认的提示词。

摘要示例字数、解读示例条目数、菜单原文括注等规范冲突在测评设计中明确说明，生成端不擅自修改或裁字。

`configs/requirements_pool.json` 保存16条自然语言推荐要求及预算、人数、硬约束、偏好的结构。每张图片按图片SHA256、要求池指纹和固定种子抽取一条，不因难度、无解或未知重抽。两个模型复测同一图片、池和种子时使用同一要求；要求正文及结构写入生成结果。三个任务采用独立对话，摘要或解读的答案不会喂给推荐。

## 获取代码与依赖

私有仓库需要有读取权限的GitHub账号：

```bash
git clone https://github.com/hitmhj/qwen-image-text-eval.git
cd qwen-image-text-eval
```

已拉取过旧版时，在仓库中运行 `git pull --ff-only`。

平台一建议 Linux、Python 3.10以上、NVIDIA CUDA。需要包含完整多模态权重、分词器、图像处理配置和聊天模板的本地 `Qwen3.5-35B-A3B` 目录。PyTorch、torchvision与目标驱动须匹配，环境准备见 [模型运行说明](docs/模型运行.md)。在允许的源或准备环境中：

```bash
python -m pip install -e '.[generation]'
```

离线安装应准备匹配目标OS、Python和CUDA的wheel文件及构建依赖：

```bash
python -m pip install --no-index --find-links /内部依赖目录 setuptools
python -m pip install --no-index --find-links /内部依赖目录 --no-build-isolation -e '.[generation]'
```

运行脚本不会下载模型和依赖。35B为总权重规模，A3B为激活参数规模；BF16约70GB权重之外还需视觉编码、缓存和运行空间。默认允许一个进程使用多张可见GPU。

平台二仅需Python 3.10以上与 `evaluate.py`。标准库即可运行；可选安装OpenCC增强简体检查，缺少时相关检查标为待复核。无需安装torch、Transformers、本项目包，也不需要携带任何提示词或配置文件：评审提示词、量表、协议、规则和HTML渲染器都嵌入在单文件中。

## 第一阶段：输入并调用本地Qwen

原始图片输入结构：

```text
原始图片根目录/
├── 英语/
│   ├── 截图/图片一.png
│   └── 实拍/图片二.jpg
├── 日语/
│   ├── 截图/图片三.png
│   └── 实拍/图片四.jpg
└── 其他语种/...
```

递归处理全部支持图片，截图和实拍进入同一批任务，保留分组字段。文件夹名可以不同；无法识别来源的图片标为unknown并处理。JPG/JPEG/PNG/WebP/BMP/TIFF均可；损坏图片、重复内容和失败结果明确记录。

编辑 `scripts/generate.sh` 开头，预留字段为MODEL_DIR、PRIMARY_IMAGE_ROOT、OUTPUT_ROOT和GPU编号。辅助菜单路径默认是仓库内 `datasets/menu_reference`，也可用MENU_IMAGE_ROOT指向您当前的 `gen_image`：

```bash
MODEL_DIR=/内部模型/Qwen3.5-35B-A3B \
PRIMARY_IMAGE_ROOT=/内部数据/原始图片 \
OUTPUT_ROOT=/内部结果/三任务基线一 \
CUDA_VISIBLE_DEVICES=0,1 \
bash scripts/generate.sh
```

仅运行辅助集可设置 `DATASET=menu_reference`，不需要原始图片目录；只运行正式集可设置 `DATASET=primary`。

也可直接运行Python入口：

```bash
python generate.py \
  --model-dir /内部模型/Qwen3.5-35B-A3B \
  --primary-images /内部数据/原始图片 \
  --output-root /内部结果/三任务基线一
```

模型输出原文、原始解码、耗时、令牌数、终止原因及错误均保存。相同命令断点继续；图片内容、模型、依赖、提示词、要求池或汇率设置变化时新建输出目录。每套数据最终导出一个 `generation_results.jsonl`，每行一张图，内含三个答案、推荐要求、图片身份和提示词快照。

外币菜单配人民币预算时，可以使用 `--fx-context /内部配置/汇率.json` 或脚本环境变量FX_CONTEXT。格式是 `{ "as_of": "日期", "source": "内部提供来源", "rates": { "EUR": { "CNY": "明确的正数汇率" } } }`，含义为1单位来源币种等于给定数量目标币种。没有输入就不换算、不联网查汇率、不保证符合人民币预算；相关条件进入未知状态。

## 第二阶段：独立GPT事实参考

将该数据集的生成结果文件提供给已获准处理数据的第二平台，同批原图须可访问。此项目没有跨平台数据传输功能，使用既有内部流程。

```bash
python evaluate.py reference \
  --input /内部结果/三任务基线一/primary/generation_results.jsonl \
  --image-root /第二平台图片/原始图片 \
  --output-dir /第二平台结果/primary
```

当前生成平铺的 `reference_requests.jsonl`，并校验原图哈希；请求只含原图及推荐要求，不含Qwen答案。实际GPT视觉接口由您提供后接入 `call_gpt()`。接口应注入真实模型与版本、真正装配图像像素；默认不发送请求。

取得接口回包后：

```bash
python evaluate.py reference \
  --input /内部结果/三任务基线一/primary/generation_results.jsonl \
  --image-root /第二平台图片/原始图片 \
  --output-dir /第二平台结果/primary \
  --responses /内部回包/primary_reference.jsonl
```

程序校验后生成 `ground_truth.jsonl`。它是带位置证据、置信度、任务重点的原子事实参考，推荐另外记录可行性与硬约束相关信息，允许多个正确推荐方案。模型默认draft，人工核验子集后才使用 `--human-verified`；`--replace`用于明确替换已有回包，旧评分随后失效。

辅助集另执行一遍：input指向menu_reference的生成结果，image-root指向同一套20张菜单的图片根目录，output-dir使用独立的 `/第二平台结果/menu_reference`。

## 第三阶段：GPT评分与HTML

```bash
python evaluate.py score \
  --input /内部结果/三任务基线一/primary/generation_results.jsonl \
  --image-root /第二平台图片/原始图片 \
  --ground-truth /第二平台结果/primary/ground_truth.jsonl \
  --output-dir /第二平台结果/primary
```

当前准备平铺 `score_requests.jsonl`。取得评分回包后，同命令增加 `--responses /内部回包/primary_scores.jsonl`，导入后立即生成 `scores.jsonl`、`metrics.json` 和 `report.html`。

随时可独立重新生成报告：

```bash
python evaluate.py report \
  --input /内部结果/三任务基线一/primary/generation_results.jsonl \
  --image-root /第二平台图片/原始图片 \
  --output-dir /第二平台结果/primary
```

缺少回包时只报告已完成环节，语义指标显示未测；不能解释为零分或完成评分。对menu_reference分别执行相同三条命令，保持另一输出目录。

HTML展示当前数据集、三任务及语种/来源分组，逐图展开原图、Qwen原文、抽中要求、事实证据、声明判定、维度理由、格式检查、预算复算、错误和未知条件。所有资源在本地，不依赖CDN。传入image-root时可嵌入原图供离线浏览；单图超过4MiB或总原图字节超过64MiB时不嵌入，保留原图路径，避免大批次报告失控。

## 输出结构

```text
平台一输出根目录/
├── primary/
│   ├── generation_results.jsonl     # 移交另一平台的单个结果文件
│   ├── run.json、manifest.jsonl、generation_meta.json
│   ├── prompts/                    # 三份实际提示词快照
│   └── samples/<图片ID>/            # 三任务原文及断点记录
└── menu_reference/                 # 同结构，20图独立生成

平台二结果根目录/
├── primary/
│   ├── eval_meta.json              # 指纹与评分配置快照
│   ├── reference_requests.jsonl    # 接口待接入期间的请求文件
│   ├── ground_truth.jsonl
│   ├── score_requests.jsonl
│   ├── scores.jsonl
│   ├── metrics.json
│   └── report.html
└── menu_reference/                 # 同结构，独立评分和HTML
```

第二平台没有每张图一个子目录。请求文件是接口尚未接入时的中间产物，接口实现后可直接运行两个阶段；事实参考、评分和报告保持独立。

## 代码结构

```text
qwen-image-text-eval/
├── generate.py                    # 平台一：两个数据集的生成入口
├── evaluate.py                    # 平台二：参考、评分、HTML，全在一个文件
├── scripts/generate.sh            # 预留本地模型/图片/输出/GPU路径
├── prompts/                       # 三份生成prompt及两份评审prompt原文
├── configs/                       # 生成设置、16条要求池、权重、接口样例
├── datasets/menu_reference/       # 20张公开来源菜单，按语种/实拍分类
├── src/qwen_text_eval/             # 本地生成、扫描、稳定抽样、导出及兼容旧接口
├── docs/                          # 三prompt全文、指标设计、模型运行说明
└── tests/                         # 合成图、虚拟模型、回包和隔离性验证
```

生成库中storage负责原子落盘，dataset负责语种/来源扫描，requirements负责稳定抽样，generation负责本地模型三任务，exporting负责单文件交接。包内历史评审模块保留用于旧版测试；旧CLI评审入口已停用，本次三任务评审统一使用根目录evaluate.py。

## 指标与验证范围

摘要/解读采用事实依据、重点覆盖、翻译、数字、表达/结构与提示词遵循；推荐采用菜品事实25%、需求符合30%、价格预算20%、合理性15%、表达10%。维度0~4分，程序按适用权重换算0~100；事实准确率、无依据率、矛盾率、未知比例、重点覆盖、数字正确率、硬约束违规与可确认推荐成功等另行计算并显示分母。

价格、份量、币种和套餐条件要绑定正确对象；预算计入图片明确的必付费用，未知不算满足。参考中没有收录不直接判幻觉；无原图的文本评审保留视觉盲区。严重错误独立展示。详细公式、规范冲突和人工抽检见 [测评设计](docs/测评设计.md)。

```bash
python -m unittest discover -s tests -v
```

测试使用合成数据、虚拟运行器和虚拟GPT回包，实际35B GPU推理和GPT接口尚未运行，未生成真实测评结论。20张菜单只完成素材筛选、内容哈希核验和输入扫描。
