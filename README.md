# 图片摘要与解读测评

本项目面向 Android 识图摘要卡片和解读正文：读取本地 **Qwen3.5-35B-A3B** 模型，对每张图片分别执行原版 `v04_summary`、`v04_interpret` 提示词，保存原文并做细粒度测评。截图和实拍一起处理，保留语种、来源和场景分组。

当前已经实现生成、规则检查、独立事实参考请求、逐断言评审请求、回包校验和报告。**大模型接口尚未接入；没有执行任何外部 API 请求。** 没有评审回包时，语义质量指标显示为空，不能解释为零分。

## 在华为机器上开始

拉取代码使用 `git clone` 或已有仓库的 `git pull`。`git push` 是把本地内容上传到远端，本流程只需要获取代码。

```bash
git clone https://github.com/hitmhj/qwen-image-text-eval.git
cd qwen-image-text-eval
```

仓库为私有，需要具有读取权限的 GitHub 身份。仓库只包含代码、提示词、配置样例、文档及使用合成数据的测试。公司图片、模型、输出和评审记录不进入 Git；运行目录建议放在仓库外的华为内部存储。

建议使用 Linux、Python 3.10 及以上、NVIDIA CUDA 环境。先在允许的环境准备适配 CUDA 的 PyTorch/torchvision、Transformers 和依赖；详情见 [模型运行说明](docs/模型运行.md)。环境已准备好时：

```bash
# 基础安装只用于扫描、规则检查、导入和报告。
python -m pip install -e .

# 生成环境需增加 GPU 依赖；如已安装匹配的依赖，可用 --no-deps。
python -m pip install -e '.[generation]'
```

华为环境不能访问外部软件源时，使用提前准备并获准进入环境的依赖目录：

```bash
python -m pip install --no-index --find-links /内部依赖目录 -e '.[generation]'
```

脚本不会自动下载模型或安装依赖。运行阶段强制 Hugging Face 离线读取。**35B 是总权重规模，A3B 是激活参数规模**；BF16 权重约需 70GB，另需预留视觉编码、缓存和运行空间。默认 `device_map=auto` 可使用多张可见 GPU。具体显存和驱动兼容性必须在目标机器上确认。

## 输入结构

```text
图片根目录/
├── 英语/
│   ├── 截图/
│   │   ├── 图片一.png
│   │   └── 更深的子目录/图片二.jpg
│   └── 实拍/图片三.jpg
├── 日语/
│   ├── 截图/图片四.png
│   └── 实拍/图片五.jpg
└── 其他语种/...
```

语言使用第一级目录名；来源使用第二级目录名，支持 `截图/截屏/screenshot/screenshots` 和 `实拍/拍摄/photo/photos/camera`。不符合结构的图片也会处理，并标为 `unknown`，不会静默丢弃。支持 JPG、JPEG、PNG、WebP、BMP、TIFF，扩展名不区分大小写。损坏图片进入错误清单；相同内容的副本仍生成，但另给去重后的统计。

## 一键生成

打开 `scripts/generate.sh`，开头预留了模型、图片、输出目录以及 GPU 编号。也可用环境变量填写，路径支持空格：

```bash
MODEL_DIR=/内部模型/Qwen3.5-35B-A3B \
IMAGE_ROOT=/内部数据/图片根目录 \
RUN_DIR=/内部结果/基线一 \
CUDA_VISIBLE_DEVICES=0,1 \
bash scripts/generate.sh
```

每张图运行两次，两个任务没有共享对话，不把摘要喂给解读。默认关闭思考、关闭采样；实际参数、软件版本、模型文件元数据及提示词均保存。生成文本不会被裁字、改写或去掉思考标签来制造合规结果；末尾协议令牌的解码边界另行保留在原始字段中。

重跑相同命令会继续未完成和失败任务，成功任务跳过；图片内容、提示词、模型元数据或设置变化时拒绝混用旧结果，请更换 `RUN_DIR`。运行时持续打印进度；任一任务出错或清单中存在损坏图片，最终退出码为 1；任务错误会留下结果并继续其他图片。启动配置错误为 2。对显存不足应先调整硬件或配置，再开新运行目录。

## 本地规则与当前阶段报告

只需要复制运行目录到另一个**已获准处理这些数据的内部平台**，无需携带模型权重。仓库本身不传输数据；跨平台的数据流向由华为既有流程确定。

```bash
RUN_DIR=/内部结果/基线一 bash scripts/evaluate.sh
```

此命令执行规则检查、准备事实参考请求及报告，不调用大模型。也可以单独执行：

```bash
python -m qwen_text_eval rules --run-dir /内部结果/基线一
python -m qwen_text_eval report --run-dir /内部结果/基线一
```

规则明确区分通过、失败、待复核、不适用。字数按去除全部空白后的 Unicode 码点统计，标点、数字、字母各计 1；摘要允许单个末尾换行，额外空行失败。它不把专有名词中的英文直接判错，也不猜图片事实。

## 独立事实参考与大模型评审

流程是**原图 → 独立事实标注 → 人工抽查／纠正 → 原图、事实参考和候选文本 → 逐断言评审 → 程序汇总**。参考标注不能看到 Qwen 输出，也不能把另一模型写出的整段“标准摘要”当作唯一答案。允许多种准确表达，以可定位的原子事实核验。

```bash
# 可在生成前执行；请求中完全没有候选文本。
python -m qwen_text_eval prepare-reference --run-dir /内部结果/基线一

# 等接口适配器或人工标注产生回包后导入，一行一个 JSON 对象。
python -m qwen_text_eval import-reference \
  --run-dir /内部结果/基线一 --input /内部回包/事实参考.jsonl

# 已逐条人工核验的子集可单独导入；--human-verified 不能用来替代核验。
python -m qwen_text_eval import-reference \
  --run-dir /内部结果/基线一 --input /内部回包/人工核验子集.jsonl \
  --human-verified --replace

python -m qwen_text_eval prepare-judge --run-dir /内部结果/基线一
python -m qwen_text_eval import-judge \
  --run-dir /内部结果/基线一 --input /内部回包/文本评审.jsonl

# 有可靠场景后重跑规则，确定菜单字数豁免、医学免责声明等条件。
python -m qwen_text_eval rules --run-dir /内部结果/基线一
python -m qwen_text_eval report --run-dir /内部结果/基线一
```

请求是厂商无关的本地 JSONL：`system` 为评审提示词，`input` 为待评内容，`attachments` 只包含原图相对路径与校验值；不嵌入图片，不发送请求。明天接入接口时，适配器在内部图片根目录解析路径、验证哈希，按接口要求装配图片，并注入真实的 `annotator.model/version` 或 `judge.model/version` 元数据。文本接口单凭 Qwen 输出无法判断视觉幻觉；应支持看图，或者使用已人工核验的事实参考并明确视觉盲区。

回包结构可查看：

```bash
python -m qwen_text_eval schema reference
python -m qwen_text_eval schema judge
```

回包需包含请求中的图片标识和校验值；评审还必须带候选文本、事实参考和评审提示词的 SHA256，并引用存在的事实标识。每条断言必须给类别，避免漏算价格、币种等数值错误。同一运行固定评分量表、评审提示词及评审模型／版本，改变评分口径请另建目录。参考修改或文本变化后，旧评审会显示过期，不混入有效均分。普通导入不能把模型标注自封为 `human_verified`。整份导入先校验后落盘，校验失败不导入其中一部分。

## 输出目录

```text
基线一/
├── run.json                       # 清单统计、运行标识、数据根目录
├── manifest.jsonl                 # 相对路径、语种、来源、哈希、尺寸、重复和错误
├── generation_meta.json           # 模型、配置、运行环境、指纹
├── prompts/{summary,interpretation}.json
├── evaluation_prompts/{reference,judge}_system.txt
├── judge_meta.json                # 实际评审模型、版本和提示词校验值
├── samples/<图片标识>/
│   ├── summary.json               # 生成原文、原始解码、令牌、耗时、终止原因
│   └── interpretation.json
├── requests/{reference,judge}.jsonl
├── references/<图片标识>.json      # 独立事实、证据、置信度、核验状态
├── judgements/<图片标识>/{summary,interpretation}.json
├── rubric.json                    # 评分配置快照
├── rules/<图片标识>/{summary,interpretation}.json
└── reports/
    ├── aggregate.json             # 汇总、语种／来源／场景分组、去重统计、适用分母
    ├── per_sample.jsonl
    ├── per_sample.csv             # 每张图每个任务及原文
    ├── review.csv                 # 错误、截断、过期、歧义、关键事实等复核列表
    └── report.md
```

CSV 使用 UTF-8 BOM，方便中文表格软件打开；可能被当作公式的文本加前导单引号。`per_sample.jsonl` 和原始生成文件保留未改写文本。

## 重点看哪些指标

| 类别 | 指标与用途 |
|---|---|
| 提示词遵循 | 两行、10/30字、50字速览、模块和条目数、500字限制、禁用前缀与 emoji、简体中文和医学声明 |
| 语义质量 | 事实准确、重点覆盖、翻译、数字价格、无依据内容、可读性、简洁、提示词遵循 |
| 摘要专项 | 主体识别、标题与描述互补 |
| 解读专项 | 场景理解、结构组织、实用价值 |
| 事实级 | 支持／矛盾／无依据／无法核验／背景；关键事实加权覆盖、数字断言正确率 |
| 运行质量 | 生成成功率、错误、缺失、截断、耗时、评审覆盖率 |
| 结果可信度 | 人工核验比例、无法核验占比、严重错误、适用维度权重、各指标样本数 |

维度打分为 0～4，程序按固定权重换算为 0～100；不适用项为空并重新归一化，同时保留适用权重。重要事实错误和安全严重错误分别独立统计，不能被可读性得分抵消；各断言标注价格、币种、过敏原等类别。事实精确率同时给可判定分母和含无法核验项的保守口径；没有事实或数值时，相关指标为空。

详细评分锚点、参考标注、人审抽样、提示词冲突及偏差控制见 [测评设计](docs/测评设计.md)。

## 接口接入和验证范围

`configs/judge_api.example.json` 只预留配置，默认关闭，密钥应来自环境变量。明天确认实际协议、鉴权、图片输入方式、可使用的内部模型和返回格式后再实现调用适配器。没有获准的数据外发路径时，不能把图片或模型输出发送到公共 GPT 接口；评审应在允许的数据边界内完成。

```bash
python -m unittest discover -s tests -v
```

测试使用代码生成的合成图片、虚拟模型和虚拟评分回包，覆盖路径、双任务断点、指纹变化、格式边界、错误导入、缺失分母、过期评审和报告。**已验证离线代码流程；尚未在华为 NVIDIA GPU 上运行实际 35B 模型，未生成真实图片测评结论。**
