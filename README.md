# 三任务图片文本测评

对每张图片生成摘要、解读和菜品推荐，再使用 GPT 构造独立事实参考并评估。当前生成模型为本地 Hugging Face 格式的 **Qwen3.5-35B-A3B**，运行在 NVIDIA GPU 上。

三个阶段解耦：

| 阶段 | 平台 | 工作与产物 |
| --- | --- | --- |
| 1. Qwen 生成 | 第一平台（华为机器） | 读取本地图片、生成三个任务，导出一个 `handoff.json` |
| 2. GPT 构造事实参考 | 第二平台 | 只看原图和推荐要求，生成独立参考，不看 Qwen 答案 |
| 3. GPT 评估及报告 | 第二平台 | 根据原图、独立参考和 Qwen 答案核验，程序计算指标并生成 HTML |

第二平台只需要完整 [评估.py](评估.py) 和 `handoff.json`。HTML 也在第二平台生成，不需要把评分结果送回第一平台。跨平台交接文件包含原图，应按所在平台允许的数据流转方式使用。

## 获取代码

使用有私有仓库读取权限的 GitHub 账号：

```bash
git clone https://github.com/hitmhj/qwen-image-text-eval.git
cd qwen-image-text-eval
```

已有仓库更新：

```bash
git pull --ff-only
```

以下命令均在仓库根目录运行。第一阶段直接使用原有的 `scripts/generate.sh`。

## 第一平台：激活已有环境

你提供的环境目录为 `dataset/lzm_dataset/hjma/envs/qwen-local`。以下按 `dataset` 位于 `/home/ma-user/work` 写成绝对路径；如果实际位置不同，请相应替换，不要将该相对路径拼接到仓库目录下。

```bash
source /home/ma-user/anaconda3/etc/profile.d/conda.sh
conda activate /home/ma-user/work/dataset/lzm_dataset/hjma/envs/qwen-local

# 确认使用环境内的解释器；当前加载器要求 Python 3.10 或以上。
"$CONDA_PREFIX/bin/python" -c 'import sys; print(sys.executable); print(sys.version)'
```

`conda activate` 使用环境的绝对目录。提示符显示 `(qwen-local)` 还不够：该目录必须有自己的 `bin/python`。后续通过 `PYTHON_BIN` 明确指定它，避免运行到 base 环境。

已有环境能够运行生成时直接使用，不需要重新安装。本项目通过源码启动，无需 `pip install -e`。首次准备环境或缺少依赖时，在已激活且满足 Python 版本的环境中安装：

```bash
# 华为 A800、驱动 535 对应的既定安装方案；已安装就跳过。
"$CONDA_PREFIX/bin/python" -m pip install 'torch==2.6.0+cu118' 'torchvision==0.21.0+cu118' --index-url https://download.pytorch.org/whl/cu118
"$CONDA_PREFIX/bin/python" -m pip install -r requirements-generation.txt
```

安装细节见 [模型运行说明](docs/模型运行.md)。依赖安装只在允许的软件源或离线准备方式下进行；生成脚本本身不会下载依赖或模型。

## 第一平台：图片输入和运行命令

原始图片根目录按语种组织，其下的截图和实拍一起处理；来源信息仍保留：

```text
原始图片集/
├── 英语/
│   ├── 截图/图片一.png
│   └── 实拍/图片二.jpg
└── 日语/
    ├── 截图/……
    └── 实拍/……
```

将下面的模型、图片和输出路径替换为实际绝对路径，然后执行：

```bash
MODEL_DIR="/实际模型目录/Qwen3.5-35B-A3B" \
PRIMARY_IMAGE_ROOT="/实际路径/原始图片集" \
OUTPUT_ROOT="/实际路径/生成结果/run_01" \
CUDA_VISIBLE_DEVICES="0" \
PYTHON_BIN="$CONDA_PREFIX/bin/python" \
DATASET="both" \
bash scripts/generate.sh
```

| 配置 | 说明 |
| --- | --- |
| `MODEL_DIR` | 完整本地模型目录，包含 `config.json`、处理器、分词器、聊天模板和全部权重 |
| `PRIMARY_IMAGE_ROOT` | 原始图片集根目录，不是某一语种目录，也不是两套数据的共同父目录 |
| `OUTPUT_ROOT` | 生成结果目录，程序自动创建 |
| `CUDA_VISIBLE_DEVICES` | 单卡填写 `0`，多卡如 `0,1`；仍为单进程生成 |
| `PYTHON_BIN` | 指定已激活环境的 `bin/python` |
| `DATASET` | `both` 两套都运行；`primary` 只运行原始集；`menu_reference` 只运行辅助菜单集 |
| `MENU_IMAGE_ROOT` | 可选，默认 `datasets/menu_reference`；可替换辅助菜单目录 |
| `GENERATION_CONFIG` | 可选，默认 `configs/generation.json`，控制输出长度、采样、图像像素预算和显存配置 |
| `FX_CONTEXT` | 可选，显式汇率 JSON 文件路径；未提供时不猜测汇率 |

原始图片目录可以和 `menu_reference` 同级，两套根目录不能相同或互相包含。默认辅助集是仓库内的 20 张清晰、无手菜单原图；两套数据分别生成和测评。

每张有效图片分别运行摘要、解读和菜品推荐，三个任务使用独立对话。推荐要求从 [要求池](configs/requirements_pool.json) 中抽取，例如“我一个人，总预算50元人民币，不能吃辣，请推荐一份主食”。抽取由图片哈希、要求池指纹和种子确定，同一实验复测保持一致。要求的预算、人数、忌口、偏好和汇率信息一并保存。

本地生成不调用评估 API。同一命令再次运行会跳过成功任务并重试失败任务。更换模型、图片集、提示词、依赖或生成参数时使用新的 `OUTPUT_ROOT`；同一输出目录只运行一个生成进程。

## 第一阶段输出与跨平台交接

```text
OUTPUT_ROOT/
├── handoff.json                    # 跨平台只需复制这一份生成产物
├── primary/                        # 原始集的断点与运行记录
│   ├── generation_results.jsonl
│   ├── run.json
│   ├── manifest.jsonl
│   ├── generation_meta.json
│   ├── prompts/
│   └── samples/图片ID/
│       ├── summary.json
│       ├── interpretation.json
│       └── recommendation.json
└── menu_reference/                 # 辅助集使用同样结构，独立保存
```

`handoff.json` 包含两套数据标识、三任务原文、实际提示词、推荐要求、模型运行信息和图片 SHA256。原图以 Base64 保存完整文件字节，没有再次压缩或模糊处理，因此 JSON 文件可能较大。

把 `handoff.json` 和完整 `评估.py` 放到第二平台即可。无需复制原始图片目录、模型、生成环境或第一平台的断点目录。生成任务中的 `status=ok` 只表示模型生成成功，不代表答案正确；非菜单图片输出“无法推荐”是否合理，由第二平台根据原图独立判断。

## 第二平台：接口配置与评估

完整 [评估.py](评估.py) 已集成 `hm_aigc.aigc_managed`，包含参考构造、评分、指标计算和 HTML 生成。第二平台需要 Python 3.10 或以上，以及平台提供的接口 SDK；不需要安装 Qwen、torch、Transformers 或本项目包。

在该文件顶部填写：

| 配置 | 用途 |
| --- | --- |
| `DEFAULT_ACCOUNT` | 平台子账号 |
| `DEFAULT_MODEL` | 接口实际模型名称，决定调用哪个模型 |
| `DEFAULT_MODEL_VERSION` | 测评记录中的模型或服务版本标识，不决定调用模型；未知时可填写“未提供”，同一实验保持一致 |
| `DEFAULT_MODEL_AGENT` | 默认 `yibu`，按平台配置使用 |
| `DEFAULT_HM_SDK_MODULE` | 实际 SDK 模块名，默认 `hm_aigc` |
| `DEFAULT_TIMEOUT` / `DEFAULT_MAX_RETRIES` | 超时秒数与最多调用次数 |

如平台需要初始化客户端，在文件顶部使用已有的导入与初始化代码，得到 `hm_aigc` 对象。接口配置齐全后运行：

```bash
python 评估.py handoff.json --output-dir eval_v3
```

默认依次完成独立参考、评分和 HTML，也可以分开执行：

```bash
python 评估.py handoff.json --output-dir eval_v3 --stage reference
python 评估.py handoff.json --output-dir eval_v3 --stage score
```

两次命令使用同一输出目录，第二次复用已保存的参考。参考构造不接收 Qwen 答案；评分接收原图、该图独立参考、任务答案及推荐要求。成功回包立即保存，重跑复用已完成记录，接口失败或非法 JSON 会重试。

仅检查交接文件并准备请求、预览时运行 `python 评估.py handoff.json --output-dir eval_v3 --prepare-only`，此时不会调用接口，也不会得到真实语义评分。通用 `evaluate.py` 保留回包准备与导入流程；使用已提供的平台 SDK 时直接使用 `评估.py`。

```text
第二平台目录/
├── 评估.py
├── handoff.json
└── eval_v3/
    ├── results.json                # 参考、评分、指标和进度
    └── report.html                 # 原图、三任务答案、证据、分数及原因
```

没有指定 `--output-dir` 时，结果默认写到 `handoff.json` 所在目录。尚有未完成请求时可能额外保存请求 JSON。HTML 可离线打开，两套数据分开展示，支持查看每张图的原文、事实参考、计算分子分母、推荐要求和判断依据。

新版评估使用 `task_metrics_v3`；旧评审回包和结果目录不能直接续用，需重新构造参考、评分并使用新输出目录。第一阶段和已有 `handoff.json` 无需重跑。

## 三个任务的提示词

完整原文见 [三个任务提示词](docs/三个任务提示词原文.md)，对应实际生成使用的文件：

| 任务 | 提示词 | 输出要求概览 |
| --- | --- | --- |
| 摘要 | [v04_summary](prompts/v04_summary.json) | 两行：标题不超过10字，描述不超过30字；简体中文，不编造事实 |
| 解读 | [v04_interpret](prompts/v04_interpret.json) | 自判场景，速览不超过50字，1～3个核心模块、0～2个增量模块 |
| 菜品推荐 | [v01_recommendation](prompts/v01_recommendation.json) | 根据图片和抽取要求处理预算、人数、忌口、偏好及未知信息 |

摘要与解读保留用户提供的提示词原文。菜单、餐厅之外的图片也保留三任务生成结果，以观察适用性判断和合理拒绝；评估阶段按独立证据处理。

## 评测指标与适用性

详细定义及示例见 [三页修订版 Word](docs/Qwen三任务评测指标说明_修订版.docx)。不同任务使用各自指标，不计算跨任务总分，也不使用启发式权重。

| 任务 | 主要指标及算法 |
| --- | --- |
| 摘要 | 每图事实正确率＝正确原子事实÷可判定原子事实；名词准确率＝正确实体名词÷可判定实体名词。跨图分别等权平均 |
| 解读 | 同样计算事实正确率；内容分＝100×（正确核心点数＋合格增量点数）÷（独立参考中的核心点数＋增量机会数），各点均为1点 |
| 推荐 | 正确率＝正确图片数÷（正确图片数＋不正确图片数）；同时给出逐图0、1、2等级或待判定 |

事实判定依据原图可见证据及独立参考，未知项单列，不作为正确项。内容点在参考阶段独立选定，必须适合原有篇幅要求；增量内容只有准确、相关、有价值且不重复时得分。未被独立参考充分覆盖的新内容需要复核，不自动作为错误或奖励。

推荐先判断图片是否存在可用餐品信息，再判断要求是否可行。非菜单标签本身不能直接排除：餐品海报有足够信息也可适用；没有可用餐品信息的图片标记不适用。缺价格、忌口证据或汇率则保留不确定性，不假装满足约束。已证实无解时，有图片依据的合理拒绝可以算正确；有可行方案却直接拒绝算不正确。

推荐等级分别为：

- **0级**：已确认不正确，例如虚构菜品、价格算错、超预算或违反硬性要求。
- **1级**：已确认正确，但存在经事实、价格和硬约束核验的替代组合，至少改善一项明示软偏好且其他明示偏好不退步。
- **2级**：已确认正确，全部明示软偏好满足或没有软偏好；有依据的无解拒绝也为2级。不代表已证明全局最优。
- **待判定**：正确性或偏好比较所需证据不足。软偏好待判定不抹去已经确认的硬性正确性。

价格、数量、必收费用和显式汇率由程序使用十进制复算；未知费用不能默认零，币种不能凭符号猜测。推荐正确率同时展示正确、不正确、待判定和排除数量，待判定及不适用不进入分母。摘要与解读也显示有效样本数；适用图片上的空输出或纯拒绝按零分处理。

## 两套数据和代码结构

| 数据集 | 内容 | 统计方式 |
| --- | --- | --- |
| `primary` | 用户原始图片集，含多语种、截图和实拍 | 独立统计三任务指标与分母 |
| `menu_reference` | 整理的20张清晰、无手菜单原图 | 独立统计，不与原始集混合均分或正确率 |

辅助图片来源和清单见 [图片清单](datasets/menu_reference/图片清单.json)。部分菜单没有价格，其预算结论需保留未知。

```text
qwen-image-text-eval/
├── README.md
├── generate.py                     # 第一平台：三任务生成和交接导出
├── scripts/generate.sh             # 第一阶段启动脚本，支持环境变量配置
├── 评估.py                         # 第二平台：已集成SDK的独立完整入口
├── evaluate.py                     # 通用单文件参考、评分和报告入口
├── requirements-generation.txt     # 本地生成依赖
├── prompts/                        # 生成及评审提示词
├── configs/                        # 生成参数和推荐要求池
├── datasets/menu_reference/        # 辅助菜单原图与来源清单
├── src/qwen_text_eval/              # 扫描、抽样、生成、断点和交接导出
├── docs/                           # 提示词原文、模型说明和指标Word
└── tests/
```

自动测试覆盖模拟模型、交接、评估协议、预算计算和模拟 SDK 调用，不代表真实 GPU 或接口已在所有平台验证。GPT 构造的是待核验事实参考；重要结论、疑难图片及模型间分歧需人工复核，不能保证语义判定完全无误。

## 独立展示：历史 Agent 文字质量实验

[展示文件夹](docs/agent-text-report/README.md)包含45个文字任务的输入、原始输出、最终结果、评分规则、长度统计及20张合成菜单原图。下载后打开`docs/agent-text-report/index.html`即可离线演示，无需安装依赖。

这是2026-10-11的Qwen API与Agent工程实验，**与本仓库的本地模型三任务评测分开**。30例提供事后客观百分制，其余15例保留通过性审查；不同实验的分数不合并。仅展示所需的合成菜单与文字结果在该目录公开，不改变其他原始图片和运行数据的排除规则。
