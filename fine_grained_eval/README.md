# 四项指标：三阶段独立评测工具

Python **3.10+**，运行代码只依赖标准库及平台提供的 `hm_aigc` SDK。不加载Qwen权重、不执行本地推理。现有仓库根目录的 `evaluate.py` 和 `评估.py` 保持原样；本目录为独立新版。

```text
图片目录 + 实际任务配置
        │ gen_truth.py（强模型标注 + 三次独立摘要生成 + 另一次机器复核）
        ▼
truth.jsonl ─────────────────────────────┐
                                        │ evaluate.py（逐项评审 + 程序计算）
你现有Qwen推理 → 后续change.py → predictions.jsonl
                                        ▼
                                evaluations.jsonl
                                        │ gen_html.py
                                        ▼
                         report.html（自包含、离线可看）
```

接口的强制规定见 [PROTOCOL.md](PROTOCOL.md)，公式和量表见 [METRICS.md](METRICS.md)，可机读规范见 [schemas/](schemas/)。请优先按协议编写后续 `change.py`，不要让评分程序自动猜测旧数据的字段、任务或实际提示词。

## 运行

以下命令在本目录执行。配置中的账号、模型及SDK必须按实际平台填写；示例为空，不含凭据。

```bash
# 将 api_config.example.json 复制为 api_config.local.json 后填写。
# task_config.json 中的system/user必须是你实际用于Qwen的提示词。
python gen_truth.py /path/to/images --output truth.jsonl --api-config api_config.local.json --task-config task_config.json --export-inputs input_manifest.jsonl

# change.py由你后续实现，输出严格符合协议的 predictions.jsonl。
python evaluate.py truth.jsonl predictions.jsonl --output evaluations.jsonl --api-config api_config.local.json --tasks summary interpretation recommendation phrase

python gen_html.py evaluations.jsonl --output report.html --summary-output report_summary.json
# get_html.py 是 gen_html.py 的同参数兼容入口。
```

只运行部分任务时明确指定，例如 `--tasks summary interpretation`。对每个模型及运行批次，评估器构造“truth中配置的图片×本次选择任务”的预期记录；缺失的适用模型输出计0，不静默丢弃。如果整个批次都没有输出，额外传 `--model-id 模型名称 --run-id 批次名称`，仍可统计缺失。

`--resume` 跳过同签名的已完成记录，参考生成失败和评审失败会重试；每完成一张图/一条结果即原子保存。图片、提示词、上下文、标准或评审配置变更时拒绝复用旧记录，应换输出文件。程序不会自动修改模型答案。

## API 配置与调用

在 `api_config.local.json` 填写 `account`、`reference_model/reference_version`、`judge_model/judge_version`。`model_agent` 默认为用户示例中的 `yibu`。也可使用环境变量 `FG_EVAL_ACCOUNT`、`FG_EVAL_REFERENCE_MODEL`、`FG_EVAL_REFERENCE_VERSION`、`FG_EVAL_JUDGE_MODEL`、`FG_EVAL_JUDGE_VERSION` 覆盖；不要提交本地配置。

调用严格沿用给定接口：

```python
client.aigc_managed(
    model_agent=config["model_agent"],
    req_data={"model": model, "messages": messages, **request_options},
    sub_account_name=config["account"],
    model=model,
    timeout=config["timeout"],
)
```

图片以 `image_url` data URL放入多模态消息；schema与输入作为JSON文本提供，要求接口返回JSON对象字符串。程序处理 `choices[0].message.content`，可接受一个完整的JSON代码围栏；拒绝截断、非法结构、虚构证据ID和不来自候选原文的引用，并在重试额度内重试。

`sdk_module` 指向可导入模块，支持三种形式：模块直接提供 `aigc_managed`、模块提供 `hm_aigc` 对象，或模块提供 `create_client()` 工厂。若平台要先登录或初始化，把初始化放在自建模块的 `create_client()` 中，并填写模块名即可。`request_options` 默认空对象；如平台支持 `temperature`、`max_tokens` 等，可按其实际参数加入，程序不会猜测平台特有字段。

参考生成和复核是独立请求，但不声称两个配置必然是不同模型；建议使用足够强的参考模型和独立评审配置。没有真实账号/SDK时先用准备模式，不能将模拟回包测试称为真实接口或语义评测成功。

```bash
python gen_truth.py /path/to/images --output truth.jsonl --api-config api_config.example.json --task-config task_config.json --prepare-only --export-inputs input_manifest.jsonl
python evaluate.py truth.jsonl predictions.jsonl --output evaluations.jsonl --api-config api_config.example.json --prepare-only
```

准备模式不初始化SDK、不调用API、不伪造truth或评分结果；仅写相邻的 `*.requests.jsonl`。参考后续请求依赖第一步真实参考，故参考准备模式只生成第一步请求。

## 任务与语言配置

`task_config.example.json` 是宽泛提示词示例，不是对你现有YAML的替代。必须把当前YAML加载后的真实 `system/user` 转成此配置，或用 `--manifest per_image.jsonl` 对某些图片覆盖prompt和共享上下文。

```json
{"image_name":"menu.jpg","context":{"locale":"FR","acceptable_languages":["fr"]}}
```

当地信息必须也是Qwen实际可见的输入。若需将地点/语言追加到system，请在本地推理和参考配置中保存**完全相同的追加后提示词**；不能只在评分端补充候选从未获得的地区信息。显式语言要求应同时登记在 `acceptable_languages` 中；多语地区可以列多个。没有可靠语言信息时，phrase为N/A，不能默认英语或由欧元推断法语。

原始脚本一次输出“组合＋常用语”时，可用两个JSONL记录分别写 `task=recommendation` 和 `task=phrase`，两条保留**同一份原文、同一实际prompt**，并在参考配置中给这两个task设置相同prompt。只有prompt和输出完全相同的联合记录，才自动将推荐内容作为常用语的配对依据；不同请求生成的组合不能拿来替另一份表达纠错。

需要专业外部知识时，可给 `gen_truth.py` 增加 `--knowledge knowledge.jsonl`。按图片提供真实资料摘录、来源URL、版本/日期，格式见协议；程序不自动联网检索，也不允许参考模型自编来源。未提供可靠资料的专业推断保留未知。这些资料用于建立评分依据，不能因此强求待测模型输出不属于本任务必要范围的知识细节。

## 实现的指标

每图/任务分别计算，再按模型、批次、任务及场景汇总，N/A不参与均值。报告不混合不同模型或批次。

| 指标 | 程序计算 |
|---|---|
| 摘要 | 有效事实精确率与必需覆盖率的F分数，乘 `min(1, 参考长度/实际长度)` |
| AI解读 | 各适用字段正确完成率等权平均，乘 `K/(K+H)`；重复错误不双扣 |
| 组合 | 通过约束数÷适用约束数；独立核算菜品、数量、预算、忌口和金额 |
| 常用语 | 成功表达单元数÷预定单元数；语言适用、意图、组合对应及可沟通性同时满足 |

摘要参考长度使用同一图片三次独立参考生成中，机器复核接受的至少两份完整摘要长度的中位数。字符不含空白，不把每次API的token上限当作摘要标准长度。解读保留项级证据、字段明细、额外错误和关键失败。

组合采用固定类别，避免通过大量拆分容易约束稀释违规；过敏原信息不完整或辣度未知时不能认定约束已满足，保留unknown并计未满足。人工评价界限是标签证据，不是实际餐厅食用安全保证。用户自定义、无法由程序判定的约束保留逐项语义评审及证据；它们不会被假称为程序精确证明。

模型失败/空回答/缺输出计0；参考生成或评审API失败保留null分数，单列待评数量，不能当模型0分或悄悄从最终报告消失。存在待评时，均值明确标为部分结果。`gen_html.py` 会根据逐项记录重新计算分数，发现手工改分或不一致即报错。

报告另列非空白字符数、逐图摘要长度比、实际记录的输出token数和截断情况。截断比例只以结束原因已知的输出为分母，未记录者单列；不靠字符数猜token。每张适用摘要图片正常需要5次参考阶段调用；其余任务的逐项评分各1次。未启用摘要时参考阶段为2次；重试会增加调用，实际usage以SDK返回为准。

## 验证与限制

```bash
python -m unittest discover -s tests -v
python contracts.py schemas
```

测试使用合成SDK回包验证协议、公式、重试、截断、缺失输出、缓存、引用、机器标准复核失败及HTML转义。测试不证明视觉/语义判定正确；真实接口联调需操作者提供SDK、账号和模型版本。机器参考默认 `human_verified=false`，不会自动声称完成人工核验。

主流程、所有文件和调用记录均保留在本地。没有自动上传图片、发送消息或启动GPU训练的操作；API调用是上述配置的图片发送目的地。
