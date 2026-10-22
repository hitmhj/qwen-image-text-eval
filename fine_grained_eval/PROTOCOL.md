# 严格对接协议 `fg-eval/1.1`

## 1. 文件与索引

所有交换文件采用UTF-8 JSONL：**一行一个JSON对象**，不是JSON数组；可有UTF-8 BOM和空行。拒绝重复JSON键、NaN/Infinity、未知顶层字段及不符合schema的类型。所有分数为0–1，不是0–100；未知token等元数据用null，不用0冒充。

图片主键 `image_name` 为**含扩展名的完整文件名**，如 `menu_001.jpg`，不含目录。批量输入递归扫描，目录内同名或仅大小写不同的图片立即报错，须先重命名。名称按原字符精确匹配，不删除扩展名、不做模糊匹配。额外使用原文件字节的SHA-256防止同名换图；`image_path`仅用于定位/展示，不是连接键。

| 文件 | 一行表示什么 | 唯一键 |
|---|---|---|
| truth.jsonl | 一张图及其所有已配置任务的冻结参考 | image_name |
| predictions.jsonl | 某模型某批次对一张图执行一个任务的原始输出 | image_name＋task＋model_id＋run_id |
| evaluations.jsonl | 对一个预期图片/任务/模型/批次的完整评分记录 | 同predictions |

同图多任务必须分行；多个模型或批次也必须分行。`recommendation`与`phrase`是不同任务，不允许将旧版三任务的prompt名字直接当作任意新版task。

## 2. 模型输出：后续change.py必须产生的格式

以下展示单行内容（排版为多行只为阅读）；所有字段必填，无其他字段。实际schema为 [prediction.schema.json](schemas/prediction.schema.json)。

```json
{
  "schema_version": "fg-eval/1.1",
  "image_name": "menu_001.jpg",
  "image_sha256": "原始图片文件的64位小写SHA-256",
  "task": "summary",
  "model_id": "Qwen3.5-35B-A3B/local-revision",
  "run_id": "experiment-20261022-v1",
  "prompt": {"system": "实际完整system", "user": "实际完整user"},
  "context": {"locale": null, "acceptable_languages": []},
  "status": "ok",
  "output": "未经清洗、纠错、重写或截短的模型原始输出",
  "error": null,
  "generation": {
    "elapsed_seconds": 1.234,
    "input_tokens": null,
    "output_tokens": null,
    "max_output_tokens": 1024,
    "finish_reason": null
  }
}
```

`image_sha256`示例文字是占位，适配器必须计算真实值。`task`严格为 `summary`、`interpretation`、`recommendation`、`phrase`之一。`prompt`与`context`必须与该图truth中相应任务**完全相等**；内容不同应重新生成对应参考，不能继续套用旧标准。

生成失败时写 `status="error"`、`output=""`、`error`为非空错误信息；成功时error必须为null。成功但达到输出token上限，仍保留原始文本并以 `finish_reason="length"`记录，不把它转换成空回答。输出为空但status=ok允许进入评测，程序按模型失败计0。

输入脚本没有记录token或结束原因时填null；字符数不是token数，不能靠文本长度估算usage。`max_output_tokens`填写实际运行的 `max_new_tokens`，不知道则null。其余实际生成配置如采样参数可保留在你的原始运行日志；协议中没有的位置不要任意追加，可另建以run_id关联的日志文件。

你给出的本地脚本格式为：

```text
results[image_path][prompt_file] = {
    "prompt_name": "task_version",
    "output": "...",
    "elapsed_seconds": ...
}
```

后续change.py需要读取图片原文件计算SHA，使用原脚本相同的 `load_prompt_yaml()` 逻辑取得实际system/user，并由你明确配置 `prompt_file -> task` 映射。不要只凭 `prompt_name` 字符串猜测任务，更不能从模型回答里推断prompt。出现同名图片、同图同任务两份prompt或两次结果时，分别设置run_id或先明确保留哪个实验，不能后写覆盖前写。

联合组合及表达输出可复制为两个task记录，保留同一份实际prompt与output，分开计分；参考配置也必须对应这两个实际输入。缺少对应prompt YAML或实际运行配置时，应报出数据不足，不能伪造空system以通过对接。

## 3. 任务配置与图片补充信息

`gen_truth.py`必须在看待测答案前知道要评价的**真实提示词**。`task_config.example.json`规定默认任务集合及各自prompt；只保留要执行的任务即可。可用 `--manifest overrides.jsonl` 按image_name覆盖context或指定任务的prompt。格式见 [manifest.schema.json](schemas/manifest.schema.json)。

`context.locale`为双方可见的国家/地区，`acceptable_languages`为双方已知、适用于该场景的语言代码列表，如 `["fr"]`。空列表表示无可靠目标语言，不表示默认英语。评分不得隐瞒目标国家然后要求模型猜中。已在用户prompt中明确语言的，也登记相同语言代码并由参考复核一致性。

`--export-inputs input_manifest.jsonl` 输出不包含事实答案的图片/任务/实际prompt列表，便于change.py或外部推理程序构造准确请求。它不会自行运行Qwen，也不要求外部脚本依赖本工具包。

## 4. 标准答案JSONL

完整schema见 [truth.schema.json](schemas/truth.schema.json)。主要字段如下：

| 字段 | 严格含义 |
|---|---|
| schema_version、image_name、image_path、image_sha256 | 版本与图片身份；image_path为输入目录内相对路径 |
| image_data_url | 原图内嵌字节；评分/报告不依赖原平台路径 |
| tasks、context | 实际任务prompt与共享上下文，禁止隐含变化 |
| status、error | ok或error；失败参考不产生能力分数 |
| reference.scene/readable/evidence | 场景、可辨性、原图或实际外部资料证据 |
| reference.applicability | 每任务适用性及理由；非菜单不能进入组合/点餐表达 |
| reference.summary | 必需事实、限定、允许补充范围 |
| reference.interpretation | 固定字段下的独立项、证据、未知与关键错误标记 |
| 每个事实项 | statement、qualifiers、applicability_reason、reasoning、acceptable_answers、failure_conditions、evidence_ids、unknown_expected、critical全部必填；ID唯一 |
| provided_evidence | 操作者实际提供的外部资料摘录；模型引用必须逐字段原样匹配，不接受新编来源 |
| reference.recommendation | 菜单字段、给定条件、额外语义约束；没有唯一固定选菜答案 |
| reference.phrase | 可接受目标语言、预定沟通单元、槽位及带占位符的参考模板 |
| summary_references、reference_length | 三次独立摘要回包；有效至少两份，参考长度取有效候选字符数中位数 |
| verification、human_verified | 独立机器复核回包；人工核验默认false，程序不代替操作者确认 |
| calls | 每次API的模型/版本、usage、结束原因、请求指纹等，缺usage保留null |
| input_signature、reference_id | 缓存输入签名与冻结内容指纹；标准被修改后旧评分不得继续复用 |

参考只生成独立标准，不接收候选答案。独立生成/复核均是机器过程，不能伪称人工真值。低质量参考保留error行而非静默删图。标准生成和验证模型配置可不同，实际身份均记录。

可选的 `--knowledge knowledge.jsonl` 一行如下。URL和文本必须换成实际核定资料；示例为结构占位，不能作为专业依据。该资料由操作者准备，程序不声称已经联网核实。参考中的外部证据只能原样引用此处证据；没有该输入时禁止模型新造外部来源。

```json
{"image_name":"medicine_001.jpg","evidence":[{"id":"ext1","source":"external","location":"实际文档章节/页码","text":"实际核定的原文摘录","url":"https://example.org/replace-with-real-source","version":"实际版本或访问日期"}]}
```

这些资料不自动注入Qwen输入。如需区分已有知识差距与检索补充收益，应另设run_id进行配对实验，保留各自实际prompt；不能偷偷给一组补充资料后声称是同一输入的基础能力。

## 5. 测评JSONL与汇总

完整schema见 [evaluation.schema.json](schemas/evaluation.schema.json)。每行保留：版本、图片/任务/模型/批次、完整truth与prediction、评审逐项judgement、程序重算的score和details、judge_call、状态、错误与警示。

| evaluation_status | score | 如何解释 |
|---|---|---|
| scored | 0–1 | 评审标签通过协议校验后，由程序计算 |
| missing_prediction / model_error | 0 | 适用主集的模型缺失/失败，保留分母 |
| not_applicable | null | 冻结标准明确不适用，单列数量 |
| reference_error | null | 标准不能成立，待参考复核；不能判断模型能力 |
| judge_error | null | 评审接口/回包失败，需重试；不能当模型差 |

汇总按模型、run_id、task分别计算单图等权均值，另按scene分类。存在参考或评审失败时，报告明确写“部分结果均值”并展示待评数；不把该均值说成完整评测结果。HTML以同一份逐项记录重新计算，发现手工改分等不一致即拒绝生成。

报告内图像、样式全部内嵌，离线可看，显示实际system/user、共享上下文、原始回答、参考内容、目标量表、单图计算细节、问题与结论。参考目标100%表示满足量表的理论目标，不是对另一模型另做实测得到的成绩。

输出长度/截断诊断不加入四项主分。字符数按原文非空白Unicode字符统计；token数仅用generation实际数据。截断数为finish_reason="length"的数量；截断比例分母为finish_reason非空的成功输出数量，其他结束原因未知的样本单列，避免把“没记录”误报为“没截断”。

## 6. 协议变更

此版本不接受“差不多”的字段。需要增加输出字段、别名、场景、任务或修改主公式时，应修改版本、schemas及验证逻辑，并重建相应参考/评分。图片名称只是索引，不能代替文件哈希、真实提示词和运行批次。
