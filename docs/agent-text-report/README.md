# 基本结论、分类样例与评测方案

下载完整文件夹，双击 `index.html`，可离线阅读。页面仅用黑白文字、表格和原始记录，没有筛选器、弹窗或外部依赖。

| 入口 | 内容 |
|---|---|
| [index.html](index.html) | 既有实验的基本结论、指标与长度/截断 |
| [examples.html](examples.html) | 15个分类样例，逐例展示输入、模型输出、问题、结论 |
| [all-cases.html](all-cases.html) | 全部45例；附程序修正过程、旧评分、字符数和token |
| [evaluation-design.html](evaluation-design.html) | 面向模型基础能力的新评测方案：自然分流、五类领域、四类任务、证据与评分系统 |
| [report.md](report.md) | 基本结论的Markdown版 |
| [evaluation-design.md](evaluation-design.md) | 评测方案的Markdown版，含参考来源和6个构造判分示例 |
| [evaluation-case-template.json](evaluation-case-template.json) | 未填充的样本、证据、标注与运行记录模板 |
| [data/cases.json](data/cases.json) | 既有45例数据 |
| [data/integrity.json](data/integrity.json) | 文件SHA-256清单 |

实测数据来自2026-10-11 v6.5 Agent API实验，2026-10-13作文字复核。本次2026-10-21仅重整呈现并提出新方案，没有新增模型调用；新方案和构造例子不属于实测结果。本报告与仓库本地模型三任务评测分开。

最终系统成绩含程序检查和模型重试；模型基础能力应首先看未经该次程序纠错的提交，并用新的独立金标准补测。旧版百分制保留在附录，不作为跨场景能力总分。
