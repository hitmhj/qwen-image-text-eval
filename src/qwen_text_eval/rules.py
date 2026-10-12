"""可审计的格式检查；语义、专名和场景不确定项交给人工或视觉 LLM。

不改写被测原文。字数统一按去掉所有空白后的 Unicode 码点数计算，
汉字、标点、数字及外文字符均计入。摘要允许一个文件末尾换行符；
第二个末尾换行仍代表空行。CRLF 视为一个换行符。
"""

from __future__ import annotations

import re
from functools import lru_cache
from typing import Any


MEDICAL_DISCLAIMER = "具体用药请遵医嘱，本内容不构成使用建议"

_SCENE_ALIASES = {
    "menu": "menu", "restaurant-menu": "menu", "restaurant_menu": "menu",
    "restaurant menu": "menu", "餐厅菜单": "menu", "菜单": "menu",
    "restaurant": "restaurant", "餐厅": "restaurant", "餐厅实景": "restaurant",
    "beauty": "beauty", "美妆护肤品": "beauty",
    "food": "food", "食品饮品包装": "food",
    "medicine": "medicine", "medical": "medicine", "药品保健品": "medicine",
    "药品": "medicine", "保健品": "medicine",
    "landmark": "landmark", "地标景点地图": "landmark",
    "media": "media", "书籍/海报/影视": "media",
    "document": "document", "文档/通知/票据": "document",
    "general": "general", "通用": "general", "通用（兜底）": "general",
}

_MODULE_HEADER = re.compile(
    r"^\s*(?:#{1,6}\s*)?(?:\*\*)?【([^】\r\n]+)】(?:\*\*)?\s*$"
)
_BULLET = re.compile(r"^\s*(?:[-*+•]\s+|\d+[.)、]\s*)\S")
_PREFIX = re.compile(
    r"^\s*(?:#{1,6}\s*|\*\*|-\s*)?"
    r"(?:以下是|(?:标题|描述|摘要)\s*[:：])", re.MULTILINE
)
_THINKING = re.compile(
    r"</?\s*(?:think|thinking|analysis)\b[^>]*>|```\s*(?:analysis|thinking)\b",
    re.IGNORECASE,
)
_SUMMARY_MARKDOWN = re.compile(
    r"```|`[^`]*`|\*\*|__|~~[^~]+~~|"
    r"^\s*(?:#{1,6}\s+|[-*+]\s+|\d+[.)]\s+|>\s*|(?:---+|\*\*\*+)\s*$)|"
    r"!?\[[^\]\n]+\]\([^\)\n]+\)|(?<!\w)\*[^*\n]+\*(?!\w)|"
    r"(?<!\w)_[^_\n]+_(?!\w)",
    re.MULTILINE,
)
_FOREIGN = re.compile(
    r"[A-Za-z\u00c0-\u024f\u0370-\u052f\u0590-\u08ff"
    r"\u0900-\u0dff\u0e00-\u0fff\u3040-\u30ff\uac00-\ud7af]"
)
_FOREIGN_PARENS = re.compile(r"[（(]([^（）()\r\n]+)[）)]")
_SUBJECTIVE = re.compile(r"最值得|超好用|超值|最好|最优|最佳|强烈推荐|必点|必买")

# 只检出明显图形 emoji、旗帜、键帽及使用 VS16 的符号；不会把一二三、
# ①②③、普通数字、货币符号或品牌商标误当作 emoji。
_EMOJI = re.compile(
    r"[\U0001f300-\U0001faff\U0001f1e6-\U0001f1ff]|"
    r"[0-9#*]\ufe0f?\u20e3|"
    r"[\u2300-\u23ff\u2500-\u27ff\u00a9\u00ae]\ufe0f|"
    r"[\u231a\u231b\u23e9-\u23ec\u23f0\u23f3\u25fd\u25fe"
    r"\u2614\u2615\u2648-\u2653\u267f\u2693\u26a1\u26aa\u26ab"
    r"\u26bd\u26be\u26c4\u26c5\u26ce\u26d4\u26ea\u26f2\u26f3"
    r"\u26f5\u26fa\u26fd\u2705\u270a\u270b\u2728\u274c\u274e"
    r"\u2753-\u2755\u2757\u2795-\u2797\u27b0\u27bf]"
)


def count_characters(text: str) -> int:
    """Unicode 码点数，排除空白；不进行字数截断或文本清洗。"""
    return sum(not character.isspace() for character in text)


def _lines(text: str) -> list[str]:
    lines = re.split(r"\r\n|\n|\r", text)
    if text.endswith(("\n", "\r")):
        lines.pop()  # 一个末尾换行符是终止符；额外换行符仍保留为空行。
    return lines


@lru_cache(maxsize=1)
def _opencc_converter() -> Any:
    try:
        from opencc import OpenCC  # 可选增强，不是生成平台的强制依赖。
    except ImportError:
        return None
    try:
        return OpenCC("t2s")
    except (OSError, ValueError, RuntimeError):
        return None


def evaluate_rules(text: str, task: str, scene: str | None = None) -> dict[str, Any]:
    """输出逐条证据和三态合规结论，不评估事实准确率。

    hard_pass 为 false：至少一项可以确定的硬约束失败；为 null：没有确定
    失败，但仍有需要确认的硬约束；为 true：所有适用硬约束均通过。
    语义评审可能始终未决，调用方应分别汇总机械合规率和评审待定率。
    scene 必须来自可信参考标注，不能用被测模型自报的场景替代。
    """
    if not isinstance(text, str):
        raise TypeError("text 必须是字符串")
    if task not in ("summary", "interpretation"):
        raise ValueError("task 必须为 summary 或 interpretation")
    if scene is not None and not isinstance(scene, str):
        raise TypeError("scene 必须是字符串或 None")

    scene_kind = _SCENE_ALIASES.get(scene.strip().lower()) if scene else None
    char_count = count_characters(text)
    checks: list[dict[str, Any]] = []

    def add(
        check_id: str, name: str, status: str, observed: Any,
        expected: Any, details: str,
    ) -> None:
        checks.append({
            "id": check_id, "name": name, "status": status,
            "observed": observed, "expected": expected, "details": details,
        })

    nonempty = bool(text.strip())
    add("nonempty", "有效输出", "pass" if nonempty else "fail", char_count,
        "至少一个非空白字符", "空白输出不能用通过格式检查代替生成成功。")
    if not nonempty:
        return {"task": task, "char_count": char_count, "checks": checks,
                "hard_pass": False, "needs_review": False}

    lines = _lines(text)
    thinking = _THINKING.findall(text)
    add("thinking_markup", "无思考过程标记", "fail" if thinking else "pass",
        thinking, "无 think/thinking/analysis 标记",
        "评估最终正文；残留思考标签或思考代码块为确定性失败。")
    prefixes = [match.group(0).strip() for match in _PREFIX.finditer(text)]
    add("forbidden_prefix", "无禁止前缀", "fail" if prefixes else "pass",
        prefixes, "无标题、描述、摘要标签或“以下是”前缀",
        "只检查行首前缀，避免误判正文对图片文字的引用。")
    emoji = _EMOJI.findall(text)
    add("emoji", "无 emoji", "fail" if emoji else "pass", emoji,
        "无明显 emoji", "保守检出常见 emoji；普通数字和中文序数不算 emoji。")

    converter = _opencc_converter()
    if converter is None:
        add("simplified_chinese", "简体中文", "review", None, "以简体中文输出",
            "未安装可选 opencc，简繁体需由人工或 LLM 核对；不改变被测文本。")
    else:
        converted = converter.convert(text)
        changed = list(dict.fromkeys(
            original for original, simplified in zip(text, converted)
            if original != simplified
        ))[:30]
        add("simplified_chinese", "简体中文", "review" if converted != text else "pass",
            {"conversion_changes_text": converted != text, "candidate_characters": changed},
            "以简体中文输出",
            "转换差异仅是候选证据，需排除专有名词及转换歧义；不自动改写或判错。")

    foreign = list(dict.fromkeys(_FOREIGN.findall(text)))[:30]
    add("chinese_language", "中文与专有名词边界", "review" if foreign else "pass",
        foreign, "普通内容用简体中文；解读允许品牌、地名等专名保留原文",
        "外文字符不能直接判错；需判断专名、型号、单位和普通词。摘要的简体中文要求更严格。")
    foreign_parens = [match.group(0) for match in _FOREIGN_PARENS.finditer(text)
                      if _FOREIGN.search(match.group(1))]
    add("foreign_parenthetical", "无外文括注", "review" if foreign_parens else "pass",
        foreign_parens, "不夹带普通词的原文括号注释",
        "需区分品牌专名与普通词；菜单模块要求原文名和全局禁止括注存在冲突，交给评审标注。")
    subjective = _SUBJECTIVE.findall(text)
    add("subjective_wording", "禁止评价性措辞", "review" if subjective else "pass",
        subjective, "只陈述事实和可比较维度，不作主观评价",
        "命中词需结合语境判断，图片原文引用不能直接算模型主观评价；未命中也不证明语义合规。")

    if task == "summary":
        markdown = [match.group(0) for match in _SUMMARY_MARKDOWN.finditer(text)]
        add("summary_markdown", "摘要无 Markdown", "fail" if markdown else "pass",
            markdown, "两行纯文本", "识别标题、列表、代码、加粗和链接等明显 Markdown 标记。")
        add("summary_line_count", "摘要恰好两行", "pass" if len(lines) == 2 else "fail",
            len(lines), 2, "CRLF 算一次换行，允许一个末尾终止换行；额外终止换行视为空行。")
        blank_indices = [index + 1 for index, line in enumerate(lines) if not line.strip()]
        add("summary_no_blank_lines", "摘要无空行", "fail" if blank_indices else "pass",
            blank_indices, "无空行", "仅含空格、制表符或其他空白的行也算空行。")
        if len(lines) == 2 and not blank_indices:
            title_count, description_count = map(count_characters, lines)
            add("summary_title_length", "标题字数", "pass" if title_count <= 10 else "fail",
                title_count, "1~10 个非空白码点", "标点、数字、外文均计入标题字数。")
            add("summary_description_length", "描述字数",
                "pass" if description_count <= 30 else "fail", description_count,
                "1~30 个非空白码点", "标点、数字、外文均计入描述字数。")
            normalized_title = "".join(char for char in lines[0] if not char.isspace())
            normalized_description = "".join(char for char in lines[1] if not char.isspace())
            add("summary_nonrepetition", "标题与描述不重复", "review",
                {"title_contained_in_description": normalized_title in normalized_description},
                "描述提供额外信息，不重复标题内容",
                "字符串包含只是线索；描述同义复述、主体复用及新信息价值由 LLM 判断。")
        else:
            for check_id, name, limit in (
                ("summary_title_length", "标题字数", 10),
                ("summary_description_length", "描述字数", 30),
            ):
                add(check_id, name, "not_applicable", None, f"不超过 {limit} 字",
                    "输出不是两行非空正文，无法可靠识别标题和描述；行数检查已记失败。")
            add("summary_nonrepetition", "标题与描述不重复", "not_applicable", None,
                "描述提供额外信息", "两行结构无效，暂不评估语义重复。")
    else:
        headers = [(index, _MODULE_HEADER.match(line)) for index, line in enumerate(lines)]
        headers = [(index, match) for index, match in headers if match]
        first_header = headers[0][0] if headers else len(lines)
        preamble = lines[:first_header]
        overview_lines: list[str] = []
        for line in preamble:
            if line.strip():
                overview_lines.append(line)
            elif overview_lines:
                break
        overview_count = count_characters("\n".join(overview_lines))
        add("interpretation_overview", "内容速览存在且不超过 50 字",
            "pass" if 0 < overview_count <= 50 else "fail", overview_count,
            "第一段 1~50 个非空白码点", "首个模块前的第一个非空段落视为内容速览。")
        add("interpretation_module_count", "模块总数", "pass" if 1 <= len(headers) <= 5 else "fail",
            len(headers), "1~5 个带【】标题的模块",
            "原要求是 1~3 核心和 0~2 增量；机械检查仅能确认总数，角色分配另行评审。")
        module_points: list[dict[str, Any]] = []
        for position, (start, match) in enumerate(headers):
            end = headers[position + 1][0] if position + 1 < len(headers) else len(lines)
            module_points.append({
                "title": match.group(1),
                "point_count": sum(bool(_BULLET.match(line)) for line in lines[start + 1:end]),
            })
        points_valid = all(2 <= module["point_count"] <= 4 for module in module_points)
        add("interpretation_module_points", "每模块 2~4 条要点",
            ("pass" if points_valid else "fail") if headers else "not_applicable",
            module_points, "每模块 2~4 条列表要点",
            "识别短横线、星号、加号、圆点和数字列表；续行不重复计数。")
        add("interpretation_module_roles", "核心与增量模块分配",
            "review" if headers else "not_applicable",
            [module["title"] for module in module_points], "1~3 核心模块、0~2 有价值的增量模块",
            "标题由模型自取，不能按标题关键词机械确定核心/增量角色；须结合图片评审。")
        if scene_kind == "menu":
            length_status = "not_applicable"
            length_details = "可信参考标注为餐厅菜单，原提示词明确豁免全文 500 字限制。"
        elif char_count <= 500:
            length_status = "pass"
            length_details = "全文在 500 字内，即使场景未知也符合长度上限。"
        elif scene_kind is None:
            length_status = "review"
            length_details = "全文超过 500 字，但缺少可信场景标注；须先确认是否餐厅菜单。"
        else:
            length_status = "fail"
            length_details = "可信参考标注为非菜单场景，全文必须不超过 500 字。"
        add("interpretation_total_length", "非菜单解读全文字数", length_status, char_count,
            "非菜单不超过 500 个非空白码点；菜单豁免", length_details)
        disclaimer_present = text.rstrip().rstrip("。.!！").endswith(MEDICAL_DISCLAIMER)
        if scene_kind == "medicine":
            medical_status = "pass" if disclaimer_present else "fail"
            medical_details = "药品/保健品须以指定声明结尾；允许句末标点和空白。"
        elif scene_kind is None:
            medical_status = "review"
            medical_details = "场景未知，不能据此跳过医疗声明；需参考图片判定是否药品/保健品。"
        else:
            medical_status = "not_applicable"
            medical_details = "可信参考标注为非药品/保健品场景。"
        add("medical_disclaimer", "医疗内容结尾声明", medical_status, disclaimer_present,
            MEDICAL_DISCLAIMER, medical_details)

    needs_review = any(check["status"] == "review" for check in checks)
    hard_pass: bool | None
    if any(check["status"] == "fail" for check in checks):
        hard_pass = False
    elif needs_review:
        hard_pass = None
    else:
        hard_pass = True
    return {"task": task, "char_count": char_count, "checks": checks,
            "hard_pass": hard_pass, "needs_review": needs_review}
