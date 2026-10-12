"""参考标注与评审回包的结构和引用一致性校验。"""

from .storage import fingerprint, text_sha256

SCENES = ["menu", "restaurant", "beauty", "food", "medicine", "landmark", "media", "document", "general"]
CATEGORIES = ["name", "category", "price", "currency", "quantity", "ingredient", "allergen", "time", "location", "restriction", "other"]
TASKS = ["summary", "interpretation"]
NONEMPTY = {"type": "string", "minLength": 1}
HASH = {"type": "string", "pattern": "^[0-9a-f]{64}$"}
IDS = {"type": "array", "items": NONEMPTY, "uniqueItems": True}


def obj(properties, required=None):
    return {"type": "object", "properties": properties, "required": list(properties) if required is None else required,
            "additionalProperties": False}


ANNOTATOR = obj({"model": NONEMPTY, "version": NONEMPTY})
FACT = obj({
    "fact_id": NONEMPTY, "category": {"enum": CATEGORIES}, "source_text": {"type": "string"},
    "normalized_zh": NONEMPTY, "evidence": NONEMPTY, "confidence": {"enum": ["high", "medium", "low"]},
    "importance": {"type": "integer", "minimum": 1, "maximum": 3},
    "required_for": {"type": "array", "items": {"enum": TASKS}, "uniqueItems": True},
    "numeric": {"anyOf": [{"type": "null"}, obj({"value": NONEMPTY, "unit": {"type": "string"},
                  "currency": {"type": ["string", "null"]}, "qualifier": {"type": "string"}})]},
})
REFERENCE_SCHEMA = obj({
    "schema_version": {"const": 1}, "image_id": NONEMPTY, "image_sha256": HASH, "scene": {"enum": SCENES},
    "image_quality": obj({"readability": {"enum": ["clear", "partial", "unreadable"]}, "issues": IDS}),
    "facts": {"type": "array", "items": FACT}, "unreadable_regions": IDS,
    "reference_status": {"enum": ["draft", "human_verified"]}, "annotator": ANNOTATOR,
})
DIMENSION = obj({"score": {"type": ["integer", "null"], "minimum": 0, "maximum": 4},
                 "reason": NONEMPTY, "evidence_fact_ids": IDS})
CLAIM = obj({"claim_id": NONEMPTY, "text": NONEMPTY,
             "verdict": {"enum": ["supported", "contradicted", "unsupported", "unverifiable", "background"]},
             "fact_ids": IDS, "severity": {"enum": ["minor", "major", "critical"]},
             "explanation": NONEMPTY, "category": {"enum": CATEGORIES}},
            required=["claim_id", "text", "verdict", "fact_ids", "severity", "explanation", "category"])
JUDGE_SCHEMA = obj({
    "schema_version": {"const": 1}, "image_id": NONEMPTY, "task": {"enum": TASKS},
    "candidate_sha256": HASH, "reference_sha256": HASH, "scene": {"enum": SCENES},
    "judge_prompt_sha256": HASH,
    "dimensions": {"type": "object", "additionalProperties": DIMENSION, "minProperties": 1},
    "claims": {"type": "array", "items": CLAIM}, "covered_fact_ids": IDS, "critical_errors": IDS,
    "needs_human_review": {"type": "boolean"}, "review_reasons": IDS, "judge": ANNOTATOR,
})


def validate_schema(document: dict, schema: dict) -> None:
    from jsonschema import Draft202012Validator
    failures = sorted(Draft202012Validator(schema).iter_errors(document), key=lambda e: str(list(e.path)))
    if failures:
        error = failures[0]
        field = ".".join(str(item) for item in error.path) or "根对象"
        raise ValueError(f"字段 {field} 未通过结构校验：{error.message}")


def validate_reference(reference: dict, manifest: dict) -> None:
    validate_schema(reference, REFERENCE_SCHEMA)
    if reference["image_id"] != manifest["image_id"] or reference["image_sha256"] != manifest["sha256"]:
        raise ValueError("参考标注的图片标识或摘要与清单不一致")
    ids = [fact["fact_id"] for fact in reference["facts"]]
    if len(set(ids)) != len(ids):
        raise ValueError("参考标注存在重复事实标识")
    if any(fact["required_for"] and fact["confidence"] != "high" for fact in reference["facts"]):
        raise ValueError("只有高置信事实可以设为必须覆盖的事实")
    if reference["image_quality"]["readability"] == "unreadable" and any(
            fact["required_for"] for fact in reference["facts"]):
        raise ValueError("不可读图片不能设置必须覆盖的事实")


def validate_judgement(judgement: dict, reference: dict, candidate: dict, rubric: dict,
                       judge_prompt_sha256: str | None = None) -> None:
    validate_schema(judgement, JUDGE_SCHEMA)
    if judgement["image_id"] != reference["image_id"] or judgement["task"] != candidate["task"]:
        raise ValueError("评审回包的图片标识或任务不一致")
    if judgement["scene"] != reference["scene"]:
        raise ValueError("评审场景与参考标注不一致")
    if judgement["candidate_sha256"] != text_sha256(candidate["text"]):
        raise ValueError("候选文本已经变化，评审回包过期")
    if judgement["reference_sha256"] != fingerprint(reference):
        raise ValueError("参考标注已经变化，评审回包过期")
    if judge_prompt_sha256 is not None and judgement["judge_prompt_sha256"] != judge_prompt_sha256:
        raise ValueError("评审提示词与当前运行不同，回包过期")
    required = set(rubric["tasks"][candidate["task"]]["dimensions"])
    if set(judgement["dimensions"]) != required:
        raise ValueError("评审维度必须与该任务的评分配置完全一致")
    facts = {fact["fact_id"]: fact for fact in reference["facts"]}
    claim_ids = [claim["claim_id"] for claim in judgement["claims"]]
    if len(claim_ids) != len(set(claim_ids)):
        raise ValueError("评审回包存在重复断言标识")
    citations = set(judgement["covered_fact_ids"])
    for dimension in judgement["dimensions"].values():
        citations.update(dimension["evidence_fact_ids"])
    for claim in judgement["claims"]:
        citations.update(claim["fact_ids"])
        if claim["verdict"] in {"supported", "contradicted"} and not claim["fact_ids"]:
            raise ValueError("得到支持或矛盾的断言必须引用事实证据")
    if citations - set(facts):
        raise ValueError("评审回包引用了不存在的事实标识")
    supported_ids = {fid for claim in judgement["claims"] if claim["verdict"] == "supported" for fid in claim["fact_ids"]}
    for fid in judgement["covered_fact_ids"]:
        if fid not in supported_ids:
            raise ValueError("覆盖事实必须有得到支持的断言作为证据")
        if facts[fid]["confidence"] != "high" or candidate["task"] not in facts[fid]["required_for"]:
            raise ValueError("覆盖事实必须属于本任务的高置信关键事实")
    if judgement["needs_human_review"] and not judgement["review_reasons"]:
        raise ValueError("要求人工复核时必须注明原因")


def validate_rubric(rubric: dict) -> None:
    for task in TASKS:
        dimensions = rubric["tasks"][task]["dimensions"]
        weights = [dimension["weight"] for dimension in dimensions.values()]
        if not weights or any(type(weight) not in (int, float) or weight <= 0 for weight in weights):
            raise ValueError("评分权重必须为正数")
        if abs(sum(weights) - 100) > 1e-8:
            raise ValueError(f"{task} 的权重必须合计为 100")
