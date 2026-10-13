"""按图片内容和种子稳定抽取推荐要求，保存硬约束及汇率上下文。"""

import copy
import json
from decimal import Decimal, InvalidOperation

from .storage import fingerprint


def validate_pool(pool: dict) -> None:
    if not isinstance(pool, dict) or not isinstance(pool.get("version"), str) or not pool["version"]:
        raise ValueError("要求池必须含非空版本")
    rows = pool.get("requirements")
    if not isinstance(rows, list) or not rows:
        raise ValueError("要求池不能为空")
    seen = set()
    for row in rows:
        if not isinstance(row, dict) or not all(isinstance(row.get(k), str) and row[k] for k in ["id", "text"]):
            raise ValueError("每条要求必须含非空标识和正文")
        if row["id"] in seen:
            raise ValueError("要求池存在重复标识")
        seen.add(row["id"])
        constraints = row.get("constraints", {})
        budget = constraints.get("budget", {})
        try:
            amount = Decimal(str(budget["amount"]))
        except (KeyError, InvalidOperation) as error:
            raise ValueError("要求预算必须为有效十进制金额") from error
        if not amount.is_finite() or amount <= 0 or not isinstance(budget.get("currency"), str) or budget.get("scope") != "total":
            raise ValueError("预算必须是正数、明确币种及总预算")
        if type(constraints.get("diners")) is not int or constraints["diners"] < 1:
            raise ValueError("用餐人数必须为正整数")
        for field in ["hard_constraints", "preferences"]:
            if not isinstance(constraints.get(field), list):
                raise ValueError("硬约束与偏好必须分别保存为列表")
            ids = [item.get("id") for item in constraints[field] if isinstance(item, dict)]
            if len(ids) != len(constraints[field]) or any(not isinstance(item, str) or not item for item in ids) or len(ids) != len(set(ids)):
                raise ValueError("硬约束与偏好必须包含唯一非空标识")
            if field == "hard_constraints" and set(ids) & {"budget", "diners", "min_main_dishes", "min_dishes", "exact_main_dishes", "min_non_alcoholic_drinks"}:
                raise ValueError("硬约束标识不能与预算、人数或餐品数量字段冲突")
        if not isinstance(constraints.get("meal_requirements"), dict):
            raise ValueError("要求中缺少餐品数量字段")
        for value in constraints["meal_requirements"].values():
            if value is not None and (type(value) is not int or value < 1):
                raise ValueError("餐品数量必须为正整数或 null")


def validate_fx_context(context: dict | None) -> None:
    if context is None:
        return
    if not isinstance(context, dict) or not all(context.get(k) for k in ["as_of", "source", "rates"]):
        raise ValueError("汇率上下文必须注明日期、来源及 rates")
    if not isinstance(context["rates"], dict):
        raise ValueError("rates 必须为币种对应关系，例如 EUR 到 CNY")
    for source, destinations in context["rates"].items():
        if not isinstance(source, str) or not isinstance(destinations, dict) or not destinations:
            raise ValueError("汇率需要明确来源币种和目标币种")
        for target, value in destinations.items():
            try:
                number = Decimal(str(value))
            except InvalidOperation as error:
                raise ValueError("汇率必须为有效十进制数") from error
            if not isinstance(target, str) or not number.is_finite() or number <= 0:
                raise ValueError("汇率必须为正数")


def select_requirement(image_sha256: str, pool: dict, seed: int, fx_context: dict | None = None) -> dict:
    validate_pool(pool)
    validate_fx_context(fx_context)
    if not isinstance(image_sha256, str) or len(image_sha256) != 64 or type(seed) is not int:
        raise ValueError("要求选择需要图片 SHA256 和整数种子")
    pool_hash = fingerprint(pool)
    selector = fingerprint({"image_sha256": image_sha256, "seed": seed, "pool_sha256": pool_hash})
    row = copy.deepcopy(pool["requirements"][int(selector, 16) % len(pool["requirements"])])
    return {"pool_version": pool["version"], "pool_sha256": pool_hash, "selection_seed": seed,
            **row, "fx_context": copy.deepcopy(fx_context)}


def render_recommendation_prompt(prompt: dict, requirement: dict) -> dict:
    result = copy.deepcopy(prompt)
    context = requirement.get("fx_context")
    context_text = "无" if context is None else json.dumps(context, ensure_ascii=False, sort_keys=True)
    result["user"] = prompt["user"].format(requirement=requirement["text"], context=context_text)
    return result
