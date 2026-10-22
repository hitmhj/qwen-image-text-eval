"""Deterministic V1.1 formulas; the judge supplies item labels, never scores."""
from __future__ import annotations

import re
from statistics import mean

from common import char_count
from contracts import JUDGE_SCHEMAS, unique_ids, validate


def validate_judgement(answer, truth, task, output):
    validate(answer, JUDGE_SCHEMAS[task])
    ref = truth["reference"]
    evidence = {e["id"] for e in ref["evidence"]}

    def check_rows(rows, label):
        unique_ids(rows, label)
        for x in rows:
            if "quote" in x and x["quote"] and x["quote"] not in output:
                raise ValueError(f"{label}: quote is not a candidate output substring")
            if not set(x.get("evidence_ids", [])) <= evidence:
                raise ValueError("judge cited unknown evidence")

    if task == "summary":
        rows = answer["claims"]
        check_rows(rows, "claim")
        if answer["answer_kind"] == "normal" and not rows:
            raise ValueError("normal summary needs atomic claims")
        if answer["answer_kind"] == "refusal" and rows:
            raise ValueError("refusal must not conceal claims")
        required = {x["id"] for x in ref["summary"]["required_facts"]}
        for x in rows:
            if len(x["covered_ids"]) > 1 or not set(x["covered_ids"]) <= required:
                raise ValueError("invalid summary fact coverage")
            if x["label"] != "supported_relevant" and x["covered_ids"]:
                raise ValueError("incorrect or irrelevant claim cannot cover a required fact")
            if x["label"].startswith("supported") and not x["evidence_ids"]:
                raise ValueError("supported claim needs evidence")
        keys = [re.sub(r"\s+", "", x["text"]).casefold() for x in rows]
        if len(set(keys)) != len(keys):
            raise ValueError("duplicate claims; deduplicate before scoring")
    elif task == "interpretation":
        check_rows(answer["items"], "interpretation item")
        expected = {x["id"] for f in ref["interpretation"]["fields"] for x in f["items"]}
        if {x["id"] for x in answer["items"]} != expected:
            raise ValueError("interpretation item ID set must match reference exactly")
        by_id = {x["id"]: x for x in answer["items"]}
        for x in answer["items"]:
            check_verdict(x)
        check_rows(answer["extra_claims"], "extra claim")
        normalized = set()
        for x in answer["extra_claims"]:
            related = x["related_item_id"]
            if related is not None and related not in expected:
                raise ValueError("unknown related item")
            if related and x["label"] != "valid_optional" and by_id[related]["status"] == "correct":
                raise ValueError("contradictory extra cannot coexist with correct related item")
            key = re.sub(r"\s+", "", x["text"]).casefold()
            if key in normalized:
                raise ValueError("duplicate extra claim")
            normalized.add(key)
    elif task == "recommendation":
        # A selection can be wrong; extraction must not force it to match the menu.
        ids = [s["id"] for s in answer["selections"]]
        if len(ids) != len(set(ids)):
            raise ValueError("merge repeated selections before extracting quantities")
        for s in answer["selections"] + answer["amount_claims"]:
            if s["quote"] not in output:
                raise ValueError("selection/amount quote not in candidate")
        check_rows(answer["semantic_checks"], "semantic check")
        if {s["id"] for s in answer["semantic_checks"]} != {s["id"] for s in ref["recommendation"]["semantic_constraints"]}:
            raise ValueError("semantic constraint ID set mismatch")
        for s in answer["semantic_checks"]:
            check_verdict(s)
    else:
        rows = answer["units"]
        check_rows(rows, "phrase unit")
        if {u["id"] for u in rows} != {u["id"] for u in ref["phrase"]["units"]}:
            raise ValueError("phrase unit ID set mismatch")
        for u in rows:
            conditions = (u["target_language"], u["intent_fidelity"], u["selection_consistency"], u["communicative_validity"])
            if not u["quote"] and any(conditions):
                raise ValueError("missing phrase cannot satisfy checks")
            if u["target_language"] and u["language"] not in ref["phrase"]["accepted_languages"]:
                raise ValueError("phrase language is not in frozen accepted set")


def check_verdict(x):
    if x["status"] == "missing" and x["quote"]:
        raise ValueError("missing item cannot cite candidate text")
    if x["status"] != "missing" and not x["quote"]:
        raise ValueError("non-missing item needs quote")
    if x["status"] == "correct" and not x["evidence_ids"]:
        raise ValueError("correct verdict needs evidence")


def summary_score(truth, judge, output):
    claims = judge["claims"]
    covered = {fid for x in claims for fid in x["covered_ids"]}
    count = len(claims)
    valid = sum(x["label"] == "supported_relevant" for x in claims)
    required = len(truth["reference"]["summary"]["required_facts"])
    precision = valid / count if count else 0.0
    recall = len(covered) / required
    length = char_count(output)
    standard = truth["reference_length"]
    ratio = length / standard
    penalty = min(1.0, standard / length) if length else 0.0
    f_score = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {"score": f_score * penalty, "details": {"claims": count, "valid_claims": valid,
        "covered": len(covered), "required": required, "precision": precision, "recall": recall,
        "candidate_characters": length, "reference_characters": standard, "length_ratio": ratio,
        "length_coefficient": penalty, "f_score": f_score}}


def interpretation_score(truth, judge):
    by_id = {v["id"]: v for v in judge["items"]}
    fields = []
    critical = []
    for f in truth["reference"]["interpretation"]["fields"]:
        correct = sum(by_id[x["id"]]["status"] == "correct" for x in f["items"])
        fields.append({"field": f["id"], "correct": correct, "required": len(f["items"]),
                       "score": correct / len(f["items"])})
        critical.extend(x["id"] for x in f["items"] if x["critical"] and by_id[x["id"]]["status"] != "correct")
    required = sum(f["required"] for f in fields)
    extras = sum(x["label"] != "valid_optional" and x["related_item_id"] is None for x in judge["extra_claims"])
    coefficient = required / (required + extras)
    return {"score": mean(f["score"] for f in fields) * coefficient,
            "details": {"fields": fields, "required": required, "extra_invalid": extras,
                        "extra_coefficient": coefficient, "critical_failures": critical}}


def recommendation_score(truth, judge):
    ref = truth["reference"]["recommendation"]
    menu = {x["id"]: x for x in ref["menu"]}
    picks = judge["selections"]
    req = ref["requirements"]
    checks = []

    def add(cid, status, reason, critical=False):
        checks.append({"id": cid, "status": status, "reason": reason, "critical": critical})

    exists = bool(picks) and all(p["id"] in menu for p in picks)
    quantities = bool(picks) and all(type(p["quantity"]) in (float, int) and p["quantity"] > 0
                                    and (type(p["quantity"]) is int or p["quantity"].is_integer()) for p in picks)
    add("items_exist", "passed" if exists else "failed", "全部选择必须能对应菜单菜品")
    add("positive_integer_quantities", "passed" if quantities else "failed", "全部数量必须为正整数")
    curr = {menu[p["id"]]["currency"] for p in picks if p["id"] in menu}
    known = exists and quantities and len(curr) == 1 and all(menu[p["id"]]["price_minor"] is not None for p in picks)
    total = sum(menu[p["id"]]["price_minor"] * int(p["quantity"]) for p in picks) if known else None
    currency = next(iter(curr)) if len(curr) == 1 else None
    if req["budget_minor"] is not None:
        status = "unknown" if not known or currency != req["currency"] else ("passed" if total <= req["budget_minor"] else "failed")
        add("budget", status, f"实际小计={total}，预算={req['budget_minor']}，币种={currency}", True)
    if req["no_spicy"]:
        statuses = [menu[p["id"]]["spicy"] for p in picks if p["id"] in menu]
        status = "failed" if not exists or "yes" in statuses else ("unknown" if "unknown" in statuses else "passed")
        add("no_spicy", status, "只有明确不辣才满足；未知不当作不辣", True)
    for allergen in req["allergies"]:
        selected = [menu[p["id"]] for p in picks if p["id"] in menu]
        if not exists or any(allergen.casefold() in {a.casefold() for a in item["allergens"]} for item in selected):
            status = "failed"
        elif any(not item["ingredients_complete"] for item in selected):
            status = "unknown"
        else:
            status = "passed"
        add("allergen:" + allergen, status, "核对已知成分；配料不完整时不能证明不含过敏原", True)
    # Every claimed amount is checked; at least a total must actually be stated.
    amounts_ok = known and any(a["kind"] == "total" for a in judge["amount_claims"])
    for a in judge["amount_claims"]:
        expected = None
        if a["kind"] == "total":
            expected = total
        elif a["kind"] == "balance" and known and req["budget_minor"] is not None and req["currency"] == currency:
            expected = req["budget_minor"] - total
        elif a["item_id"] in menu:
            item = menu[a["item_id"]]
            expected = item["price_minor"]
            if a["kind"] == "item_subtotal":
                qty = next((p["quantity"] for p in picks if p["id"] == a["item_id"]), None)
                expected = expected * qty if expected is not None and qty is not None else None
        amounts_ok = amounts_ok and expected is not None and a["value_minor"] == expected and a["currency"] == currency
    add("arithmetic_consistency", "passed" if amounts_ok else ("unknown" if not known else "failed"),
        "必须给出可核算总价；所有金额断言与菜单、数量、币种一致")
    semantic = {s["id"]: s for s in judge["semantic_checks"]}
    for c in ref["semantic_constraints"]:
        s = semantic[c["id"]]
        add("user:" + c["id"], "passed" if s["status"] == "correct" else "failed", s["reason"], c["critical"])
    passed = sum(c["status"] == "passed" for c in checks)
    return {"score": passed / len(checks), "details": {"constraints": checks, "passed": passed,
        "total_constraints": len(checks), "all_passed": passed == len(checks), "computed_total_minor": total,
        "computed_currency": currency, "critical_failures": [c["id"] for c in checks if c["critical"] and c["status"] != "passed"]}}


def phrase_score(judge):
    names = ("target_language", "intent_fidelity", "selection_consistency", "communicative_validity")
    units = [{"id": u["id"], "success": all(u[k] for k in names),
              "failed_checks": [k for k in names if not u[k]]} for u in judge["units"]]
    return {"score": sum(u["success"] for u in units) / len(units), "details": {"units": units}}


def calculate(truth, task, judge, output):
    validate_judgement(judge, truth, task, output)
    if task == "summary":
        return summary_score(truth, judge, output)
    if task == "interpretation":
        return interpretation_score(truth, judge)
    if task == "recommendation":
        return recommendation_score(truth, judge)
    return phrase_score(judge)
