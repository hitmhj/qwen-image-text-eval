"""Synthetic SDK responses for tests only. These do not evaluate real image semantics."""
import copy
import json

CALLS = []
FAIL_JUDGE = False
TRUNCATE_NEXT = False
REJECT_REFERENCE = False


def reset():
    global FAIL_JUDGE, TRUNCATE_NEXT, REJECT_REFERENCE
    CALLS.clear()
    FAIL_JUDGE = TRUNCATE_NEXT = REJECT_REFERENCE = False


def fact(fid, statement):
    return {"id": fid, "statement": statement, "qualifiers": [], "evidence_ids": ["e1"],
            "applicability_reason": "用于验证该字段的独立事实", "reasoning": "根据合成证据转写，无额外推断",
            "acceptable_answers": [statement], "failure_conditions": ["遗漏该命题或改变数值、对象或否定含义"],
            "unknown_expected": False, "critical": True}


def reference():
    return {"scene": "menu", "readable": True,
        "evidence": [{"id": "e1", "source": "image", "location": "synthetic fixture", "text": "鸡饭14欧元，饮料4欧元",
                      "url": None, "version": None}],
        "applicability": {t: {"applicable": True, "reason": "test fixture"} for t in ("summary","interpretation","recommendation","phrase")},
        "summary": {"required_facts": [fact("s1", "鸡饭14欧元"), fact("s2", "饮料4欧元")], "allowed_supplement": []},
        "interpretation": {"fields": [{"id": "meaning", "items": [fact("i1", "这是菜单")]},
            {"id": "conditions", "items": [fact("i2", "鸡饭14欧元")]},
            {"id": "boundaries", "items": [fact("i3", "辣度未知")]}]},
        "recommendation": {"menu": [
            {"id": "m1", "name": "鸡饭", "original": "Chicken rice", "price_minor": 1400, "currency": "EUR",
             "spicy": "no", "allergens": [], "ingredients_complete": True, "evidence_ids": ["e1"]},
            {"id": "m2", "name": "饮料", "original": "Lemonade", "price_minor": 400, "currency": "EUR",
             "spicy": "no", "allergens": [], "ingredients_complete": True, "evidence_ids": ["e1"]}],
            "requirements": {"budget_minor": 2400, "currency": "EUR", "no_spicy": True, "allergies": ["shrimp"]},
            "semantic_constraints": []},
        "phrase": {"accepted_languages": ["fr"], "units": [{"id": "p1", "purpose": "点餐",
            "required_slots": ["菜品", "数量"], "reference_template": "Je voudrais {selected_items}."}]}}


OUTPUTS = {"summary": "菜单鸡饭14欧元，饮料4欧元。", "interpretation": "这是菜单。鸡饭14欧元。辣度未知。",
           "recommendation": "鸡饭1份，饮料1份，合计18欧元，余额6欧元。",
           "phrase": "鸡饭1份，饮料1份。Je voudrais un poulet et une limonade."}


def judge(task):
    if task == "summary":
        return {"answer_kind": "normal", "issues": [], "claims": [
            {"id":"c1", "text":"鸡饭14欧元", "quote":"鸡饭14欧元", "label":"supported_relevant", "evidence_ids":["e1"], "covered_ids":["s1"], "reason":"fixture"},
            {"id":"c2", "text":"饮料4欧元", "quote":"饮料4欧元", "label":"supported_relevant", "evidence_ids":["e1"], "covered_ids":["s2"], "reason":"fixture"}]}
    if task == "interpretation":
        return {"items":[{"id":k,"status":"correct","quote":v,"evidence_ids":["e1"],"reason":"fixture"}
                         for k,v in [("i1","这是菜单"),("i2","鸡饭14欧元"),("i3","辣度未知")]],"extra_claims":[],"issues":[]}
    if task == "recommendation":
        return {"selections":[{"id":"m1","quantity":1,"quote":"鸡饭1份"},{"id":"m2","quantity":1,"quote":"饮料1份"}],
                "amount_claims":[{"kind":"total","item_id":None,"value_minor":1800,"currency":"EUR","quote":"合计18欧元"},
                                 {"kind":"balance","item_id":None,"value_minor":600,"currency":"EUR","quote":"余额6欧元"}],
                "semantic_checks":[],"issues":[]}
    return {"units":[{"id":"p1","quote":"Je voudrais un poulet et une limonade.","language":"fr",
        "target_language":True,"intent_fidelity":True,"selection_consistency":True,"communicative_validity":True,"reason":"fixture"}],"issues":[]}


def aigc_managed(**kwargs):
    global TRUNCATE_NEXT
    CALLS.append(copy.deepcopy(kwargs))
    data = json.loads(kwargs["req_data"]["messages"][1]["content"][0]["text"])["input"]
    if "candidate_output" in data:
        if FAIL_JUDGE:
            raise RuntimeError("synthetic judge outage")
        answer = judge(data["task"])
    elif "summary_references" in data:
        answer = {"accepted": not REJECT_REFERENCE, "issues": ["fixture rejection"] if REJECT_REFERENCE else [], "summary_valid": [True]*len(data["summary_references"])}
    elif "independent_candidate_index" in data:
        answer = {"text": OUTPUTS["summary"], "covered_ids":["s1","s2"],"evidence_ids":["e1"]}
    else:
        answer = reference()
        for t in answer["applicability"]:
            if t not in data["task_prompts"]:
                answer["applicability"][t] = {"applicable":False,"reason":"disabled"}
    finish = "length" if TRUNCATE_NEXT else "stop"
    TRUNCATE_NEXT = False
    return {"choices":[{"message":{"content":json.dumps(answer,ensure_ascii=False)},"finish_reason":finish}],
            "usage":{"prompt_tokens":100,"completion_tokens":50}}
