"""Versioned wire contracts. Python 3.10+, no third-party runtime dependencies."""
from __future__ import annotations

VERSION = "fg-eval/1.1"
TASKS = ("summary", "interpretation", "recommendation", "phrase")
SCENES = ("menu", "medicine", "beauty", "food", "road_sign", "other")


def obj(properties, optional=()):
    return {"type": "object", "properties": properties,
            "required": [k for k in properties if k not in optional], "additionalProperties": False}


def arr(items, minimum=0):
    return {"type": "array", "items": items, "minItems": minimum}


def enum(*values):
    return {"type": "string", "enum": list(values)}


TEXT = {"type": "string"}
NAME = {"type": "string", "minLength": 1}
HASH = {"type": "string", "pattern": "^[a-f0-9]{64}$"}
BOOL = {"type": "boolean"}
INT = {"type": "integer"}
NUMBER = {"type": "number"}
NULL_INT = {"type": ["integer", "null"]}
NULL_TEXT = {"type": ["string", "null"]}
STRINGS = arr(NAME)
PROMPT = obj({"system": TEXT, "user": NAME})
CONTEXT = obj({"locale": NULL_TEXT, "acceptable_languages": STRINGS})
TASK_CONFIG = obj({"prompt": PROMPT})
CONFIG_SCHEMA = obj({"schema_version": enum(VERSION), "context": CONTEXT,
                     "tasks": obj({t: TASK_CONFIG for t in TASKS}, TASKS)})
MANIFEST_SCHEMA = obj({"image_name": NAME, "context": CONTEXT,
                       "tasks": obj({t: TASK_CONFIG for t in TASKS}, TASKS)}, ("context", "tasks"))
PREDICTION_SCHEMA = obj({
    "schema_version": enum(VERSION), "image_name": NAME, "image_sha256": HASH,
    "task": enum(*TASKS), "model_id": NAME, "run_id": NAME,
    "prompt": PROMPT, "context": CONTEXT, "status": enum("ok", "error"),
    "output": TEXT, "error": NULL_TEXT,
    "generation": obj({"elapsed_seconds": {"type": ["number", "null"], "minimum": 0},
                       "input_tokens": dict(NULL_INT, minimum=0), "output_tokens": dict(NULL_INT, minimum=0),
                       "max_output_tokens": dict(NULL_INT, minimum=0), "finish_reason": NULL_TEXT})})
EVIDENCE = obj({"id": NAME, "source": enum("image", "external"), "location": NAME,
                "text": NAME, "url": NULL_TEXT, "version": NULL_TEXT})
FACT = obj({"id": NAME, "statement": NAME, "qualifiers": STRINGS,
            "applicability_reason": NAME, "reasoning": NAME,
            "acceptable_answers": arr(NAME, 1), "failure_conditions": arr(NAME, 1),
            "evidence_ids": arr(NAME, 1), "unknown_expected": BOOL, "critical": BOOL})
CONSTRAINT = obj({"id": NAME, "statement": NAME, "critical": BOOL})
MENU_ITEM = obj({"id": NAME, "name": NAME, "original": NAME, "price_minor": NULL_INT,
                 "currency": NAME, "spicy": enum("yes", "no", "unknown"),
                 "allergens": STRINGS, "ingredients_complete": BOOL, "evidence_ids": arr(NAME, 1)})
TASK_REFERENCE = obj({"applicable": BOOL, "reason": NAME})
REFERENCE_SCHEMA = obj({
    "scene": enum(*SCENES), "readable": BOOL, "evidence": arr(EVIDENCE, 1),
    "applicability": obj({t: TASK_REFERENCE for t in TASKS}),
    "summary": obj({"required_facts": arr(FACT), "allowed_supplement": STRINGS}),
    "interpretation": obj({"fields": arr(obj({"id": enum("meaning", "conditions", "boundaries"),
                                               "items": arr(FACT, 1)}))}),
    "recommendation": obj({"menu": arr(MENU_ITEM), "requirements": obj({
        "budget_minor": NULL_INT, "currency": NULL_TEXT, "no_spicy": BOOL, "allergies": STRINGS}),
        "semantic_constraints": arr(CONSTRAINT)}),
    "phrase": obj({"accepted_languages": STRINGS,
                   "units": arr(obj({"id": NAME, "purpose": NAME, "required_slots": STRINGS,
                                     "reference_template": NAME}))})})
SUMMARY_REFERENCE_SCHEMA = obj({"text": NAME, "covered_ids": STRINGS, "evidence_ids": arr(NAME, 1)})
VERIFY_SCHEMA = obj({"accepted": BOOL, "issues": STRINGS,
                     "summary_valid": arr(BOOL)})
CLAIM = obj({"id": NAME, "text": NAME, "quote": NAME,
             "label": enum("supported_relevant", "supported_irrelevant", "contradicted", "unsupported"),
             "evidence_ids": STRINGS, "covered_ids": STRINGS, "reason": NAME})
ITEM_VERDICT = obj({"id": NAME, "status": enum("correct", "incorrect", "missing"),
                    "quote": TEXT, "evidence_ids": STRINGS, "reason": NAME})
EXTRA = obj({"id": NAME, "text": NAME, "quote": NAME,
             "label": enum("valid_optional", "unsupported", "contradicted", "irrelevant"),
             "related_item_id": NULL_TEXT, "evidence_ids": STRINGS, "reason": NAME})
SELECTION = obj({"id": NAME, "quantity": {"type": ["number", "null"]}, "quote": NAME})
AMOUNT = obj({"kind": enum("total", "balance", "unit_price", "item_subtotal"),
              "item_id": NULL_TEXT, "value_minor": NULL_INT, "currency": NAME, "quote": NAME})
PHRASE_VERDICT = obj({"id": NAME, "quote": TEXT, "language": NULL_TEXT,
    "target_language": BOOL, "intent_fidelity": BOOL, "selection_consistency": BOOL,
    "communicative_validity": BOOL, "reason": NAME})
JUDGE_SCHEMAS = {
    "summary": obj({"answer_kind": enum("normal", "refusal"), "claims": arr(CLAIM), "issues": STRINGS}),
    "interpretation": obj({"items": arr(ITEM_VERDICT), "extra_claims": arr(EXTRA), "issues": STRINGS}),
    "recommendation": obj({"selections": arr(SELECTION), "amount_claims": arr(AMOUNT),
                            "semantic_checks": arr(ITEM_VERDICT), "issues": STRINGS}),
    "phrase": obj({"units": arr(PHRASE_VERDICT), "issues": STRINGS})}


def validate(value, schema, path="$"):
    """Validate our documented JSON-Schema subset, including rejection of extra keys."""
    import math
    import re
    types = schema.get("type", [])
    types = [types] if isinstance(types, str) else types
    checks = {"object": lambda x: isinstance(x, dict), "array": lambda x: isinstance(x, list),
              "string": lambda x: isinstance(x, str), "boolean": lambda x: isinstance(x, bool),
              "integer": lambda x: type(x) is int,
              "number": lambda x: type(x) is int or (type(x) is float and math.isfinite(x)), "null": lambda x: x is None}
    if types and not any(checks[t](value) for t in types):
        raise ValueError(f"{path}: expected {types}, got {type(value).__name__}")
    if "enum" in schema and value not in schema["enum"]:
        raise ValueError(f"{path}: not in {schema['enum']}")
    if type(value) in (int, float):
        if "minimum" in schema and value < schema["minimum"]:
            raise ValueError(f"{path}: below minimum")
        if "maximum" in schema and value > schema["maximum"]:
            raise ValueError(f"{path}: above maximum")
    if isinstance(value, str):
        if len(value.strip()) < schema.get("minLength", 0):
            raise ValueError(f"{path}: empty string")
        if "pattern" in schema and not re.fullmatch(schema["pattern"], value):
            raise ValueError(f"{path}: invalid pattern")
    if isinstance(value, dict):
        required, props = schema.get("required", []), schema.get("properties", {})
        missing = set(required) - value.keys()
        if missing:
            raise ValueError(f"{path}: missing keys {sorted(missing)}")
        if schema.get("additionalProperties") is False and set(value) - props.keys():
            raise ValueError(f"{path}: unknown keys {sorted(set(value)-props.keys())}")
        for k, v in value.items():
            if k in props:
                validate(v, props[k], f"{path}.{k}")
    if isinstance(value, list):
        if len(value) < schema.get("minItems", 0):
            raise ValueError(f"{path}: too few items")
        for i, v in enumerate(value):
            validate(v, schema["items"], f"{path}[{i}]")


def unique_ids(items, label):
    ids = [x["id"] for x in items]
    if len(ids) != len(set(ids)):
        raise ValueError(f"duplicate {label} IDs")
    return set(ids)


def check_reference(ref, contexts):
    validate(ref, REFERENCE_SCHEMA)
    evidence = unique_ids(ref["evidence"], "evidence")
    if not any(e["source"] == "image" for e in ref["evidence"]):
        raise ValueError("reference requires image evidence")
    for e in ref["evidence"]:
        if e["source"] == "external" and (not e["url"] or not e["version"]):
            raise ValueError("external evidence needs URL and version/date")
    if not ref["readable"] and any(x["applicable"] for x in ref["applicability"].values()):
        raise ValueError("unreadable image must not enter primary metrics")
    for task in ("recommendation", "phrase"):
        if ref["scene"] != "menu" and ref["applicability"][task]["applicable"]:
            raise ValueError("only menu images may enter recommendation/phrase metrics")
    facts = list(ref["summary"]["required_facts"])
    fields = ref["interpretation"]["fields"]
    unique_ids(fields, "field")
    for f in fields:
        facts.extend(f["items"])
    unique_ids(facts, "fact (across tasks)")
    for fact in facts:
        if not set(fact["evidence_ids"]) <= evidence:
            raise ValueError("unknown evidence ID in fact")
    if ref["applicability"]["summary"]["applicable"] and not ref["summary"]["required_facts"]:
        raise ValueError("applicable summary needs required facts")
    if ref["applicability"]["interpretation"]["applicable"] and not fields:
        raise ValueError("applicable interpretation needs fields")
    menu = ref["recommendation"]["menu"]
    unique_ids(menu, "menu")
    for item in menu:
        if item["price_minor"] is not None and item["price_minor"] < 0:
            raise ValueError("negative menu price")
        if not set(item["evidence_ids"]) <= evidence:
            raise ValueError("unknown menu evidence")
    if ref["applicability"]["recommendation"]["applicable"] and not menu:
        raise ValueError("recommendation needs menu")
    req = ref["recommendation"]["requirements"]
    if req["budget_minor"] is not None and (req["budget_minor"] < 0 or not req["currency"]):
        raise ValueError("budget needs nonnegative amount and currency")
    if len(req["allergies"]) != len(set(a.casefold() for a in req["allergies"])):
        raise ValueError("duplicate allergy constraints")
    unique_ids(ref["recommendation"]["semantic_constraints"], "semantic constraint")
    phrase = ref["phrase"]
    unique_ids(phrase["units"], "phrase unit")
    if ref["applicability"]["phrase"]["applicable"]:
        if not phrase["units"] or not phrase["accepted_languages"]:
            raise ValueError("applicable phrase needs units and known target languages")
        allowed = contexts["acceptable_languages"]
        if not allowed or not set(phrase["accepted_languages"]) <= set(allowed):
            raise ValueError("target languages must come from shared, explicit context")


def check_prediction(row):
    import re
    validate(row, PREDICTION_SCHEMA)
    if not re.fullmatch(r"[a-f0-9]{64}", row["image_sha256"]):
        raise ValueError("image_sha256 must be lowercase SHA-256")
    name = row["image_name"]
    if any(x in name for x in ("/", "\\")) or name in (".", ".."):
        raise ValueError("image_name is basename including extension, not a path")
    if row["status"] == "error" and (not row["error"] or row["output"]):
        raise ValueError("error prediction needs error text and empty output")
    if row["status"] == "ok" and row["error"] is not None:
        raise ValueError("ok prediction must have error=null")
    for k, v in row["generation"].items():
        if k != "finish_reason" and v is not None and v < 0:
            raise ValueError("negative generation metadata")


def nullable(schema):
    return dict(schema, type=[schema["type"], "null"])


ANY_OBJECT = {"type": "object"}
TRUTH_SCHEMA = obj({"schema_version": enum(VERSION), "image_name": NAME, "image_path": NAME,
    "image_sha256": HASH, "image_data_url": NAME, "provided_evidence": arr(EVIDENCE),
    "tasks": obj({t: TASK_CONFIG for t in TASKS}, TASKS), "context": CONTEXT,
    "status": enum("ok", "error"), "error": NULL_TEXT, "input_signature": NAME,
    "created_at": NAME, "reference": nullable(REFERENCE_SCHEMA),
    "summary_references": arr(SUMMARY_REFERENCE_SCHEMA), "reference_length": {"type": ["number", "null"]},
    "verification": nullable(VERIFY_SCHEMA), "human_verified": BOOL, "calls": arr(ANY_OBJECT),
    "reference_id": {"type": ["string", "null"], "pattern": "^[a-f0-9]{64}$"}})
EVALUATION_SCHEMA = obj({"schema_version": enum(VERSION), "image_name": NAME, "image_sha256": HASH,
    "task": enum(*TASKS), "model_id": NAME, "run_id": NAME, "created_at": NAME, "input_signature": NAME,
    "truth": TRUTH_SCHEMA, "prediction": nullable(PREDICTION_SCHEMA),
    "evaluation_status": enum("scored", "not_applicable", "reference_error", "judge_error", "missing_prediction", "model_error", "pending"),
    "score": {"type": ["number", "null"], "minimum": 0, "maximum": 1}, "details": ANY_OBJECT, "judgement": nullable(ANY_OBJECT),
    "judge_call": nullable(ANY_OBJECT), "error": NULL_TEXT, "warnings": STRINGS,
    "paired_recommendation_output": NULL_TEXT})
KNOWLEDGE_SCHEMA = obj({"image_name": NAME, "evidence": arr(EVIDENCE, 1)})


if __name__ == "__main__":
    import argparse
    import json
    from pathlib import Path
    p = argparse.ArgumentParser(description="Export the versioned JSON Schemas")
    p.add_argument("output_dir", type=Path)
    target = p.parse_args().output_dir
    target.mkdir(parents=True, exist_ok=True)
    for name, schema in {"prediction": PREDICTION_SCHEMA, "truth": TRUTH_SCHEMA,
                         "evaluation": EVALUATION_SCHEMA, "task_config": CONFIG_SCHEMA,
                         "manifest": MANIFEST_SCHEMA, "knowledge": KNOWLEDGE_SCHEMA}.items():
        (target / (name + ".schema.json")).write_text(json.dumps({"$schema": "https://json-schema.org/draft/2020-12/schema", **schema},
            ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
