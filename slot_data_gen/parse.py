from __future__ import annotations

import json
import re
from typing import Any

PUNCT = set(".,;:!?\"'()[]{}…–—-/")


def strip_punct_tokens(sample: dict) -> dict:
    if "tokens" not in sample or "tags" not in sample:
        return sample
    tokens = sample["tokens"]
    tags = sample["tags"]
    cleaned_tokens: list[str] = []
    cleaned_tags: list[str] = []
    for tok, tag in zip(tokens, tags):
        if all(ch in PUNCT for ch in tok):
            continue
        cleaned_tokens.append(tok)
        cleaned_tags.append(tag)
    sample["tokens"] = cleaned_tokens
    sample["tags"] = cleaned_tags
    if "query" in sample:
        sample["query"] = " ".join(cleaned_tokens)
    return sample


_COMBO_PATTERNS = [
    re.compile(r"\(slot\s*types?\s*selected\)\s*:\s*(.+)$", re.IGNORECASE),
    re.compile(r"\*{0,2}slot\s*types?\s*selected\*{0,2}\s*:\s*(.+)$", re.IGNORECASE),
    re.compile(r"^\d+[\.\)]\s*(.+)$"),
]


def parse_stage1_combos(text: str) -> list[list[str]]:
    combos: list[list[str]] = []
    for line in text.splitlines():
        stripped = line.strip()
        raw: str | None = None
        for pat in _COMBO_PATTERNS:
            m = pat.search(stripped)
            if m:
                raw = m.group(1).strip()
                break
        if raw is None:
            continue
        raw = raw.replace("*", "")
        parts = [p.strip() for p in raw.split(";") if p.strip()]
        slot_like = [p for p in parts if "_" in p or p.replace("_", "").isalpha()]
        if slot_like:
            combos.append(slot_like)

    uniq: list[list[str]] = []
    seen: set[tuple[str, ...]] = set()
    for c in combos:
        key = tuple(sorted(c))
        if key not in seen:
            seen.add(key)
            uniq.append(c)
    return uniq


def _extract_json(text: str) -> str:
    text = text.strip()
    m = re.search(r"```(?:json)?\s*\n?(.*?)```", text, re.DOTALL)
    if m:
        return m.group(1).strip()
    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1 and end > start:
        return text[start : end + 1]
    return text


def _repair_truncated_json(text: str) -> str:
    brackets: list[str] = []
    in_string = False
    escape = False
    for ch in text:
        if escape:
            escape = False
            continue
        if ch == "\\":
            escape = True
            continue
        if ch == '"':
            in_string = not in_string
            continue
        if in_string:
            continue
        if ch in ("{", "["):
            brackets.append(ch)
        elif ch == "}" and brackets and brackets[-1] == "{":
            brackets.pop()
        elif ch == "]" and brackets and brackets[-1] == "[":
            brackets.pop()

    closing = {"[": "]", "{": "}"}
    suffix = "".join(closing[b] for b in reversed(brackets))
    return text + suffix


def parse_stage2_json(text: str) -> list[dict]:
    cleaned = _extract_json(text)
    try:
        obj = json.loads(cleaned)
    except json.JSONDecodeError:
        cleaned = _repair_truncated_json(cleaned)
        obj = json.loads(cleaned)
    samples = obj.get("samples", [])
    if not isinstance(samples, list):
        return []
    return [strip_punct_tokens(s) for s in samples]


def _parse_kv_line(line: str) -> dict[str, str]:
    out: dict[str, str] = {}
    chunks = [c.strip() for c in line.split(";") if c.strip()]
    for ch in chunks:
        if "</res>" not in ch:
            continue
        t, v = ch.split("</res>", 1)
        out[t.strip()] = v.strip()
    return out


def parse_stage2_freeform(text: str) -> list[dict]:
    samples: list[dict] = []
    cur: dict = {}

    def flush():
        nonlocal cur
        if not cur:
            return
        if "tokens" in cur and "tags" in cur and len(cur["tokens"]) == len(cur["tags"]):
            samples.append(cur)
        cur = {}

    for line in text.splitlines():
        s = line.strip()
        if s.startswith("(type</res>value):"):
            flush()
            cur = {}
            cur["slot_values_map"] = _parse_kv_line(s.split(":", 1)[1].strip())
        elif s.startswith("(query):"):
            cur["query"] = s.split(":", 1)[1].strip()
        elif s.startswith("(tokens):"):
            cur["tokens"] = s.split(":", 1)[1].strip().split()
        elif s.startswith("(tags):"):
            cur["tags"] = s.split(":", 1)[1].strip().split()

    flush()
    return [strip_punct_tokens(s) for s in samples]


def bio_to_spans(tokens: list[str], tags: list[str]) -> list[dict[str, Any]]:
    spans: list[dict[str, Any]] = []
    cur_type: str | None = None
    start: int | None = None

    for i, tag in enumerate(tags):
        if tag == "O":
            if cur_type is not None:
                value = " ".join(tokens[start:i])
                spans.append(
                    {"type": cur_type, "start": start, "end": i, "value": value}
                )
                cur_type, start = None, None
            continue

        prefix, stype = tag.split("-", 1)

        if prefix == "B":
            if cur_type is not None:
                value = " ".join(tokens[start:i])
                spans.append(
                    {"type": cur_type, "start": start, "end": i, "value": value}
                )
            cur_type = stype
            start = i
        elif cur_type is None or cur_type != stype:
            if cur_type is not None:
                value = " ".join(tokens[start:i])
                spans.append(
                    {"type": cur_type, "start": start, "end": i, "value": value}
                )
            cur_type = stype
            start = i

    if cur_type is not None:
        end = len(tags)
        value = " ".join(tokens[start:end])
        spans.append({"type": cur_type, "start": start, "end": end, "value": value})
    return spans
