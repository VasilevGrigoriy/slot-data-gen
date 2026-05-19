from __future__ import annotations

import json
import logging
import re
from typing import Any

from slot_data_gen.model import BaseLLM, GenConfig
from slot_data_gen.prompts import build_judge_prompt

logger = logging.getLogger(__name__)

_JSON_RE = re.compile(r"\{.*\}", re.DOTALL)
_CHECKS = ("plausible", "spans_ok", "tags_clean", "no_missing")


def _parse_judge_output(text: str) -> dict | None:
    text = text.strip()
    text = re.sub(r"^```(?:json)?\s*", "", text)
    text = re.sub(r"\s*```$", "", text)
    m = _JSON_RE.search(text)
    if not m:
        return None
    try:
        return json.loads(m.group(0))
    except json.JSONDecodeError:
        return None


def llm_judge(
    llm: BaseLLM,
    domain: str,
    combo: list[str],
    sample: dict[str, Any],
    cfg: GenConfig,
) -> tuple[bool, str]:
    try:
        raw = llm.generate(build_judge_prompt(domain, combo, sample), cfg).strip()
    except Exception as e:
        return False, f"llm_call_failed: {e}"

    parsed = _parse_judge_output(raw)
    if parsed is None:
        return False, f"unparseable: {raw[:120]!r}"

    failed = [c for c in _CHECKS if not bool(parsed.get(c, False))]
    if not failed:
        return True, ""
    model_reason = str(parsed.get("reason", "")).strip()
    return False, f"failed={failed}; {model_reason}"
