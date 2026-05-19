from __future__ import annotations

import json
from typing import Any

from slot_data_gen.parse import bio_to_spans


def build_stage1_prompt(domain: str, slot_type_set: list[str]) -> str:
    slot_list = "; ".join(slot_type_set)
    return f"""Stage 1: Slot Combinations Selection
Task Description:
I would like you to help me select some slot combinations from a set of slot types.
Conditions:
1. The values corresponding to the slots in the combination can appear in the same user query.
2. The user query only includes the selected slots, avoid including other slots.
3. Provide the reasoning process along with the slot combinations.

Input:
Now, given a new domain: {domain}. and given a new list of slot types: {slot_list}.
Please select 10 slot combinations and their corresponding reasoning processes.

Output format (use EXACTLY these two lines, repeat 10 times; no markdown, no bold, no numbering, no bullets, no headers, no separators):
(slot types selected): a; b; c
(inference): "...example query that only includes selected slots..."
"""


def build_stage2_prompt_freeform(
    domain: str, slot_combo: list[str], n_samples: int
) -> str:
    combo = ";".join(slot_combo)
    return f"""Stage 2: Data Synthesis
Task Description:
First, given a selected set of slot types and a domain, you need to provide a corresponding slot value
for each slot type. Then, using these related slot values, generate a user query with a given domain.
Finally, annotate the query using the IOB2 format (BIO tags) aligned to whitespace tokenization.

Conditions:
1. The user query cannot contain words corresponding to slots other than the selected slot types.
2. The sentence can only contain slot combinations of slot values. (no extra slot values)
3. Generate samples as diverse as possible.

Input:
(domain): {domain}
(slot type combination): {combo}

Please generate {n_samples} samples.

Output format (repeat {n_samples} times):
(type</res>value): slot1</res>value1;slot2</res>value2;...
(query): ...
(tokens): token1 token2 ...
(tags): O B-slot1 I-slot1 ...
"""


def build_stage2_prompt_json(domain: str, slot_combo: list[str], n_samples: int) -> str:
    schema = {
        "domain": domain,
        "slot_type_combination": slot_combo,
        "samples": [
            {
                "slot_values": [{"type": "slot_type", "value": "string"}],
                "query": "string",
                "tokens": ["string"],
                "tags": ["O|B-...|I-..."],
            }
        ],
    }

    return f"""You are generating synthetic data for a slot filling dataset.

Constraints:
- Only use slot types from the given slot_type_combination.
- The query MUST NOT contain any other slot value mentions for slot types not in the combination.
- Tokenization: whitespace split.
- Tags: BIO (IOB2). Length of tokens == length of tags.

Return ONLY valid JSON (no markdown). Must match this schema:
{json.dumps(schema, ensure_ascii=False)}

Input:
domain = {domain}
slot_type_combination = {json.dumps(slot_combo, ensure_ascii=False)}
n_samples = {n_samples}
"""


def build_judge_prompt(domain: str, combo: list[str], sample: dict[str, Any]) -> str:
    tokens = sample.get("tokens", [])
    tags = sample.get("tags", [])
    query = sample.get("query") or " ".join(tokens)
    spans_str = _format_spans(tokens, tags)
    allowed = ", ".join(combo) if combo else "(none)"

    return f"""You are validating ONE synthetic slot-filling example for the domain "{domain}".

Allowed slot types for this example: [{allowed}]

Query: {query}

Annotated spans:
{spans_str}

Evaluate FOUR independent criteria:

A) plausible — Is the query a realistic, grammatically sensible utterance that a real user
   could say in the "{domain}" domain? Reject nonsense (e.g. repeated tokens like "artist artist artist"),
   broken syntax, off-domain content, or tokens that look randomly assembled.

B) spans_ok — For EACH annotated span, does the slot value truly instantiate the declared
   slot type? Examples of failures:
   * "artist": "jazz"           -> FAIL (jazz is a genre, not an artist name)
   * "city": "tomorrow"         -> FAIL (tomorrow is time, not a city)
   * "restaurant_name": "food"  -> FAIL (generic word, not a name)
   The slot value should be a plausible real-world instance of its slot type.

C) tags_clean — Are ALL tagged slot types contained in the allowed list above?
   Both B-X and I-X are valid for slot type X.

D) no_missing — Are there NO obvious unannotated slot mentions in the query that should have
   been tagged with a slot type from the allowed list?

Return ONLY a single-line JSON object (no markdown, no commentary):
{{"plausible": <true|false>, "spans_ok": <true|false>, "tags_clean": <true|false>, "no_missing": <true|false>, "reason": "<short reason if any check fails, else empty>"}}
"""


def _format_spans(tokens: list[str], tags: list[str]) -> str:
    spans = bio_to_spans(tokens, tags)
    if not spans:
        return "  (no slot annotations)"
    return "\n".join(f'  - {s["type"]}: "{s["value"]}"' for s in spans)
