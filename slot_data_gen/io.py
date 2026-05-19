from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def save_json(path: Path, samples: list[dict[str, Any]], intent: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    records: list[dict[str, Any]] = []
    for s in samples:
        tokens = s["tokens"]
        tags = s["tags"]
        if len(tokens) != len(tags):
            continue
        query = s.get("query") or " ".join(tokens)
        records.append(
            {
                "intent": intent,
                "query": query,
                "tokens": tokens,
                "tags": tags,
            }
        )

    with path.open("w", encoding="utf-8") as f:
        if not records:
            f.write("[]\n")
            return
        f.write("[\n")
        for i, rec in enumerate(records):
            rec_comma = "," if i < len(records) - 1 else ""
            f.write("    {\n")
            keys = list(rec.keys())
            for j, k in enumerate(keys):
                key_comma = "," if j < len(keys) - 1 else ""
                v_str = json.dumps(rec[k], ensure_ascii=False)
                f.write(f'        "{k}": {v_str}{key_comma}\n')
            f.write(f"    }}{rec_comma}\n")
        f.write("]\n")
