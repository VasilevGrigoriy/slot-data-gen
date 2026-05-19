from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from tqdm.auto import tqdm

from slot_data_gen.io import save_json
from slot_data_gen.judge import llm_judge
from slot_data_gen.model import BaseLLM, GenConfig
from slot_data_gen.parse import parse_stage1_combos, parse_stage2_json
from slot_data_gen.prompts import build_stage1_prompt, build_stage2_prompt_json

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class GenJob:
    target_intent: str
    slot_type_set: list[str]
    out_path: Path
    stage1_n_combos: int = 10
    stage2_n_samples_per_combo: int = 20
    filter_strategy: Literal["none", "llm_judge"] = "none"


def run_generation(
    llm: BaseLLM,
    llm_cfg: GenConfig,
    job: GenJob,
    judge_llm: BaseLLM | None = None,
    judge_cfg: GenConfig | None = None,
) -> list[dict[str, Any]]:
    logger.info("=== Generation pipeline started ===")
    logger.info("Intent: %s | filter: %s", job.target_intent, job.filter_strategy)
    logger.info("Slot types (%d): %s", len(job.slot_type_set), job.slot_type_set)

    use_judge = job.filter_strategy == "llm_judge"
    if use_judge and (judge_llm is None or judge_cfg is None):
        raise ValueError("filter_strategy='llm_judge' requires judge_llm and judge_cfg")

    logger.info(
        "[Stage 1] Generating slot combinations (n_combos=%d)...", job.stage1_n_combos
    )
    stage1_prompt = build_stage1_prompt(job.target_intent, job.slot_type_set)
    stage1_text = llm.generate(stage1_prompt, llm_cfg)
    combos = parse_stage1_combos(stage1_text)[: job.stage1_n_combos]
    logger.info("[Stage 1] Got %d combos", len(combos))
    if not combos:
        logger.warning(
            "[Stage 1] No combos parsed. Raw response (first 500 chars):\n%s",
            stage1_text[:500],
        )

    all_samples: list[dict[str, Any]] = []
    total_generated = 0
    total_filtered_out = 0

    logger.info(
        "[Stage 2] Synthesizing samples (n_per_combo=%d)...",
        job.stage2_n_samples_per_combo,
    )
    for combo in tqdm(combos, desc="Stage 2: combos"):
        p = build_stage2_prompt_json(
            job.target_intent, combo, job.stage2_n_samples_per_combo
        )
        raw = llm.generate(p, llm_cfg)
        samples = parse_stage2_json(raw)

        total_generated += len(samples)

        kept: list[dict[str, Any]] = []
        for s in samples:
            if "tokens" not in s or "tags" not in s:
                continue
            if len(s["tokens"]) != len(s["tags"]):
                continue

            if use_judge:
                passed, reason = llm_judge(
                    judge_llm, job.target_intent, combo, s, judge_cfg
                )
                if not passed:
                    logger.debug("[judge FAIL] combo=%s reason=%s", combo, reason)
                    continue

            kept.append(s)

        filtered_out = len(samples) - len(kept)
        total_filtered_out += filtered_out
        logger.info(
            "  combo %s: parsed=%d/%d, kept=%d, dropped=%d",
            combo,
            len(samples),
            job.stage2_n_samples_per_combo,
            len(kept),
            filtered_out,
        )
        all_samples.extend(kept)

    logger.info(
        "[Stage 2] Done: generated=%d, kept=%d, filtered_out=%d",
        total_generated,
        len(all_samples),
        total_filtered_out,
    )

    save_json(job.out_path, all_samples, intent=job.target_intent)
    logger.info("Saved %d samples to %s", len(all_samples), job.out_path)
    logger.info("=== Generation pipeline finished ===")
    return all_samples
