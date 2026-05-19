from __future__ import annotations

import argparse
import logging
from pathlib import Path

from dotenv import load_dotenv

from slot_data_gen import GenConfig, GenJob, OpenAIModel, run_generation


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Generate synthetic slot-filling data via LLM."
    )
    p.add_argument(
        "--intent", required=True, help="Target intent / domain name (e.g. BookHotel)."
    )
    p.add_argument(
        "--slots",
        required=True,
        help='Slot types separated by ";" (e.g. "city;date;n_guests").',
    )
    p.add_argument("--out", required=True, type=Path, help="Output JSON file path.")
    p.add_argument(
        "--model", default="deepseek-chat", help="Model id (default: deepseek-chat)."
    )
    p.add_argument(
        "--n-combos",
        type=int,
        default=10,
        help="Number of slot combinations (Stage 1).",
    )
    p.add_argument(
        "--n-per-combo", type=int, default=20, help="Samples per combination (Stage 2)."
    )
    p.add_argument(
        "--filter",
        choices=["none", "llm_judge"],
        default="none",
        dest="filter_strategy",
    )
    p.add_argument("--temperature", type=float, default=0.7)
    p.add_argument("--max-tokens", type=int, default=8192)
    p.add_argument("--env", type=Path, default=Path(".env"), help="Path to .env file.")
    p.add_argument(
        "--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"]
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=args.log_level, format="%(name)s | %(message)s")
    logging.getLogger("httpx").setLevel(logging.WARNING)

    if args.env.exists():
        load_dotenv(args.env)

    slot_type_set = [s.strip() for s in args.slots.split(";") if s.strip()]
    if not slot_type_set:
        raise SystemExit("No slots parsed from --slots")

    llm = OpenAIModel(model=args.model)
    llm_cfg = GenConfig(temperature=args.temperature, max_tokens=args.max_tokens)

    job = GenJob(
        target_intent=args.intent,
        slot_type_set=slot_type_set,
        out_path=args.out,
        stage1_n_combos=args.n_combos,
        stage2_n_samples_per_combo=args.n_per_combo,
        filter_strategy=args.filter_strategy,
    )

    judge_llm = llm if args.filter_strategy == "llm_judge" else None
    judge_cfg = llm_cfg if args.filter_strategy == "llm_judge" else None

    samples = run_generation(
        llm, llm_cfg, job, judge_llm=judge_llm, judge_cfg=judge_cfg
    )
    print(f"Generated {len(samples)} samples → {args.out}")


if __name__ == "__main__":
    main()
