# slot-data-gen

Synthetic data generation for unseen slot-filling tasks. 

You bring two things — an ***intent name*** (string) and a ***list of slot types*** — and the tool generates BIO-annotated training data using any OpenAI-compatible API.

## How it works

Two-stage pipeline (after [Li et al., ACL 2025](https://aclanthology.org/2025.findings-acl.1097/)):

1. **Stage 1** — the LLM proposes N realistic combinations of your slot types.
2. **Stage 2** — for each combination, the LLM synthesizes M user utterances and BIO tags.
3. *(Optional)* **LLM-as-a-judge** filter validates each sample on 4 criteria: plausibility, span correctness, tag cleanliness, missing-tag check.

Output: pretty-printed JSON array.

## Install

Requires Python ≥ 3.10 and [uv](https://docs.astral.sh/uv/). To install dependencies from project root run this command:

```bash
uv sync
cp .env.example .env
```

## Quickstart

### CLI

```bash
uv run python generate.py \
    --intent BookHotel \
    --slots "city;check_in_date;n_guests;room_type" \
    --n-combos 10 \
    --n-per-combo 20 \
    --out ./out/book_hotel.json \
    --model deepseek-chat \
    --filter llm_judge
```

### Python

See [`examples/demo.ipynb`](examples/demo.ipynb)

## Output format

JSON array, one record per sample:
```json
[
  {
    "intent": "BookHotel",
    "query": "book a room in Berlin for 3 people",
    "tokens": ["book", "a", "room", "in", "Berlin", "for", "3", "people"],
    "tags": ["O", "O", "O", "O", "B-city", "O", "B-n_guests", "O"]
  }
]
```
