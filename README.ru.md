# slot-data-gen

*[English version](README.md)*

Генерация синтетических данных для новых задач slot-filling (заполнения слотов).

Вы передаёте две вещи - ***название интента*** (строка) и ***список типов слотов*** - а инструмент генерирует размеченные в формате BIO обучающие данные через любой OpenAI-совместимый API.

## Как это работает

Двухэтапный пайплайн (по мотивам [Li et al., ACL 2025](https://aclanthology.org/2025.findings-acl.1097/)):

1. **Этап 1** - LLM предлагает N реалистичных комбинаций ваших типов слотов.
2. **Этап 2** - для каждой комбинации LLM синтезирует M пользовательских реплик и BIO-разметку.
3. *(Опционально)* **LLM-as-a-judge** фильтр проверяет каждый пример по 4 критериям: правдоподобность, корректность границ спанов, чистота тегов, проверка пропущенных тегов.

Результат: JSON-массив с форматированным выводом.

## Установка

Требуется Python ≥ 3.10 и [uv](https://docs.astral.sh/uv/). Чтобы установить зависимости, из корня проекта выполните:

```bash
uv sync
cp .env.example .env
```

## Быстрый старт

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

См. [`examples/demo.ipynb`](examples/demo.ipynb)

## Формат вывода

JSON-массив, по одной записи на каждый пример:
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
