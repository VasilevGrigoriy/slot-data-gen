from __future__ import annotations

import os
import random
import time
from dataclasses import dataclass

from openai import OpenAI


@dataclass(frozen=True)
class GenConfig:
    temperature: float = 0.0
    top_p: float | None = None
    top_k: int | None = None
    max_tokens: int | None = None


class BaseLLM:
    def generate(self, prompt: str, cfg: GenConfig) -> str:
        raise NotImplementedError


class OpenAIModel(BaseLLM):
    def __init__(
        self,
        model: str,
        api_key: str | None = None,
        base_url: str | None = None,
        api: str = "chat",
        system_prompt: str | None = None,
        timeout: float = 60.0,
        max_retries: int = 6,
    ):
        self.model = model
        self.api = api
        self.system_prompt = system_prompt
        self.timeout = timeout
        self.max_retries = max_retries

        api_key = api_key or os.environ.get("OPENAI_API_KEY")
        base_url = base_url or os.environ.get("OPENAI_BASE_URL")

        kwargs: dict[str, object] = {}
        if api_key:
            kwargs["api_key"] = api_key
        if base_url:
            kwargs["base_url"] = base_url

        self.client = OpenAI(**kwargs)

    def generate(self, prompt: str, cfg: GenConfig) -> str:
        last_err: Exception | None = None

        for attempt in range(self.max_retries + 1):
            try:
                if self.api == "responses":
                    resp = self.client.responses.create(
                        model=self.model,
                        input=prompt,
                        temperature=cfg.temperature,
                        max_output_tokens=cfg.max_tokens,
                        top_p=cfg.top_p,
                    )
                    text = getattr(resp, "output_text", None)
                    if isinstance(text, str):
                        return text
                    return str(resp)

                if self.api == "chat":
                    messages: list[dict[str, str]] = []
                    if self.system_prompt:
                        messages.append(
                            {"role": "system", "content": self.system_prompt}
                        )
                    messages.append({"role": "user", "content": prompt})

                    resp = self.client.chat.completions.create(
                        model=self.model,
                        messages=messages,
                        temperature=cfg.temperature,
                        top_p=cfg.top_p,
                        max_tokens=cfg.max_tokens,
                    )
                    return resp.choices[0].message.content or ""

                raise ValueError(
                    f"Unknown api='{self.api}', expected 'chat' or 'responses'"
                )

            except Exception as e:
                last_err = e
                if attempt >= self.max_retries:
                    break
                sleep_s = min(30.0, (2.0**attempt) + random.random())
                time.sleep(sleep_s)

        assert last_err is not None
        raise last_err
