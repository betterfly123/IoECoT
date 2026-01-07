# llm_clients.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional
import time
import os

class BaseChatClient:
    def chat(self, prompt: str, temperature: float = 0.0, max_tokens: Optional[int] = None) -> str:
        raise NotImplementedError


@dataclass
class OpenAIChatClient(BaseChatClient):
    model: str
    api_key: Optional[str] = None
    base_url: Optional[str] = None
    max_retries: int = 2
    retry_sleep: float = 2.0

    def __post_init__(self):

        from openai import OpenAI

        api_key = self.api_key or os.getenv("OPENAI_API_KEY", "")
        base_url = self.base_url or os.getenv("OPENAI_BASE_URL", None)
        self._client = OpenAI(api_key=api_key, base_url=base_url)

    def chat(self, prompt: str, temperature: float = 0.0, max_tokens: Optional[int] = None) -> str:
        last_err = None
        for _ in range(self.max_retries + 1):
            try:
                resp = self._client.chat.completions.create(
                    model=self.model,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=temperature,
                    max_tokens=max_tokens,
                )
                return resp.choices[0].message.content or ""
            except Exception as e:
                last_err = e
                time.sleep(self.retry_sleep)
        raise RuntimeError(f"OpenAI call failed: {last_err}")


class ChatGLMClient(BaseChatClient):


    def __init__(self, model_path: str, device: Optional[str] = None):
        import torch
        from transformers import AutoTokenizer, AutoModel

        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"

        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        self.model = AutoModel.from_pretrained(model_path, trust_remote_code=True).to(device).eval()

    def chat(self, prompt: str, temperature: float = 0.0, max_tokens: Optional[int] = None) -> str:

        kwargs = {"history": [], "temperature": temperature}
        if max_tokens is not None:
            kwargs["max_length"] = max_tokens
        out, _ = self.model.chat(self.tokenizer, prompt, **kwargs)
        return out or ""
