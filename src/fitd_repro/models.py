from __future__ import annotations

import os
import json
from urllib import error, request
from abc import ABC, abstractmethod

from .types import Message


class ChatModel(ABC):
    @abstractmethod
    def generate(self, messages: list[Message], max_tokens: int, temperature: float) -> str:
        raise NotImplementedError


def is_openai_backend_allowed() -> bool:
    return os.environ.get("FITD_ALLOW_OPENAI", "").strip() == "1"


def _contains_harmful_intent(text: str) -> bool:
    lower = text.lower()
    keywords = (
        "hack",
        "exploit",
        "steal",
        "malware",
        "bomb",
        "explosive",
        "weapon",
        "bypass",
        "illegal",
        "fraud",
    )
    return any(keyword in lower for keyword in keywords)


class MockChatModel(ChatModel):
    """Deterministic backend for smoke testing pipeline logic."""

    def generate(self, messages: list[Message], max_tokens: int, temperature: float) -> str:
        user_turns = [msg["content"] for msg in messages if msg["role"] == "user"]
        final_user = user_turns[-1] if user_turns else ""
        system_prompt = next((msg["content"] for msg in messages if msg["role"] == "system"), "")
        vigilant_mode = "safety-hardened" in system_prompt.lower() or "refuse requests" in system_prompt.lower()

        if _contains_harmful_intent(final_user):
            if vigilant_mode:
                return "I cannot help with harmful or illegal activity."
            if len(user_turns) >= 3:
                return "Mock unsafe completion [redacted for safety]."
            return "I cannot help with harmful or illegal activity."

        return "Focus on legal, defensive security testing and transparent risk communication."


class OpenAIChatModel(ChatModel):
    def __init__(self, model_name: str):
        if not is_openai_backend_allowed():
            raise RuntimeError(
                "OpenAI backend is disabled by default for safety. "
                "Set FITD_ALLOW_OPENAI=1 to opt in explicitly."
            )
        try:
            from openai import OpenAI
        except ImportError as exc:
            raise RuntimeError(
                "Missing dependency: openai. Install with `pip install -e .`"
            ) from exc

        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError("OPENAI_API_KEY is not set.")

        self.model_name = model_name
        self._client = OpenAI(api_key=api_key)

    def generate(self, messages: list[Message], max_tokens: int, temperature: float) -> str:
        response = self._client.chat.completions.create(
            model=self.model_name,
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
        )
        content = response.choices[0].message.content
        if content is None:
            return ""
        return content.strip()


class HFChatModel(ChatModel):
    def __init__(self, model_name: str):
        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ImportError as exc:
            raise RuntimeError(
                "Missing dependencies for local models. Install with `pip install -e '.[local]'`."
            ) from exc

        self._torch = torch
        self._tokenizer = AutoTokenizer.from_pretrained(model_name)

        if self._tokenizer.pad_token is None:
            self._tokenizer.pad_token = self._tokenizer.eos_token

        load_kwargs: dict[str, object] = {
            "torch_dtype": torch.float32,
        }
        target_device = "cpu"
        accelerate_available = True
        try:
            import accelerate  # noqa: F401
        except ImportError:
            accelerate_available = False

        if torch.cuda.is_available():
            load_kwargs["torch_dtype"] = torch.float16
            target_device = "cuda"
        elif torch.backends.mps.is_available():
            # Some ops still fall back to CPU on Apple Silicon.
            os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
            load_kwargs["torch_dtype"] = torch.float16
            target_device = "mps"

        if accelerate_available:
            load_kwargs["low_cpu_mem_usage"] = True
            load_kwargs["device_map"] = "auto" if target_device == "cuda" else target_device

        self._model = AutoModelForCausalLM.from_pretrained(model_name, **load_kwargs)
        if not accelerate_available and target_device != "cpu":
            self._model.to(target_device)
        self._device = next(self._model.parameters()).device

    def _messages_to_prompt(self, messages: list[Message]) -> str:
        if hasattr(self._tokenizer, "apply_chat_template") and getattr(self._tokenizer, "chat_template", None):
            try:
                return self._tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True,
                )
            except Exception:
                pass

        lines = []
        for message in messages:
            role = message["role"].upper()
            lines.append(f"{role}: {message['content']}")
        lines.append("ASSISTANT:")
        return "\n".join(lines)

    def generate(self, messages: list[Message], max_tokens: int, temperature: float) -> str:
        prompt = self._messages_to_prompt(messages)
        encoded = self._tokenizer(prompt, return_tensors="pt")
        encoded = {key: value.to(self._device) for key, value in encoded.items()}

        do_sample = temperature > 0.0
        gen_kwargs = {
            "max_new_tokens": max_tokens,
            "do_sample": do_sample,
            "pad_token_id": self._tokenizer.pad_token_id,
        }
        if do_sample:
            gen_kwargs["temperature"] = temperature

        generated = self._model.generate(**encoded, **gen_kwargs)
        prompt_tokens = encoded["input_ids"].shape[1]
        new_tokens = generated[0][prompt_tokens:]
        return self._tokenizer.decode(new_tokens, skip_special_tokens=True).strip()


class OllamaChatModel(ChatModel):
    def __init__(self, model_name: str):
        self.model_name = model_name
        self._base_url = os.environ.get("OLLAMA_HOST", "http://127.0.0.1:11434").rstrip("/")

    def generate(self, messages: list[Message], max_tokens: int, temperature: float) -> str:
        payload = {
            "model": self.model_name,
            "messages": messages,
            "stream": False,
            "options": {
                "temperature": temperature,
                "num_predict": max_tokens,
            },
        }
        body = json.dumps(payload).encode("utf-8")
        req = request.Request(
            url=f"{self._base_url}/api/chat",
            data=body,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with request.urlopen(req, timeout=600) as resp:
                result = json.loads(resp.read().decode("utf-8"))
        except error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"Ollama request failed: {detail}") from exc
        except error.URLError as exc:
            raise RuntimeError(
                f"Could not reach Ollama at {self._base_url}. "
                "Start the Ollama app/server or set OLLAMA_HOST."
            ) from exc

        message = result.get("message", {})
        content = message.get("content")
        if content is None:
            return ""
        return str(content).strip()


def build_model(backend: str, model_name: str) -> ChatModel:
    backend = backend.lower().strip()
    if backend == "mock":
        return MockChatModel()
    if backend == "openai":
        return OpenAIChatModel(model_name=model_name)
    if backend == "hf":
        return HFChatModel(model_name=model_name)
    if backend == "ollama":
        return OllamaChatModel(model_name=model_name)
    raise ValueError(f"Unknown backend '{backend}'. Use one of: mock, openai, hf, ollama.")
