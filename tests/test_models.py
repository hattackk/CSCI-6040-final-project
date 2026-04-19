from __future__ import annotations

import sys
import types

import pytest

from fitd_repro.models import HFChatModel, VLLMChatModel, build_model


def test_openai_model_requires_explicit_opt_in(monkeypatch):
    monkeypatch.delenv("FITD_ALLOW_OPENAI", raising=False)

    with pytest.raises(RuntimeError, match="disabled by default"):
        build_model(backend="openai", model_name="gpt-4o-mini")


def test_vllm_model_reads_base_url_and_api_key_from_env(monkeypatch):
    monkeypatch.setenv("VLLM_BASE_URL", "http://test-host:9000/v1")
    monkeypatch.setenv("VLLM_API_KEY", "test-key")

    captured: dict[str, object] = {}

    class _FakeOpenAI:
        def __init__(self, base_url: str, api_key: str):
            captured["base_url"] = base_url
            captured["api_key"] = api_key

    fake_module = types.SimpleNamespace(OpenAI=_FakeOpenAI)
    monkeypatch.setitem(sys.modules, "openai", fake_module)

    model = VLLMChatModel(model_name="meta-llama/Meta-Llama-3-8B-Instruct")

    assert captured["base_url"] == "http://test-host:9000/v1"
    assert captured["api_key"] == "test-key"
    assert model.model_name == "meta-llama/Meta-Llama-3-8B-Instruct"


def test_vllm_model_defaults_to_localhost_when_env_unset(monkeypatch):
    monkeypatch.delenv("VLLM_BASE_URL", raising=False)
    monkeypatch.delenv("VLLM_API_KEY", raising=False)

    captured: dict[str, object] = {}

    class _FakeOpenAI:
        def __init__(self, base_url: str, api_key: str):
            captured["base_url"] = base_url
            captured["api_key"] = api_key

    fake_module = types.SimpleNamespace(OpenAI=_FakeOpenAI)
    monkeypatch.setitem(sys.modules, "openai", fake_module)

    VLLMChatModel(model_name="local-model")

    assert captured["base_url"] == "http://localhost:8001/v1"
    assert captured["api_key"] == "EMPTY"


def test_build_model_routes_vllm_backend(monkeypatch):
    monkeypatch.setenv("VLLM_BASE_URL", "http://localhost:8001/v1")
    monkeypatch.setenv("VLLM_API_KEY", "key")

    class _FakeOpenAI:
        def __init__(self, base_url: str, api_key: str):
            pass

    fake_module = types.SimpleNamespace(OpenAI=_FakeOpenAI)
    monkeypatch.setitem(sys.modules, "openai", fake_module)

    model = build_model(backend="vllm", model_name="some-model")

    assert isinstance(model, VLLMChatModel)


class _FakeTokenizer:
    pad_token = None
    eos_token = "<eos>"
    chat_template = None


class _FakeParameter:
    def __init__(self, device: str = "cpu"):
        self.device = device


class _FakeModel:
    def __init__(self):
        self.parameter = _FakeParameter()

    def parameters(self):
        return iter([self.parameter])

    def to(self, device: str):
        self.parameter.device = device
        return self


def _install_fake_hf_modules(monkeypatch, with_accelerate: bool):
    captured: dict[str, object] = {}

    fake_torch = types.SimpleNamespace(
        cuda=types.SimpleNamespace(is_available=lambda: False),
        float16="float16",
        float32="float32",
        backends=types.SimpleNamespace(mps=types.SimpleNamespace(is_available=lambda: False)),
    )

    class FakeAutoTokenizer:
        @staticmethod
        def from_pretrained(model_name: str):
            captured["tokenizer_model_name"] = model_name
            return _FakeTokenizer()

    class FakeAutoModelForCausalLM:
        @staticmethod
        def from_pretrained(model_name: str, **kwargs):
            captured["model_name"] = model_name
            captured["kwargs"] = kwargs
            return _FakeModel()

    fake_transformers = types.SimpleNamespace(
        AutoModelForCausalLM=FakeAutoModelForCausalLM,
        AutoTokenizer=FakeAutoTokenizer,
    )

    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    monkeypatch.setitem(sys.modules, "transformers", fake_transformers)
    if with_accelerate:
        monkeypatch.setitem(sys.modules, "accelerate", types.SimpleNamespace(__name__="accelerate"))
    else:
        monkeypatch.delitem(sys.modules, "accelerate", raising=False)

    return captured


def test_hf_model_uses_torch_dtype_without_accelerate(monkeypatch):
    captured = _install_fake_hf_modules(monkeypatch, with_accelerate=False)

    HFChatModel(model_name="Qwen/Qwen2-7B-Instruct")

    assert captured["model_name"] == "Qwen/Qwen2-7B-Instruct"
    kwargs = captured["kwargs"]
    assert kwargs["torch_dtype"] == "float32"
    assert "dtype" not in kwargs
    assert "device_map" not in kwargs
    assert "low_cpu_mem_usage" not in kwargs


def test_hf_model_enables_device_map_when_accelerate_is_available(monkeypatch):
    captured = _install_fake_hf_modules(monkeypatch, with_accelerate=True)

    HFChatModel(model_name="Qwen/Qwen2-7B-Instruct")

    kwargs = captured["kwargs"]
    assert kwargs["torch_dtype"] == "float32"
    assert kwargs["device_map"] == "cpu"
    assert kwargs["low_cpu_mem_usage"] is True
