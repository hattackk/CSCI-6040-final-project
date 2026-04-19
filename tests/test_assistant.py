from __future__ import annotations

import pytest

from fitd_repro.assistant import (
    AssistantParseError,
    FakeAssistant,
    LLMAssistant,
    _parse_numbered_list,
    _parse_yes_no,
    assistant_deviation_note,
    assistant_paper_faithful,
    assistant_prompts_source,
    build_assistant,
)


class _StubChatModel:
    def __init__(self, reply: str):
        self.reply = reply
        self.calls: list[dict] = []

    def generate(self, messages, max_tokens, temperature):
        self.calls.append(
            {"messages": messages, "max_tokens": max_tokens, "temperature": temperature}
        )
        return self.reply


class TestParseNumberedList:
    def test_parses_dot_form(self):
        text = "1. first\n2. second\n3. third"
        assert _parse_numbered_list(text, 3) == ["first", "second", "third"]

    def test_parses_paren_form(self):
        text = "1) alpha\n2) beta"
        assert _parse_numbered_list(text, 2) == ["alpha", "beta"]

    def test_parses_wrapped_paren_form(self):
        text = "(1) alpha\n(2) beta"
        assert _parse_numbered_list(text, 2) == ["alpha", "beta"]

    def test_accepts_leading_prose(self):
        text = "Here are the queries:\n\n1. first\n2. second\n"
        assert _parse_numbered_list(text, 2) == ["first", "second"]

    def test_raises_when_fewer_items(self):
        with pytest.raises(AssistantParseError):
            _parse_numbered_list("1. only one\n", 3)


class TestParseYesNo:
    @pytest.mark.parametrize("raw,expected", [
        ("Yes.", True),
        ("YES", True),
        ("yes, because it does", True),
        ("no", False),
        ("No, it does not.", False),
        ("NO — mismatched", False),
    ])
    def test_known(self, raw, expected):
        assert _parse_yes_no(raw) is expected

    def test_empty_raises(self):
        with pytest.raises(AssistantParseError):
            _parse_yes_no("")

    def test_ambiguous_raises(self):
        with pytest.raises(AssistantParseError):
            _parse_yes_no("maybe")


class TestFakeAssistant:
    def test_get_level_query_returns_distinct(self):
        assistant = FakeAssistant()
        out = assistant.get_level_query(12, "build X")
        assert len(out) == 12
        assert len(set(out)) == 12

    def test_is_rejection_detects_refusal(self):
        assert FakeAssistant().is_rejection("I cannot help with that") is True
        assert FakeAssistant().is_rejection("Here is the answer") is False

    def test_is_align_via_token_overlap(self):
        assistant = FakeAssistant()
        assert assistant.is_align("how to bake bread", "Here's a bread recipe") is True
        assert assistant.is_align("unrelated question", "totally different topic") is False

    def test_get_mid_is_deterministic(self):
        assistant = FakeAssistant()
        a = assistant.get_mid("qa", "qb")
        b = assistant.get_mid("qa", "qb")
        assert a == b

    def test_get_align_prompt_contains_paper_text(self):
        assistant = FakeAssistant()
        p = assistant.get_align_prompt("some last query", "some last response")
        assert "Re-Align" not in p  # template body, not header
        assert "previous response" in p
        assert "some last query" in p
        assert "some last response" in p


class TestLLMAssistant:
    def test_get_mid_uses_ssp_template(self):
        chat = _StubChatModel("middle-query")
        a = LLMAssistant(chat)
        out = a.get_mid("q_last", "q_i")
        assert out == "middle-query"
        [call] = chat.calls
        [msg] = call["messages"]
        # §A.3.4 verbatim markers
        assert "level of negativity" in msg["content"]
        assert "q_last" in msg["content"]
        assert "q_i" in msg["content"]

    def test_get_level_query_parses_numbered_list(self):
        chat = _StubChatModel("1. one\n2. two\n3. three")
        a = LLMAssistant(chat)
        assert a.get_level_query(3, "final") == ["one", "two", "three"]

    def test_is_align_parses_yes(self):
        chat = _StubChatModel("Yes")
        a = LLMAssistant(chat)
        assert a.is_align("q", "r") is True

    def test_is_rejection_parses_no(self):
        chat = _StubChatModel("no")
        a = LLMAssistant(chat)
        assert a.is_rejection("r") is False

    def test_get_align_prompt_no_chat_call(self):
        chat = _StubChatModel("should not be called")
        a = LLMAssistant(chat)
        p = a.get_align_prompt("q", "r")
        assert "previous response" in p
        assert chat.calls == []

    def test_paraphrase_strips_whitespace(self):
        chat = _StubChatModel("  rephrased content  \n")
        a = LLMAssistant(chat)
        assert a.paraphrase("q", history=[]) == "rephrased content"


class TestBuildAssistant:
    def test_mock_returns_fake_assistant(self):
        assert isinstance(build_assistant("mock", None), FakeAssistant)

    def test_non_mock_requires_model(self):
        with pytest.raises(ValueError, match="requires model_name"):
            build_assistant("openai", None)


class TestPaperFaithfulness:
    def test_openai_gpt4o_mini_is_faithful(self):
        assert assistant_paper_faithful("openai", "gpt-4o-mini") is True
        assert assistant_deviation_note("openai", "gpt-4o-mini") is None

    def test_vllm_is_deviation(self):
        assert assistant_paper_faithful("vllm", "meta-llama/Meta-Llama-3-8B-Instruct") is False
        note = assistant_deviation_note("vllm", "meta-llama/Meta-Llama-3-8B-Instruct")
        assert note is not None
        assert "M" in note  # references the paper's assistant symbol


class TestPromptsSource:
    def test_reports_paper_verbatim_keys(self):
        src = assistant_prompts_source()
        assert src["get_mid"] == "paper_verbatim_A.3.4"
        assert src["get_align_prompt"] == "paper_verbatim_A.3.5"
        for derived_key in ("get_level_query", "is_align", "is_rejection", "paraphrase"):
            assert src[derived_key] == "derived"
