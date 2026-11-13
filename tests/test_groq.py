"""Tests for Groq integration (fully mocked — no API key needed)."""

from unittest.mock import MagicMock

import pytest

from rag.config import RAGConfig
from rag.generation.fallback import FallbackResponder
from rag.generation.groq_client import (
    GroqAuthError,
    GroqClient,
    GroqEmptyResponseError,
    GroqGenerationError,
    GroqNetworkError,
    GroqRateLimitError,
)
from rag.generation.prompts import SYSTEM_PROMPT_AR, SYSTEM_PROMPT_EN, build_messages
from rag.schemas import RetrievedContext


def make_config(api_key="test-key-123"):
    config = RAGConfig()
    config.groq_api_key = api_key
    config.groq_model = "openai/gpt-oss-120b"
    config.groq_max_retries = 1
    config.groq_timeout_seconds = 5
    return config


def fake_groq_response(content):
    response = MagicMock()
    response.choices[0].message.content = content
    return response


class MockCompletionAPI:
    """Stand-in for client.chat.completions.create."""

    def __init__(self, results):
        self.results = list(results)
        self.calls = []

    def __call__(self, **kwargs):
        self.calls.append(kwargs)
        result = self.results.pop(0)
        if isinstance(result, Exception):
            raise result
        return result


def make_client(results, config):
    client = GroqClient(config=config)
    sdk = MagicMock()
    sdk.chat.completions.create = MockCompletionAPI(results)
    client._injected_client = sdk
    return client


class TestGroqClient:
    def test_generates_answer(self):
        config = make_config()
        client = make_client([fake_groq_response("The price is 700 EGP [giza_pyramids]")], config)
        answer = client.generate([{"role": "user", "content": "hi"}])
        assert "700 EGP" in answer

    def test_generation_parameters_passed(self):
        config = make_config()
        client = make_client([fake_groq_response("ok")], config)
        client.generate([{"role": "user", "content": "q"}])
        call = client._injected_client.chat.completions.create.calls[0]
        assert call["temperature"] == 0.2
        assert call["top_p"] == 0.9
        assert call["max_tokens"] == 1024
        assert call["model"] == "openai/gpt-oss-120b"

    def test_missing_key_raises_auth_error(self):
        config = make_config(api_key=None)
        config.groq_api_key = None
        client = GroqClient(config=config)
        with pytest.raises(GroqAuthError):
            client.generate([{"role": "user", "content": "q"}])
        assert not client.is_available

    def test_placeholder_key_rejected(self):
        config = make_config(api_key="your_groq_api_key_here")
        client = GroqClient(config=config)
        assert not client.is_available
        with pytest.raises(GroqAuthError):
            client.generate([{"role": "user", "content": "q"}])

    def test_empty_response_raises(self):
        client = make_client([fake_groq_response("")], make_config())
        with pytest.raises(GroqEmptyResponseError):
            client.generate([{"role": "user", "content": "q"}])

    def test_rate_limit_retries_then_succeeds(self):
        client = make_client(
            [GroqRateLimitError("429 too many"), fake_groq_response("ok")],
            make_config(),
        )
        answer = client.generate([{"role": "user", "content": "q"}])
        assert answer == "ok"
        assert len(client._injected_client.chat.completions.create.calls) == 2

    def test_rate_limit_exhausted(self):
        client = make_client(
            [GroqRateLimitError("429"), GroqRateLimitError("429")],
            make_config(),
        )
        with pytest.raises(GroqRateLimitError):
            client.generate([{"role": "user", "content": "q"}])

    def test_network_error_retries_then_fails(self):
        client = make_client(
            [GroqNetworkError("timeout"), GroqNetworkError("timeout")],
            make_config(),
        )
        with pytest.raises(GroqNetworkError):
            client.generate([{"role": "user", "content": "q"}])

    def test_auth_error_not_retried(self):
        client = make_client([AuthStub(), fake_groq_response("should not happen")], make_config())
        with pytest.raises(GroqAuthError):
            client.generate([{"role": "user", "content": "q"}])
        # no retry after auth failure
        assert len(client._injected_client.chat.completions.create.calls) == 1

    def test_secrets_not_in_error_messages(self):
        config = make_config(api_key="super-secret-key")
        client = make_client([GroqNetworkError("boom"), GroqNetworkError("boom2")], config)
        with pytest.raises(GroqNetworkError) as excinfo:
            client.generate([{"role": "user", "content": "q"}])
        assert "super-secret-key" not in str(excinfo.value)


class AuthStub(Exception):
    def __init__(self):
        super().__init__("401 invalid api key")
        self.status_code = 401


# ----------------------------------------------------------------------
# Prompts
# ----------------------------------------------------------------------
class TestPrompts:
    def _context(self):
        return RetrievedContext(
            query="q",
            text="EVIDENCE TEXT",
            sources=["giza_pyramids: giza_pyramids/info.json"],
        )

    def test_build_messages_english(self):
        messages = build_messages(self._context(), "What is the price?", "en")
        assert messages[0]["role"] == "system"
        assert SYSTEM_PROMPT_EN.split("\n")[0] in messages[0]["content"]
        assert "EVIDENCE TEXT" in messages[1]["content"]

    def test_build_messages_arabic(self):
        messages = build_messages(self._context(), "ما هو السعر؟", "ar")
        assert "رحلة" in messages[0]["content"]


# ----------------------------------------------------------------------
# Fallback responder (template-based) + validation guardrails
# ----------------------------------------------------------------------
class TestFallback:
    def _context(self, text="STRUCTURED EVIDENCE"):
        return RetrievedContext(query="q", text=text, sources=["s1: file.json"])

    def test_fallback_no_raw_evidence_dump(self):
        responder = FallbackResponder()
        answer = responder.respond(self._context(), "en")
        # concise not-found sentence, never a raw evidence dump
        assert "couldn't find this information" in answer.lower()
        assert "STRUCTURED EVIDENCE" not in answer

    def test_fallback_arabic_notice(self):
        responder = FallbackResponder()
        answer = responder.respond(self._context(), "ar")
        assert "لم أجد" in answer
        assert "قاعدة المعرفة" in answer

    def test_empty_context_reports_unavailable_naturally(self):
        responder = FallbackResponder()
        answer = responder.respond(self._context(""), "en")
        assert "couldn't find this information" in answer
        # no internal debug markers, no raw JSON
        assert "{" not in answer and "}" not in answer

    def test_guardrail_detects_unsupported_numbers(self):
        responder = FallbackResponder()
        context = self._context("price is 700 EGP")
        warnings = responder.validate_answer("It costs 999 EGP total", context)
        assert any(w.startswith("unsupported_numbers") and "999" in w for w in warnings)

    def test_guardrail_accepts_supported_numbers(self):
        responder = FallbackResponder()
        context = self._context("price is 700 EGP")
        warnings = responder.validate_answer("It costs 700 EGP [giza_pyramids]", context)
        assert not any(w.startswith("unsupported_numbers") for w in warnings)

    def test_guardrail_empty_answer(self):
        responder = FallbackResponder()
        warnings = responder.validate_answer("   ", self._context())
        assert "empty_answer" in warnings
