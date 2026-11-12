"""Groq API client — the primary (and only) LLM generation provider.

Design:
  - official ``groq`` Python SDK, OpenAI-compatible chat completion endpoint
  - API key read from ``GROQ_API_KEY`` (never hardcoded, never logged)
  - model from ``GROQ_MODEL`` (configurable without code changes)
  - configurable timeout + retries with exponential backoff
  - explicit error classes for auth / rate-limit / network / empty responses
  - the client may be injected for testing (mocked LLM responses)
"""

from __future__ import annotations

import logging
import time
from typing import Any, Dict, List, Optional

from rag.config import RAGConfig

logger = logging.getLogger(__name__)


class GroqGenerationError(Exception):
    """Base class for generation failures."""

    def __init__(self, message: str, kind: str = "generation_error") -> None:
        super().__init__(message)
        self.kind = kind


class GroqAuthError(GroqGenerationError):
    def __init__(self, message: str) -> None:
        super().__init__(message, kind="auth_error")


class GroqRateLimitError(GroqGenerationError):
    def __init__(self, message: str) -> None:
        super().__init__(message, kind="rate_limit_error")


class GroqNetworkError(GroqGenerationError):
    def __init__(self, message: str) -> None:
        super().__init__(message, kind="network_error")


class GroqEmptyResponseError(GroqGenerationError):
    def __init__(self, message: str = "LLM returned an empty response") -> None:
        super().__init__(message, kind="empty_response")


class GroqClient:
    """Thin, testable wrapper around the Groq chat completions API."""

    def __init__(
        self,
        config: Optional[RAGConfig] = None,
        client: Optional[Any] = None,
    ) -> None:
        self.config = config or RAGConfig()
        self._injected_client = client
        self._client: Optional[Any] = None

    # ------------------------------------------------------------------
    def _get_client(self):
        """Lazily instantiate the Groq SDK client."""
        if self._injected_client is not None:
            return self._injected_client
        if self._client is None:
            if not self.config.has_groq_key:
                raise GroqAuthError(
                    "GROQ_API_KEY is not set. Add your key to the .env file."
                )
            try:
                from groq import Groq
            except ImportError as exc:  # pragma: no cover
                raise GroqGenerationError(
                    "The 'groq' package is not installed", kind="import_error"
                ) from exc
            self._client = Groq(
                api_key=self.config.groq_api_key,
                timeout=self.config.groq_timeout_seconds,
                max_retries=0,  # retries are handled explicitly below
            )
        return self._client

    @property
    def is_available(self) -> bool:
        """True when generation can be attempted (key present or injected client)."""
        if self._injected_client is not None:
            return True
        return self.config.has_groq_key

    # ------------------------------------------------------------------
    def generate(
        self,
        messages: List[Dict[str, str]],
        max_output_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
    ) -> str:
        """Call Groq chat completion; returns the answer text.

        Retries with exponential backoff on rate limits and transient network
        errors. Authentication and empty-response failures raise immediately.
        """
        if not self.is_available:
            raise GroqAuthError(
                "GROQ_API_KEY is not configured; generation is unavailable"
            )

        client = self._get_client()
        last_error: Optional[Exception] = None
        max_attempts = max(1, self.config.groq_max_retries + 1)

        for attempt in range(1, max_attempts + 1):
            try:
                response = client.chat.completions.create(
                    model=self.config.groq_model,
                    messages=messages,
                    temperature=(
                        self.config.groq_temperature if temperature is None
                        else temperature
                    ),
                    top_p=self.config.groq_top_p if top_p is None else top_p,
                    max_tokens=max_output_tokens or self.config.groq_max_output_tokens,
                )
                content = self._extract_content(response)
                if not content or not content.strip():
                    raise GroqEmptyResponseError()
                return content.strip()

            except GroqEmptyResponseError:
                logger.error("Groq returned an empty response")
                raise
            except Exception as exc:  # noqa: BLE001 - mapped below
                mapped = self._map_error(exc)
                if mapped is None:
                    raise
                last_error = mapped
                if mapped.kind in ("rate_limit_error", "network_error") and attempt < max_attempts:
                    backoff = 2 ** (attempt - 1)
                    logger.warning(
                        "Groq attempt %d/%d failed (%s); retrying in %ss",
                        attempt, max_attempts, mapped.kind, backoff,
                    )
                    time.sleep(backoff)
                    continue
                logger.error("Groq generation failed: %s", mapped)
                raise mapped

        raise last_error or GroqGenerationError("Groq generation failed")

    # ------------------------------------------------------------------
    @staticmethod
    def _extract_content(response: Any) -> str:
        try:
            choice = response.choices[0]
            return choice.message.content or ""
        except (AttributeError, IndexError, KeyError, TypeError):
            return ""

    @staticmethod
    def _map_error(exc: Exception) -> Optional[GroqGenerationError]:
        """Map SDK/HTTP exceptions to typed errors; None means re-raise."""
        status = getattr(exc, "status_code", None)
        message = str(exc)
        # never include the API key in logged/raised messages
        api_key = __import__("os").getenv("GROQ_API_KEY", "")
        if api_key:
            message = message.replace(api_key, "<redacted>")
        if status == 401 or "api key" in message.lower() or "authentication" in message.lower():
            return GroqAuthError(f"Groq authentication failed: {message[:300]}")
        if status == 429 or "rate limit" in message.lower():
            return GroqRateLimitError(f"Groq rate limit hit: {message[:300]}")
        if status is not None and 500 <= int(status) < 600:
            return GroqNetworkError(f"Groq server error: {message[:300]}")
        if "timeout" in message.lower() or "connection" in message.lower():
            return GroqNetworkError(f"Groq network error: {message[:300]}")
        if isinstance(exc, GroqGenerationError):
            return exc
        return GroqGenerationError(f"Groq error: {message[:300]}")
