"""Shared transport for providers with native Anthropic Messages endpoints."""

from __future__ import annotations

import json
from collections.abc import AsyncIterator, Iterator
from typing import Any, Literal

import httpx
from loguru import logger

from core.anthropic import get_user_facing_error_message
from providers.base import BaseProvider, ProviderConfig
from providers.error_mapping import map_error
from providers.rate_limit import GlobalRateLimiter

ANTHROPIC_DEFAULT_MAX_TOKENS = 81920
StreamChunkMode = Literal["line", "event"]


class AnthropicMessagesTransport(BaseProvider):
    """Base class for providers that stream from an Anthropic-compatible endpoint."""

    stream_chunk_mode: StreamChunkMode = "line"

    def __init__(
        self,
        config: ProviderConfig,
        *,
        provider_name: str,
        default_base_url: str,
    ):
        super().__init__(config)
        self._provider_name = provider_name
        self._api_key = config.api_key
        self._base_url = (config.base_url or default_base_url).rstrip("/")
        self._global_rate_limiter = GlobalRateLimiter.get_scoped_instance(
            provider_name.lower(),
            rate_limit=config.rate_limit,
            rate_window=config.rate_window,
            max_concurrency=config.max_concurrency,
        )
        self._client = httpx.AsyncClient(
            base_url=self._base_url,
            proxy=config.proxy or None,
            timeout=httpx.Timeout(
                config.http_read_timeout,
                connect=config.http_connect_timeout,
                read=config.http_read_timeout,
                write=config.http_write_timeout,
            ),
        )

    async def cleanup(self) -> None:
        """Release HTTP client resources."""
        await self._client.aclose()

    def _request_headers(self) -> dict[str, str]:
        """Return headers for the native messages request."""
        return {"Content-Type": "application/json"}

    def _build_request_body(self, request: Any) -> dict:
        """Build a native Anthropic request body."""
        thinking_enabled = self._is_thinking_enabled(request)
        body = request.model_dump(exclude_none=True)

        body.pop("extra_body", None)
        body.pop("original_model", None)
        body.pop("resolved_provider_model", None)

        if "thinking" in body:
            thinking_cfg = body.pop("thinking")
            if thinking_enabled and isinstance(thinking_cfg, dict):
                thinking_payload = {"type": "enabled"}
                budget_tokens = thinking_cfg.get("budget_tokens")
                if isinstance(budget_tokens, int):
                    thinking_payload["budget_tokens"] = budget_tokens
                body["thinking"] = thinking_payload

        if "max_tokens" not in body:
            body["max_tokens"] = ANTHROPIC_DEFAULT_MAX_TOKENS

        return body

    async def _send_stream_request(self, body: dict) -> httpx.Response:
        """Create a streaming messages response."""
        request = self._client.build_request(
            "POST",
            "/messages",
            json=body,
            headers=self._request_headers(),
        )
        return await self._client.send(request, stream=True)

    async def _raise_for_status(
        self, response: httpx.Response, *, req_tag: str
    ) -> None:
        """Raise for non-200 responses after logging the upstream body."""
        try:
            response.raise_for_status()
        except httpx.HTTPStatusError as error:
            response_text = await self._read_error_body(response)
            if response_text:
                logger.error(
                    "{}_ERROR:{} HTTP {}: {}",
                    self._provider_name,
                    req_tag,
                    response.status_code,
                    response_text,
                )
            raise error

    async def _read_error_body(self, response: httpx.Response) -> str:
        """Read a response body for diagnostics."""
        aread = getattr(response, "aread", None)
        if aread is None:
            return ""
        body = await aread()
        if isinstance(body, bytes):
            return body.decode("utf-8", errors="replace")
        return str(body)

    async def _iter_sse_lines(self, response: httpx.Response) -> AsyncIterator[str]:
        """Yield raw SSE line chunks preserving local provider behavior."""
        async for line in response.aiter_lines():
            if line:
                yield f"{line}\n"
            else:
                yield "\n"

    async def _iter_sse_events(self, response: httpx.Response) -> AsyncIterator[str]:
        """Group line-delimited SSE responses into full SSE events."""
        event_lines: list[str] = []
        async for line in response.aiter_lines():
            if line:
                event_lines.append(line)
                continue
            if event_lines:
                yield "\n".join(event_lines) + "\n\n"
                event_lines.clear()
        if event_lines:
            yield "\n".join(event_lines) + "\n\n"

    def _new_stream_state(self, request: Any, *, thinking_enabled: bool) -> Any:
        """Return per-stream provider state for event transformation."""
        return None

    def _transform_stream_event(
        self,
        event: str,
        state: Any,
        *,
        thinking_enabled: bool,
    ) -> str | None:
        """Transform or drop a grouped SSE event before yielding it downstream."""
        return event

    def _format_error_message(self, base_message: str, request_id: str | None) -> str:
        """Apply provider-specific request-id formatting to an error message."""
        if request_id:
            return f"{base_message}\nRequest ID: {request_id}"
        return base_message

    def _get_error_message(self, error: Exception, request_id: str | None) -> str:
        """Map an exception into a user-facing provider error message."""
        mapped_error = map_error(error, rate_limiter=self._global_rate_limiter)
        if getattr(mapped_error, "status_code", None) == 405:
            base_message = (
                f"Upstream provider {self._provider_name} rejected the request method "
                "or endpoint (HTTP 405)."
            )
        else:
            base_message = get_user_facing_error_message(
                mapped_error, read_timeout_s=self._config.http_read_timeout
            )
        return self._format_error_message(base_message, request_id)

    def _emit_error_events(
        self,
        *,
        request: Any,
        input_tokens: int,
        error_message: str,
        sent_any_event: bool,
    ) -> Iterator[str]:
        """Emit a native Anthropic error event."""
        error_event = {
            "type": "error",
            "error": {"type": "api_error", "message": error_message},
        }
        yield f"event: error\ndata: {json.dumps(error_event)}\n\n"

    async def _iter_stream_chunks(
        self,
        response: httpx.Response,
        *,
        state: Any,
        thinking_enabled: bool,
    ) -> AsyncIterator[str]:
        """Yield stream chunks according to the provider's observable chunk shape."""
        if self.stream_chunk_mode == "line":
            async for chunk in self._iter_sse_lines(response):
                yield chunk
            return

        async for event in self._iter_sse_events(response):
            output_event = self._transform_stream_event(
                event,
                state,
                thinking_enabled=thinking_enabled,
            )
            if output_event is not None:
                yield output_event

    async def stream_response(
        self,
        request: Any,
        input_tokens: int = 0,
        *,
        request_id: str | None = None,
    ) -> AsyncIterator[str]:
        """Stream response via a native Anthropic-compatible messages endpoint."""
        tag = self._provider_name
        req_tag = f" request_id={request_id}" if request_id else ""
        thinking_enabled = self._is_thinking_enabled(request)
        body = self._build_request_body(request)

        logger.info(
            "{}_STREAM:{} natively passing Anthropic request model={} msgs={} tools={}",
            tag,
            req_tag,
            body.get("model"),
            len(body.get("messages", [])),
            len(body.get("tools", [])),
        )

        response: httpx.Response | None = None
        sent_any_event = False
        state = self._new_stream_state(request, thinking_enabled=thinking_enabled)

        async with self._global_rate_limiter.concurrency_slot():
            try:
                response = await self._global_rate_limiter.execute_with_retry(
                    self._send_stream_request, body
                )

                if response.status_code != 200:
                    await self._raise_for_status(response, req_tag=req_tag)

                async for chunk in self._iter_stream_chunks(
                    response,
                    state=state,
                    thinking_enabled=thinking_enabled,
                ):
                    sent_any_event = True
                    yield chunk

            except Exception as error:
                logger.error(
                    "{}_ERROR:{} {}: {}", tag, req_tag, type(error).__name__, error
                )
                error_message = self._get_error_message(error, request_id)

                if response is not None and not response.is_closed:
                    await response.aclose()

                logger.info(
                    "{}_STREAM: Emitting native SSE error event for {}{}",
                    tag,
                    type(error).__name__,
                    req_tag,
                )
                for event in self._emit_error_events(
                    request=request,
                    input_tokens=input_tokens,
                    error_message=error_message,
                    sent_any_event=sent_any_event,
                ):
                    yield event
                return
            finally:
                if response is not None and not response.is_closed:
                    await response.aclose()
