"""OpenRouter provider implementation."""

from __future__ import annotations

import json
import uuid
from collections.abc import Iterator
from dataclasses import dataclass, field
from typing import Any

from core.anthropic import SSEBuilder, append_request_id
from providers.anthropic_messages import AnthropicMessagesTransport, StreamChunkMode
from providers.base import ProviderConfig
from providers.defaults import OPENROUTER_BASE_URL

from .request import build_request_body

_ANTHROPIC_VERSION = "2023-06-01"


@dataclass
class _SSEFilterState:
    """Track Anthropic content block index remapping while filtering thinking."""

    next_index: int = 0
    index_map: dict[int, int] = field(default_factory=dict)
    dropped_indexes: set[int] = field(default_factory=set)
    open_block_types: dict[int, str] = field(default_factory=dict)
    closed_indexes: set[int] = field(default_factory=set)
    message_stopped: bool = False


class OpenRouterProvider(AnthropicMessagesTransport):
    """OpenRouter provider using the native Anthropic-compatible messages API."""

    stream_chunk_mode: StreamChunkMode = "event"

    def __init__(self, config: ProviderConfig):
        super().__init__(
            config,
            provider_name="OPENROUTER",
            default_base_url=OPENROUTER_BASE_URL,
        )

    def _build_request_body(self, request: Any) -> dict:
        """Internal helper for tests and direct request dispatch."""
        return build_request_body(
            request,
            thinking_enabled=self._is_thinking_enabled(request),
        )

    def _request_headers(self) -> dict[str, str]:
        """Return OpenRouter's Anthropic-compatible messages headers."""
        return {
            "Accept": "text/event-stream",
            "Authorization": f"Bearer {self._api_key}",
            "Content-Type": "application/json",
            "anthropic-version": _ANTHROPIC_VERSION,
        }

    @staticmethod
    def _format_sse_event(event_name: str | None, data_text: str) -> str:
        """Format an SSE event from its event name and data payload."""
        lines: list[str] = []
        if event_name:
            lines.append(f"event: {event_name}")
        lines.extend(f"data: {line}" for line in data_text.splitlines())
        return "\n".join(lines) + "\n\n"

    @staticmethod
    def _parse_sse_event(event: str) -> tuple[str | None, str]:
        """Extract the event name and raw data payload from an SSE event."""
        event_name = None
        data_lines: list[str] = []
        for line in event.strip().splitlines():
            if line.startswith("event:"):
                event_name = line[6:].strip()
            elif line.startswith("data:"):
                data_lines.append(line[5:].lstrip())
        return event_name, "\n".join(data_lines)

    @staticmethod
    def _is_terminal_done_event(event_name: str | None, data_text: str) -> bool:
        """Return whether an event is OpenAI-style terminal noise."""
        return (event_name is None or event_name in {"data", "done"}) and (
            data_text.strip().upper() == "[DONE]"
        )

    @staticmethod
    def _remap_index(
        payload: dict[str, Any], state: _SSEFilterState, *, create: bool
    ) -> int | None:
        """Return the downstream index for a content block event."""
        upstream_index = payload.get("index")
        if not isinstance(upstream_index, int):
            return None
        if upstream_index in state.dropped_indexes:
            return None
        mapped_index = state.index_map.get(upstream_index)
        if mapped_index is None and create:
            mapped_index = state.next_index
            state.index_map[upstream_index] = mapped_index
            state.next_index += 1
        return mapped_index

    def _close_open_blocks_before(
        self, state: _SSEFilterState, upstream_index: int
    ) -> str:
        """Close overlapping upstream blocks before starting a new block."""
        events: list[str] = []
        for open_upstream_index in list(state.open_block_types):
            if open_upstream_index == upstream_index:
                continue
            mapped_index = state.index_map.get(open_upstream_index)
            if mapped_index is None:
                continue
            payload = {"type": "content_block_stop", "index": mapped_index}
            events.append(
                self._format_sse_event("content_block_stop", json.dumps(payload))
            )
            state.closed_indexes.add(open_upstream_index)
            state.open_block_types.pop(open_upstream_index, None)
        return "".join(events)

    @staticmethod
    def _should_drop_block_type(block_type: Any, *, thinking_enabled: bool) -> bool:
        if not isinstance(block_type, str):
            return False
        if block_type.startswith("redacted_thinking"):
            return True
        return not thinking_enabled and "thinking" in block_type

    def _transform_sse_payload(
        self,
        event: str,
        state: _SSEFilterState,
        *,
        thinking_enabled: bool,
    ) -> str | None:
        """Normalize OpenRouter SSE events and enforce local thinking policy."""
        event_name, data_text = self._parse_sse_event(event)
        if not event_name or not data_text:
            return event

        try:
            payload = json.loads(data_text)
        except json.JSONDecodeError:
            return event

        if event_name == "content_block_start":
            block = payload.get("content_block")
            if not isinstance(block, dict):
                return event
            block_type = block.get("type")
            upstream_index = payload.get("index")
            if self._should_drop_block_type(
                block_type, thinking_enabled=thinking_enabled
            ):
                if isinstance(upstream_index, int):
                    state.dropped_indexes.add(upstream_index)
                return None

            mapped_index = self._remap_index(payload, state, create=True)
            if mapped_index is not None:
                payload["index"] = mapped_index
                if isinstance(upstream_index, int) and isinstance(block_type, str):
                    prefix = self._close_open_blocks_before(state, upstream_index)
                    state.open_block_types[upstream_index] = block_type
                    return prefix + self._format_sse_event(
                        event_name, json.dumps(payload)
                    )
                return self._format_sse_event(event_name, json.dumps(payload))
            return None if not thinking_enabled else event

        if event_name == "content_block_delta":
            delta = payload.get("delta")
            if not isinstance(delta, dict):
                return event
            delta_type = delta.get("type")
            if self._should_drop_block_type(
                delta_type, thinking_enabled=thinking_enabled
            ):
                return None

            mapped_index = self._remap_index(payload, state, create=False)
            if mapped_index is not None:
                payload["index"] = mapped_index
                return self._format_sse_event(event_name, json.dumps(payload))
            if payload.get("index") in state.dropped_indexes:
                return None
            if not thinking_enabled:
                return None

        if event_name == "content_block_stop":
            upstream_index = payload.get("index")
            if (
                isinstance(upstream_index, int)
                and upstream_index in state.closed_indexes
            ):
                state.closed_indexes.discard(upstream_index)
                return None
            mapped_index = self._remap_index(payload, state, create=False)
            if mapped_index is not None:
                payload["index"] = mapped_index
                if isinstance(upstream_index, int):
                    state.open_block_types.pop(upstream_index, None)
                return self._format_sse_event(event_name, json.dumps(payload))
            if payload.get("index") in state.dropped_indexes:
                return None
            if not thinking_enabled:
                return None

        return event

    def _new_stream_state(self, request: Any, *, thinking_enabled: bool) -> Any:
        """Create per-stream state for thinking block filtering."""
        return _SSEFilterState()

    def _transform_stream_event(
        self,
        event: str,
        state: Any,
        *,
        thinking_enabled: bool,
    ) -> str | None:
        """Drop provider-specific terminal noise and hidden thinking events."""
        if isinstance(state, _SSEFilterState):
            event_name, data_text = self._parse_sse_event(event)
            if state.message_stopped or self._is_terminal_done_event(
                event_name, data_text
            ):
                return None
            if event_name == "message_stop":
                state.message_stopped = True

        if thinking_enabled:
            if isinstance(state, _SSEFilterState):
                return self._transform_sse_payload(
                    event, state, thinking_enabled=thinking_enabled
                )
            return event
        if isinstance(state, _SSEFilterState):
            return self._transform_sse_payload(
                event, state, thinking_enabled=thinking_enabled
            )
        return event

    def _format_error_message(self, base_message: str, request_id: str | None) -> str:
        """Keep OpenRouter's existing request-id suffix format."""
        return append_request_id(base_message, request_id)

    def _emit_error_events(
        self,
        *,
        request: Any,
        input_tokens: int,
        error_message: str,
        sent_any_event: bool,
    ) -> Iterator[str]:
        """Emit the Anthropic SSE error shape expected by Claude clients."""
        sse = SSEBuilder(f"msg_{uuid.uuid4()}", request.model, input_tokens)
        if not sent_any_event:
            yield sse.message_start()
        yield from sse.emit_error(error_message)
        yield sse.message_delta("end_turn", 1)
        yield sse.message_stop()
