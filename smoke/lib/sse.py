"""SSE parsing and Anthropic stream assertions for smoke tests.

Canonical implementation lives in :mod:`core.anthropic.stream_contracts`; this
module re-exports it for smoke and historical import paths.
"""

from __future__ import annotations

from core.anthropic.stream_contracts import (
    SSEEvent,
    assert_anthropic_stream_contract,
    event_names,
    has_tool_use,
    parse_sse_lines,
    parse_sse_text,
    text_content,
    thinking_content,
)

__all__ = [
    "SSEEvent",
    "assert_anthropic_stream_contract",
    "event_names",
    "has_tool_use",
    "parse_sse_lines",
    "parse_sse_text",
    "text_content",
    "thinking_content",
]
