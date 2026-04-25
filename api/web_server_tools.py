"""Local handlers for Anthropic web server tools.

OpenAI-compatible upstreams can emit regular function calls, but Anthropic's
web tools are server-side: the API response itself must include the tool result.
"""

from __future__ import annotations

import html
import json
import re
import uuid
from collections.abc import AsyncIterator
from datetime import UTC, datetime
from html.parser import HTMLParser
from typing import Any
from urllib.parse import parse_qs, unquote, urlparse

import httpx

from .models.anthropic import MessagesRequest

_REQUEST_TIMEOUT_S = 20.0
_MAX_SEARCH_RESULTS = 10
_MAX_FETCH_CHARS = 24_000


class _SearchResultParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.results: list[dict[str, str]] = []
        self._href: str | None = None
        self._title_parts: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag != "a":
            return
        href = dict(attrs).get("href")
        if not href or "uddg=" not in href:
            return
        parsed = urlparse(href)
        query = parse_qs(parsed.query)
        uddg = query.get("uddg", [""])[0]
        if not uddg:
            return
        self._href = unquote(uddg)
        self._title_parts = []

    def handle_data(self, data: str) -> None:
        if self._href is not None:
            self._title_parts.append(data)

    def handle_endtag(self, tag: str) -> None:
        if tag != "a" or self._href is None:
            return
        title = " ".join("".join(self._title_parts).split())
        if title and not any(result["url"] == self._href for result in self.results):
            self.results.append({"title": html.unescape(title), "url": self._href})
        self._href = None
        self._title_parts = []


class _HTMLTextParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.title = ""
        self.text_parts: list[str] = []
        self._in_title = False
        self._skip_depth = 0

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag in {"script", "style", "noscript"}:
            self._skip_depth += 1
        elif tag == "title":
            self._in_title = True

    def handle_endtag(self, tag: str) -> None:
        if tag in {"script", "style", "noscript"} and self._skip_depth:
            self._skip_depth -= 1
        elif tag == "title":
            self._in_title = False

    def handle_data(self, data: str) -> None:
        text = " ".join(data.split())
        if not text:
            return
        if self._in_title:
            self.title = f"{self.title} {text}".strip()
        elif not self._skip_depth:
            self.text_parts.append(text)


def _format_event(event_type: str, data: dict[str, Any]) -> str:
    return f"event: {event_type}\ndata: {json.dumps(data)}\n\n"


def _content_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, dict):
                parts.append(str(item.get("text", "")))
            else:
                parts.append(str(getattr(item, "text", "")))
        return "\n".join(part for part in parts if part)
    return str(content)


def _request_text(request: MessagesRequest) -> str:
    return "\n".join(_content_text(message.content) for message in request.messages)


def _web_tool_name(request: MessagesRequest) -> str | None:
    for tool in request.tools or []:
        name = tool.name
        tool_type = tool.type or ""
        if name in {"web_search", "web_fetch"} or tool_type.startswith("web_"):
            return name
    return None


def is_web_server_tool_request(request: MessagesRequest) -> bool:
    return _web_tool_name(request) in {"web_search", "web_fetch"}


def _extract_query(text: str) -> str:
    match = re.search(r"query:\s*(.+)", text, flags=re.IGNORECASE | re.DOTALL)
    if match:
        return match.group(1).strip().strip("\"'")
    return text.strip()


def _extract_url(text: str) -> str:
    match = re.search(r"https?://\S+", text)
    return match.group(0).rstrip(").,]") if match else text.strip()


async def _run_web_search(query: str) -> list[dict[str, str]]:
    async with httpx.AsyncClient(
        timeout=_REQUEST_TIMEOUT_S,
        follow_redirects=True,
        headers={"User-Agent": "Mozilla/5.0 compatible; free-claude-code/2.0"},
    ) as client:
        response = await client.get(
            "https://lite.duckduckgo.com/lite/",
            params={"q": query},
        )
        response.raise_for_status()

    parser = _SearchResultParser()
    parser.feed(response.text)
    return parser.results[:_MAX_SEARCH_RESULTS]


async def _run_web_fetch(url: str) -> dict[str, str]:
    async with httpx.AsyncClient(
        timeout=_REQUEST_TIMEOUT_S,
        follow_redirects=True,
        headers={"User-Agent": "Mozilla/5.0 compatible; free-claude-code/2.0"},
    ) as client:
        response = await client.get(url)
        response.raise_for_status()

    content_type = response.headers.get("content-type", "text/plain")
    title = url
    data = response.text
    if "html" in content_type.lower():
        parser = _HTMLTextParser()
        parser.feed(response.text)
        title = parser.title or url
        data = "\n".join(parser.text_parts)
    return {
        "url": str(response.url),
        "title": title,
        "media_type": "text/plain",
        "data": data[:_MAX_FETCH_CHARS],
    }


def _search_summary(query: str, results: list[dict[str, str]]) -> str:
    if not results:
        return f"No web search results found for: {query}"
    lines = [f"Search results for: {query}"]
    for index, result in enumerate(results, start=1):
        lines.append(f"{index}. {result['title']}\n{result['url']}")
    return "\n\n".join(lines)


async def stream_web_server_tool_response(
    request: MessagesRequest, input_tokens: int
) -> AsyncIterator[str]:
    tool_name = _web_tool_name(request)
    if tool_name is None:
        return

    text = _request_text(request)
    message_id = f"msg_{uuid.uuid4()}"
    tool_id = f"srvtoolu_{uuid.uuid4().hex}"
    output_tokens = 1
    usage_key = (
        "web_search_requests" if tool_name == "web_search" else "web_fetch_requests"
    )
    tool_input = (
        {"query": _extract_query(text)}
        if tool_name == "web_search"
        else {"url": _extract_url(text)}
    )

    yield _format_event(
        "message_start",
        {
            "type": "message_start",
            "message": {
                "id": message_id,
                "type": "message",
                "role": "assistant",
                "content": [],
                "model": request.model,
                "stop_reason": None,
                "stop_sequence": None,
                "usage": {"input_tokens": input_tokens, "output_tokens": 1},
            },
        },
    )
    yield _format_event(
        "content_block_start",
        {
            "type": "content_block_start",
            "index": 0,
            "content_block": {
                "type": "server_tool_use",
                "id": tool_id,
                "name": tool_name,
                "input": tool_input,
            },
        },
    )
    yield _format_event(
        "content_block_stop", {"type": "content_block_stop", "index": 0}
    )

    try:
        if tool_name == "web_search":
            query = str(tool_input["query"])
            results = await _run_web_search(query)
            result_content: Any = [
                {
                    "type": "web_search_result",
                    "title": result["title"],
                    "url": result["url"],
                }
                for result in results
            ]
            summary = _search_summary(query, results)
            result_block_type = "web_search_tool_result"
        else:
            fetched = await _run_web_fetch(str(tool_input["url"]))
            result_content = {
                "type": "web_fetch_result",
                "url": fetched["url"],
                "content": {
                    "type": "document",
                    "source": {
                        "type": "text",
                        "media_type": fetched["media_type"],
                        "data": fetched["data"],
                    },
                    "title": fetched["title"],
                    "citations": {"enabled": True},
                },
                "retrieved_at": datetime.now(UTC).isoformat(),
            }
            summary = fetched["data"][:_MAX_FETCH_CHARS]
            result_block_type = "web_fetch_tool_result"
    except Exception as error:
        result_block_type = (
            "web_search_tool_result"
            if tool_name == "web_search"
            else "web_fetch_tool_result"
        )
        error_type = (
            "web_search_tool_result_error"
            if tool_name == "web_search"
            else "web_fetch_tool_error"
        )
        result_content = {"type": error_type, "error_code": "unavailable"}
        summary = f"{tool_name} failed: {type(error).__name__}"

    output_tokens = max(1, len(summary) // 4)

    yield _format_event(
        "content_block_start",
        {
            "type": "content_block_start",
            "index": 1,
            "content_block": {
                "type": result_block_type,
                "tool_use_id": tool_id,
                "content": result_content,
            },
        },
    )
    yield _format_event(
        "content_block_stop", {"type": "content_block_stop", "index": 1}
    )
    yield _format_event(
        "content_block_start",
        {
            "type": "content_block_start",
            "index": 2,
            "content_block": {"type": "text", "text": summary},
        },
    )
    yield _format_event(
        "content_block_stop", {"type": "content_block_stop", "index": 2}
    )
    yield _format_event(
        "message_delta",
        {
            "type": "message_delta",
            "delta": {"stop_reason": "end_turn", "stop_sequence": None},
            "usage": {
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "server_tool_use": {usage_key: 1},
            },
        },
    )
    yield _format_event("message_stop", {"type": "message_stop"})
