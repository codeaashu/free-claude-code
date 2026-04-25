import json

import pytest

from api.models.anthropic import Message, MessagesRequest, Tool
from api.web_server_tools import (
    is_web_server_tool_request,
    stream_web_server_tool_response,
)


def _event_data(event: str) -> dict:
    data_line = next(line for line in event.splitlines() if line.startswith("data: "))
    return json.loads(data_line.removeprefix("data: "))


def test_detects_web_search_server_tool_request():
    request = MessagesRequest(
        model="claude-haiku-4-5-20251001",
        max_tokens=100,
        messages=[Message(role="user", content="search")],
        tools=[Tool(name="web_search", type="web_search_20250305")],
    )

    assert is_web_server_tool_request(request)


@pytest.mark.asyncio
async def test_streams_web_search_server_tool_result(monkeypatch):
    async def fake_search(query: str) -> list[dict[str, str]]:
        assert query == "DeepSeek V4 model release 2026"
        return [{"title": "DeepSeek V4 Released", "url": "https://example.com/v4"}]

    monkeypatch.setattr("api.web_server_tools._run_web_search", fake_search)
    request = MessagesRequest(
        model="claude-haiku-4-5-20251001",
        max_tokens=100,
        messages=[
            Message(
                role="user",
                content=(
                    "Perform a web search for the query: DeepSeek V4 model release 2026"
                ),
            )
        ],
        tools=[Tool(name="web_search", type="web_search_20250305")],
        tool_choice={"type": "tool", "name": "web_search"},
    )

    events = [
        event
        async for event in stream_web_server_tool_response(request, input_tokens=42)
    ]
    payloads = [_event_data(event) for event in events]

    assert payloads[1]["content_block"]["type"] == "server_tool_use"
    assert payloads[1]["content_block"]["name"] == "web_search"
    assert payloads[3]["content_block"]["type"] == "web_search_tool_result"
    assert payloads[3]["content_block"]["content"][0]["url"] == "https://example.com/v4"
    assert payloads[-2]["usage"]["server_tool_use"] == {"web_search_requests": 1}


@pytest.mark.asyncio
async def test_streams_web_fetch_server_tool_result(monkeypatch):
    async def fake_fetch(url: str) -> dict[str, str]:
        assert url == "https://example.com/article"
        return {
            "url": url,
            "title": "Example Article",
            "media_type": "text/plain",
            "data": "Article body",
        }

    monkeypatch.setattr("api.web_server_tools._run_web_fetch", fake_fetch)
    request = MessagesRequest(
        model="claude-haiku-4-5-20251001",
        max_tokens=100,
        messages=[
            Message(role="user", content="Fetch https://example.com/article please")
        ],
        tools=[Tool(name="web_fetch", type="web_fetch_20250910")],
        tool_choice={"type": "tool", "name": "web_fetch"},
    )

    events = [
        event
        async for event in stream_web_server_tool_response(request, input_tokens=42)
    ]
    payloads = [_event_data(event) for event in events]

    assert payloads[1]["content_block"]["type"] == "server_tool_use"
    assert payloads[3]["content_block"]["type"] == "web_fetch_tool_result"
    assert payloads[3]["content_block"]["content"]["content"]["title"] == (
        "Example Article"
    )
    assert payloads[-2]["usage"]["server_tool_use"] == {"web_fetch_requests": 1}
