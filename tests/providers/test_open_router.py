"""Tests for OpenRouter providers."""

import json
from contextlib import asynccontextmanager
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from providers.base import ProviderConfig
from providers.open_router import OpenRouterProvider
from providers.open_router.request import OPENROUTER_DEFAULT_MAX_TOKENS


class MockMessage:
    def __init__(self, role, content):
        self.role = role
        self.content = content


class MockRequest:
    def __init__(self, **kwargs):
        self.model = "stepfun/step-3.5-flash:free"
        self.messages = [MockMessage("user", "Hello")]
        self.max_tokens = 100
        self.temperature = 0.5
        self.top_p = 0.9
        self.system = "System prompt"
        self.stop_sequences = None
        self.tools = []
        self.tool_choice = None
        self.metadata = None
        self.extra_body = {}
        self.original_model = "claude-3-sonnet"
        self.resolved_provider_model = "open_router/stepfun/step-3.5-flash:free"
        self.thinking = MagicMock()
        self.thinking.enabled = True
        for k, v in kwargs.items():
            setattr(self, k, v)


class FakeResponse:
    def __init__(self, *, status_code=200, lines=None, text=""):
        self.status_code = status_code
        self._lines = lines or []
        self._text = text
        self.is_closed = False

    async def aiter_lines(self):
        for line in self._lines:
            yield line

    async def aread(self):
        return self._text.encode()

    def raise_for_status(self):
        import httpx

        response = httpx.Response(
            self.status_code,
            request=httpx.Request("POST", "https://openrouter.ai/api/v1/messages"),
            text=self._text,
        )
        response.raise_for_status()

    async def aclose(self):
        self.is_closed = True


@pytest.fixture
def open_router_config():
    return ProviderConfig(
        api_key="test_openrouter_key",
        base_url="https://openrouter.ai/api/v1",
        rate_limit=10,
        rate_window=60,
    )


@pytest.fixture(autouse=True)
def mock_rate_limiter():
    @asynccontextmanager
    async def _slot():
        yield

    with patch("providers.anthropic_messages.GlobalRateLimiter") as mock:
        instance = mock.get_scoped_instance.return_value

        async def _passthrough(fn, *args, **kwargs):
            return await fn(*args, **kwargs)

        instance.execute_with_retry = AsyncMock(side_effect=_passthrough)
        instance.concurrency_slot.side_effect = _slot
        yield instance


@pytest.fixture
def open_router_provider(open_router_config):
    return OpenRouterProvider(open_router_config)


def test_init(open_router_config):
    """Test provider initialization."""
    with patch("httpx.AsyncClient") as mock_client:
        provider = OpenRouterProvider(open_router_config)
        assert provider._api_key == "test_openrouter_key"
        assert provider._base_url == "https://openrouter.ai/api/v1"
        mock_client.assert_called_once()


def test_init_uses_configurable_timeouts():
    """Provider passes configurable read/write/connect timeouts to httpx."""
    config = ProviderConfig(
        api_key="test_openrouter_key",
        base_url="https://openrouter.ai/api/v1",
        http_read_timeout=600.0,
        http_write_timeout=15.0,
        http_connect_timeout=5.0,
    )
    with patch("httpx.AsyncClient") as mock_client:
        OpenRouterProvider(config)
        timeout = mock_client.call_args.kwargs["timeout"]
        assert timeout.read == 600.0
        assert timeout.write == 15.0
        assert timeout.connect == 5.0


def test_build_request_body_is_native_anthropic(open_router_provider):
    req = MockRequest()
    body = open_router_provider._build_request_body(req)

    assert body["model"] == "stepfun/step-3.5-flash:free"
    assert body["temperature"] == 0.5
    assert body["stream"] is True
    assert body["messages"] == [{"role": "user", "content": "Hello"}]
    assert body["system"] == "System prompt"
    assert body["reasoning"] == {"enabled": True}
    assert "extra_body" not in body
    assert "original_model" not in body
    assert "resolved_provider_model" not in body


def test_build_request_body_omits_reasoning_when_globally_disabled(
    open_router_config,
):
    provider = OpenRouterProvider(
        open_router_config.model_copy(update={"enable_thinking": False})
    )

    body = provider._build_request_body(MockRequest())

    assert "reasoning" not in body


def test_build_request_body_omits_reasoning_when_request_disables_thinking(
    open_router_provider,
):
    req = MockRequest()
    req.thinking.enabled = False

    body = open_router_provider._build_request_body(req)

    assert "reasoning" not in body


def test_build_request_body_omits_reasoning_when_native_thinking_disabled(
    open_router_provider,
):
    req = MockRequest(thinking={"type": "disabled"})

    body = open_router_provider._build_request_body(req)

    assert "reasoning" not in body


def test_build_request_body_maps_thinking_budget_to_reasoning_max_tokens(
    open_router_provider,
):
    req = MockRequest(thinking={"type": "enabled", "budget_tokens": 4096})

    body = open_router_provider._build_request_body(req)

    assert body["reasoning"] == {"enabled": True, "max_tokens": 4096}


def test_build_request_body_default_max_tokens(open_router_provider):
    req = MockRequest(max_tokens=None)

    body = open_router_provider._build_request_body(req)

    assert body["max_tokens"] == OPENROUTER_DEFAULT_MAX_TOKENS
    assert body["max_tokens"] == 81920


def test_build_request_body_strips_unsigned_thinking_history(open_router_provider):
    req = MockRequest(
        messages=[
            MockMessage("user", "hello"),
            MockMessage(
                "assistant",
                [
                    {"type": "thinking", "thinking": "hidden"},
                    {"type": "redacted_thinking", "data": "opaque"},
                    {"type": "text", "text": "Hello"},
                ],
            ),
            MockMessage("user", "can you think hard about 2+2"),
        ]
    )

    body = open_router_provider._build_request_body(req)

    assert body["messages"][1]["content"] == [{"type": "text", "text": "Hello"}]


def test_build_request_body_preserves_signed_thinking_history(open_router_provider):
    req = MockRequest(
        messages=[
            MockMessage(
                "assistant",
                [
                    {
                        "type": "thinking",
                        "thinking": "signed",
                        "signature": "sig_123",
                    }
                ],
            )
        ]
    )

    body = open_router_provider._build_request_body(req)

    assert body["messages"][0]["content"] == [
        {"type": "thinking", "thinking": "signed", "signature": "sig_123"}
    ]


def test_build_request_body_flattens_system_blocks(open_router_provider):
    req = MockRequest(
        system=[
            {"type": "text", "text": "First system block."},
            {"type": "text", "text": "Second system block."},
        ]
    )

    body = open_router_provider._build_request_body(req)

    assert body["system"] == "First system block.\n\nSecond system block."


@pytest.mark.asyncio
async def test_stream_response_passes_native_sse_events(open_router_provider):
    req = MockRequest()
    response = FakeResponse(
        lines=[
            "event: message_start",
            'data: {"type":"message_start","message":{}}',
            "",
            "event: content_block_delta",
            'data: {"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"Hello"}}',
            "",
            "event: message_stop",
            'data: {"type":"message_stop"}',
            "",
            "event: data",
            "data: [DONE]",
            "",
        ]
    )

    with (
        patch.object(open_router_provider._client, "build_request") as mock_build,
        patch.object(
            open_router_provider._client,
            "send",
            new_callable=AsyncMock,
            return_value=response,
        ),
    ):
        events = [e async for e in open_router_provider.stream_response(req)]

    _, kwargs = mock_build.call_args
    assert kwargs["headers"]["Authorization"] == "Bearer test_openrouter_key"
    assert kwargs["headers"]["anthropic-version"] == "2023-06-01"
    assert events[0].startswith("event: message_start")
    assert events[-1].startswith("event: message_stop")
    assert any("Hello" in event for event in events)
    assert "[DONE]" not in "".join(events)
    assert response.is_closed


@pytest.mark.asyncio
async def test_stream_response_suppresses_native_thinking_when_disabled(
    open_router_config,
):
    provider = OpenRouterProvider(
        open_router_config.model_copy(update={"enable_thinking": False})
    )
    response = FakeResponse(
        lines=[
            "event: content_block_start",
            'data: {"type":"content_block_start","index":0,"content_block":{"type":"thinking","thinking":""}}',
            "",
            "event: content_block_delta",
            'data: {"type":"content_block_delta","index":0,"delta":{"type":"thinking_delta","thinking":"secret"}}',
            "",
            "event: content_block_stop",
            'data: {"type":"content_block_stop","index":0}',
            "",
            "event: content_block_start",
            'data: {"type":"content_block_start","index":1,"content_block":{"type":"text","text":""}}',
            "",
            "event: content_block_delta",
            'data: {"type":"content_block_delta","index":1,"delta":{"type":"text_delta","text":"Answer"}}',
            "",
            "event: content_block_stop",
            'data: {"type":"content_block_stop","index":1}',
            "",
        ]
    )

    with (
        patch.object(provider._client, "build_request"),
        patch.object(
            provider._client,
            "send",
            new_callable=AsyncMock,
            return_value=response,
        ),
    ):
        events = [e async for e in provider.stream_response(MockRequest())]

    event_text = "".join(events)
    assert "thinking_delta" not in event_text
    assert "secret" not in event_text
    assert "Answer" in event_text

    text_start = next(event for event in events if "content_block_start" in event)
    payload = json.loads(text_start.split("data: ", 1)[1])
    assert payload["index"] == 0


@pytest.mark.asyncio
async def test_stream_response_drops_redacted_thinking_when_enabled(
    open_router_provider,
):
    response = FakeResponse(
        lines=[
            "event: content_block_start",
            'data: {"type":"content_block_start","index":0,"content_block":{"type":"redacted_thinking","data":"opaque"}}',
            "",
            "event: content_block_stop",
            'data: {"type":"content_block_stop","index":0}',
            "",
            "event: content_block_start",
            'data: {"type":"content_block_start","index":1,"content_block":{"type":"text","text":""}}',
            "",
            "event: content_block_delta",
            'data: {"type":"content_block_delta","index":1,"delta":{"type":"text_delta","text":"Answer"}}',
            "",
            "event: content_block_stop",
            'data: {"type":"content_block_stop","index":1}',
            "",
            "event: message_stop",
            'data: {"type":"message_stop"}',
            "",
        ]
    )

    with (
        patch.object(open_router_provider._client, "build_request"),
        patch.object(
            open_router_provider._client,
            "send",
            new_callable=AsyncMock,
            return_value=response,
        ),
    ):
        events = [e async for e in open_router_provider.stream_response(MockRequest())]

    event_text = "".join(events)
    assert "redacted_thinking" not in event_text
    assert "Answer" in event_text

    start_event = next(event for event in events if "content_block_start" in event)
    payload = json.loads(start_event.split("data: ", 1)[1])
    assert payload["index"] == 0
    assert payload["content_block"]["type"] == "text"


@pytest.mark.asyncio
async def test_stream_response_closes_overlapping_thinking_before_text(
    open_router_provider,
):
    response = FakeResponse(
        lines=[
            "event: content_block_start",
            'data: {"type":"content_block_start","index":0,"content_block":{"type":"thinking","thinking":"","signature":""}}',
            "",
            "event: content_block_delta",
            'data: {"type":"content_block_delta","index":0,"delta":{"type":"thinking_delta","thinking":"reason"}}',
            "",
            "event: content_block_start",
            'data: {"type":"content_block_start","index":1,"content_block":{"type":"text","text":""}}',
            "",
            "event: content_block_delta",
            'data: {"type":"content_block_delta","index":1,"delta":{"type":"text_delta","text":"Answer"}}',
            "",
            "event: content_block_stop",
            'data: {"type":"content_block_stop","index":1}',
            "",
            "event: content_block_stop",
            'data: {"type":"content_block_stop","index":0}',
            "",
        ]
    )

    with (
        patch.object(open_router_provider._client, "build_request"),
        patch.object(
            open_router_provider._client,
            "send",
            new_callable=AsyncMock,
            return_value=response,
        ),
    ):
        events = [e async for e in open_router_provider.stream_response(MockRequest())]

    event_text = "".join(events)
    thinking_stop = event_text.index('"type": "content_block_stop", "index": 0')
    text_start = event_text.index('"content_block": {"type": "text"')
    assert thinking_stop < text_start
    assert event_text.count('"index": 0') == 3
    assert "Answer" in event_text


@pytest.mark.asyncio
async def test_stream_response_error_path(open_router_provider):
    req = MockRequest()

    with (
        patch.object(open_router_provider._client, "build_request"),
        patch.object(
            open_router_provider._client,
            "send",
            new_callable=AsyncMock,
            side_effect=RuntimeError("API failed"),
        ),
    ):
        events = [e async for e in open_router_provider.stream_response(req)]

    event_text = "".join(events)
    assert "message_start" in event_text
    assert "API failed" in event_text
    assert "message_stop" in event_text
