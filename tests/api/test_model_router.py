from unittest.mock import patch

import pytest

from api.model_router import ModelRouter
from api.models.anthropic import Message, MessagesRequest, TokenCountRequest
from config.settings import Settings


@pytest.fixture
def settings():
    settings = Settings()
    settings.model = "nvidia_nim/fallback-model"
    settings.model_opus = None
    settings.model_sonnet = None
    settings.model_haiku = None
    return settings


def test_model_router_resolves_default_model(settings):
    resolved = ModelRouter(settings).resolve("claude-3-opus")

    assert resolved.original_model == "claude-3-opus"
    assert resolved.provider_id == "nvidia_nim"
    assert resolved.provider_model == "fallback-model"
    assert resolved.provider_model_ref == "nvidia_nim/fallback-model"


def test_model_router_applies_opus_override(settings):
    settings.model_opus = "open_router/deepseek/deepseek-r1"

    request = MessagesRequest(
        model="claude-opus-4-20250514",
        max_tokens=100,
        messages=[Message(role="user", content="hello")],
    )
    routed = ModelRouter(settings).resolve_messages_request(request)

    assert routed.request.model == "deepseek/deepseek-r1"
    assert routed.resolved.provider_model_ref == "open_router/deepseek/deepseek-r1"
    assert routed.resolved.original_model == "claude-opus-4-20250514"
    assert request.model == "claude-opus-4-20250514"


def test_model_router_applies_haiku_override(settings):
    settings.model_haiku = "lmstudio/qwen2.5-7b"

    routed = ModelRouter(settings).resolve_messages_request(
        MessagesRequest(
            model="claude-3-haiku-20240307",
            max_tokens=100,
            messages=[Message(role="user", content="hello")],
        )
    )

    assert routed.request.model == "qwen2.5-7b"
    assert routed.resolved.provider_model_ref == "lmstudio/qwen2.5-7b"


def test_model_router_applies_sonnet_override(settings):
    settings.model_sonnet = "nvidia_nim/meta/llama-3.3-70b-instruct"

    routed = ModelRouter(settings).resolve_messages_request(
        MessagesRequest(
            model="claude-sonnet-4-20250514",
            max_tokens=100,
            messages=[Message(role="user", content="hello")],
        )
    )

    assert routed.request.model == "meta/llama-3.3-70b-instruct"
    assert (
        routed.resolved.provider_model_ref == "nvidia_nim/meta/llama-3.3-70b-instruct"
    )


def test_model_router_routes_token_count_request(settings):
    settings.model_haiku = "lmstudio/qwen2.5-7b"

    request = TokenCountRequest(
        model="claude-3-haiku-20240307",
        messages=[Message(role="user", content="hello")],
    )
    routed = ModelRouter(settings).resolve_token_count_request(request)

    assert routed.request.model == "qwen2.5-7b"
    assert request.model == "claude-3-haiku-20240307"


def test_model_router_logs_mapping(settings):
    with patch("api.model_router.logger.debug") as mock_log:
        ModelRouter(settings).resolve("claude-2.1")

    mock_log.assert_called()
    args = mock_log.call_args[0]
    assert "MODEL MAPPING" in args[0]
    assert args[1] == "claude-2.1"
    assert args[2] == "fallback-model"


def test_resolve_model_strips_classifier_suffix(settings):
    """Test that [1m] suffix is stripped for model routing."""
    settings.model_opus = "open_router/opus-model"
    settings.model_sonnet = "nvidia_nim/sonnet-model"
    settings.model_haiku = "lmstudio/haiku-model"

    # Test all model types with [1m] suffix
    assert Settings._strip_classifier_suffix("claude-opus-4[1m]") == "claude-opus-4"
    assert Settings._strip_classifier_suffix("claude-3-5-sonnet[1m]") == "claude-3-5-sonnet"
    assert Settings._strip_classifier_suffix("claude-3-haiku[1m]") == "claude-3-haiku"
    assert Settings._strip_classifier_suffix("claude-sonnet-4-20250514[1m]") == "claude-sonnet-4-20250514"
    assert Settings._strip_classifier_suffix("claude-opus-4") == "claude-opus-4"


def test_model_router_resolves_with_classifier_suffix(settings):
    """Test that model routing works with [1m] classifier suffix."""
    settings.model_opus = "open_router/opus-model"
    settings.model_sonnet = "nvidia_nim/sonnet-model"
    settings.model_haiku = "lmstudio/haiku-model"

    router = ModelRouter(settings)

    # Test Opus with [1m]
    resolved = router.resolve("claude-opus-4[1m]")
    assert resolved.provider_model_ref == "open_router/opus-model"
    assert resolved.original_model == "claude-opus-4[1m]"

    # Test Sonnet with [1m]
    resolved = router.resolve("claude-sonnet-4-20250514[1m]")
    assert resolved.provider_model_ref == "nvidia_nim/sonnet-model"
    assert resolved.original_model == "claude-sonnet-4-20250514[1m]"

    # Test Haiku with [1m]
    resolved = router.resolve("claude-3-haiku[1m]")
    assert resolved.provider_model_ref == "lmstudio/haiku-model"
    assert resolved.original_model == "claude-3-haiku[1m]"


def test_model_router_messages_request_with_classifier_suffix(settings):
    """Test that messages request routing preserves original model with [1m] suffix."""
    settings.model_sonnet = "nvidia_nim/sonnet-model"

    request = MessagesRequest(
        model="claude-sonnet-4-20250514[1m]",
        max_tokens=100,
        messages=[Message(role="user", content="hello")],
    )
    routed = ModelRouter(settings).resolve_messages_request(request)

    # Routed model should be the provider model (without [1m])
    assert routed.request.model == "sonnet-model"
    # Resolved should preserve original model with [1m]
    assert routed.resolved.original_model == "claude-sonnet-4-20250514[1m]"
    assert routed.resolved.provider_model_ref == "nvidia_nim/sonnet-model"
    # Original request unchanged
    assert request.model == "claude-sonnet-4-20250514[1m]"
