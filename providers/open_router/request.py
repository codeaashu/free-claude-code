"""Native Anthropic Messages request builder for OpenRouter."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from loguru import logger
from pydantic import BaseModel

OPENROUTER_DEFAULT_MAX_TOKENS = 81920

_REQUEST_FIELDS = (
    "model",
    "messages",
    "system",
    "max_tokens",
    "stop_sequences",
    "stream",
    "temperature",
    "top_p",
    "top_k",
    "metadata",
    "tools",
    "tool_choice",
    "thinking",
    "extra_body",
    "original_model",
    "resolved_provider_model",
)

_INTERNAL_FIELDS = {
    "thinking",
    "extra_body",
    "original_model",
    "resolved_provider_model",
}
_THINKING_HISTORY_BLOCK_TYPES = {"thinking", "redacted_thinking"}


def _serialize_value(value: Any) -> Any:
    """Convert Pydantic models and lightweight objects into JSON-ready values."""
    if isinstance(value, BaseModel):
        return value.model_dump(exclude_none=True)
    if isinstance(value, dict):
        return {
            key: _serialize_value(item)
            for key, item in value.items()
            if item is not None
        }
    if isinstance(value, Sequence) and not isinstance(value, str | bytes | bytearray):
        return [_serialize_value(item) for item in value]
    if value is None or isinstance(value, str | int | float | bool):
        return value
    if hasattr(value, "__dict__"):
        return {
            key: _serialize_value(item)
            for key, item in vars(value).items()
            if not key.startswith("_") and item is not None
        }
    return value


def _dump_request_fields(request_data: Any) -> dict[str, Any]:
    """Extract the public request fields forwarded to OpenRouter."""
    if isinstance(request_data, BaseModel):
        return request_data.model_dump(exclude_none=True)

    dumped: dict[str, Any] = {}
    for field in _REQUEST_FIELDS:
        value = getattr(request_data, field, None)
        if value is not None:
            dumped[field] = _serialize_value(value)
    return dumped


def _strip_unsigned_thinking_history(messages: Any) -> Any:
    """Remove assistant thinking history blocks that OpenRouter cannot replay."""
    if not isinstance(messages, list):
        return messages

    sanitized_messages: list[Any] = []
    for message in messages:
        if not isinstance(message, dict):
            sanitized_messages.append(message)
            continue

        content = message.get("content")
        if not isinstance(content, list):
            sanitized_messages.append(message)
            continue

        sanitized_content = [
            block
            for block in content
            if not (
                isinstance(block, dict)
                and block.get("type") in _THINKING_HISTORY_BLOCK_TYPES
                and not isinstance(block.get("signature"), str)
            )
        ]

        sanitized_message = dict(message)
        sanitized_message["content"] = sanitized_content or ""
        sanitized_messages.append(sanitized_message)

    return sanitized_messages


def _normalize_system_prompt(system: Any) -> Any:
    """Flatten Claude SDK system blocks for OpenRouter's native endpoint."""
    if not isinstance(system, list):
        return system

    text_parts: list[str] = []
    for block in system:
        if not isinstance(block, dict):
            continue
        if block.get("type") == "text" and isinstance(block.get("text"), str):
            text_parts.append(block["text"])
    return "\n\n".join(text_parts).strip() if text_parts else system


def _apply_reasoning(body: dict[str, Any], thinking_cfg: Any) -> None:
    """Map Anthropic thinking controls onto OpenRouter reasoning controls."""
    reasoning = body.setdefault("reasoning", {"enabled": True})
    if not isinstance(reasoning, dict):
        return
    reasoning.setdefault("enabled", True)
    if not isinstance(thinking_cfg, dict):
        return
    budget_tokens = thinking_cfg.get("budget_tokens")
    if isinstance(budget_tokens, int):
        reasoning.setdefault("max_tokens", budget_tokens)


def build_request_body(request_data: Any, *, thinking_enabled: bool) -> dict:
    """Build an Anthropic-format request body for OpenRouter's messages API."""
    logger.debug(
        "OPENROUTER_REQUEST: conversion start model={} msgs={}",
        getattr(request_data, "model", "?"),
        len(getattr(request_data, "messages", [])),
    )

    dumped_request = _dump_request_fields(request_data)
    request_extra = dumped_request.pop("extra_body", None)
    thinking_cfg = dumped_request.get("thinking")
    body = {
        key: value
        for key, value in dumped_request.items()
        if key not in _INTERNAL_FIELDS
    }

    if isinstance(request_extra, dict):
        body.update(request_extra)

    body["messages"] = _strip_unsigned_thinking_history(body.get("messages"))
    if "system" in body:
        body["system"] = _normalize_system_prompt(body["system"])
    body["stream"] = True
    if body.get("max_tokens") is None:
        body["max_tokens"] = OPENROUTER_DEFAULT_MAX_TOKENS

    if thinking_enabled:
        _apply_reasoning(body, thinking_cfg)

    logger.debug(
        "OPENROUTER_REQUEST: conversion done model={} msgs={} tools={}",
        body.get("model"),
        len(body.get("messages", [])),
        len(body.get("tools", [])),
    )
    return body
