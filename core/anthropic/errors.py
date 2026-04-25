"""User-facing error formatting shared by API, providers, and integrations."""

import httpx
import openai


def get_user_facing_error_message(
    e: Exception,
    *,
    read_timeout_s: float | None = None,
) -> str:
    """Return a readable, non-empty error message for users."""
    message = str(e).strip()
    if message:
        return message

    if isinstance(e, httpx.ReadTimeout):
        if read_timeout_s is not None:
            return f"Provider request timed out after {read_timeout_s:g}s."
        return "Provider request timed out."
    if isinstance(e, httpx.ConnectTimeout):
        return "Could not connect to provider."
    if isinstance(e, TimeoutError):
        if read_timeout_s is not None:
            return f"Provider request timed out after {read_timeout_s:g}s."
        return "Request timed out."

    name = type(e).__name__
    status_code = getattr(e, "status_code", None)
    if isinstance(e, openai.RateLimitError) or name == "RateLimitError":
        return "Provider rate limit reached. Please retry shortly."
    if isinstance(e, openai.AuthenticationError) or name == "AuthenticationError":
        return "Provider authentication failed. Check API key."
    if isinstance(e, openai.BadRequestError) or name == "InvalidRequestError":
        return "Invalid request sent to provider."
    if name == "OverloadedError":
        return "Provider is currently overloaded. Please retry."
    if name == "APIError":
        if status_code in (502, 503, 504):
            return "Provider is temporarily unavailable. Please retry."
        return "Provider API request failed."
    if name.endswith("ProviderError") or name == "ProviderError":
        return "Provider request failed."

    return "Provider request failed unexpectedly."


def append_request_id(message: str, request_id: str | None) -> str:
    """Append request_id suffix when available."""
    base = message.strip() or "Provider request failed unexpectedly."
    if request_id:
        return f"{base} (request_id={request_id})"
    return base
