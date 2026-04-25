"""FastAPI application factory and configuration."""

import os
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI, Request
from fastapi.exception_handlers import request_validation_exception_handler
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from loguru import logger

from config.logging_config import configure_logging
from config.settings import get_settings
from providers.exceptions import ProviderError

from .routes import router
from .runtime import AppRuntime

# Opt-in to future behavior for python-telegram-bot
os.environ["PTB_TIMEDELTA"] = "1"

# Configure logging first (before any module logs)
_settings = get_settings()
configure_logging(_settings.log_file)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager."""
    runtime = AppRuntime.for_app(app, settings=get_settings())
    await runtime.startup()

    yield

    await runtime.shutdown()


def create_app() -> FastAPI:
    """Create and configure the FastAPI application."""
    app = FastAPI(
        title="Claude Code Proxy",
        version="2.0.0",
        lifespan=lifespan,
    )

    # Register routes
    app.include_router(router)

    # Exception handlers
    @app.exception_handler(RequestValidationError)
    async def validation_error_handler(request: Request, exc: RequestValidationError):
        """Log request shape for 422 debugging without content values."""
        body: Any
        try:
            body = await request.json()
        except Exception as e:
            body = {"_json_error": type(e).__name__}

        messages = body.get("messages") if isinstance(body, dict) else None
        message_summary: list[dict[str, Any]] = []
        if isinstance(messages, list):
            for msg in messages:
                if not isinstance(msg, dict):
                    message_summary.append({"message_kind": type(msg).__name__})
                    continue
                content = msg.get("content")
                item: dict[str, Any] = {
                    "role": msg.get("role"),
                    "content_kind": type(content).__name__,
                }
                if isinstance(content, list):
                    item["block_types"] = [
                        block.get("type", "dict")
                        if isinstance(block, dict)
                        else type(block).__name__
                        for block in content[:12]
                    ]
                    item["block_keys"] = [
                        sorted(str(key) for key in block)[:12]
                        for block in content[:5]
                        if isinstance(block, dict)
                    ]
                elif isinstance(content, str):
                    item["content_length"] = len(content)
                message_summary.append(item)

        logger.debug(
            "Request validation failed: path={} query={} error_locs={} error_types={} message_summary={} tool_names={}",
            request.url.path,
            str(request.url.query),
            [list(error.get("loc", ())) for error in exc.errors()],
            [str(error.get("type", "")) for error in exc.errors()],
            message_summary,
            [
                str(tool.get("name", ""))
                for tool in body.get("tools", [])
                if isinstance(body, dict)
                and isinstance(body.get("tools"), list)
                and isinstance(tool, dict)
            ],
        )
        return await request_validation_exception_handler(request, exc)

    @app.exception_handler(ProviderError)
    async def provider_error_handler(request: Request, exc: ProviderError):
        """Handle provider-specific errors and return Anthropic format."""
        logger.error(f"Provider Error: {exc.error_type} - {exc.message}")
        return JSONResponse(
            status_code=exc.status_code,
            content=exc.to_anthropic_format(),
        )

    @app.exception_handler(Exception)
    async def general_error_handler(request: Request, exc: Exception):
        """Handle general errors and return Anthropic format."""
        logger.error(f"General Error: {exc!s}")
        import traceback

        logger.error(traceback.format_exc())
        return JSONResponse(
            status_code=500,
            content={
                "type": "error",
                "error": {
                    "type": "api_error",
                    "message": "An unexpected error occurred.",
                },
            },
        )

    return app


# Default app instance for uvicorn
app = create_app()
