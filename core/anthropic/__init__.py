"""Anthropic protocol helpers shared across API, providers, and integrations."""

from .content import extract_text_from_content, get_block_attr, get_block_type
from .conversion import AnthropicToOpenAIConverter, build_base_request_body
from .errors import append_request_id, get_user_facing_error_message
from .sse import ContentBlockManager, SSEBuilder, map_stop_reason
from .thinking import ContentChunk, ContentType, ThinkTagParser
from .tokens import get_token_count
from .tools import HeuristicToolParser
from .utils import set_if_not_none

__all__ = [
    "AnthropicToOpenAIConverter",
    "ContentBlockManager",
    "ContentChunk",
    "ContentType",
    "HeuristicToolParser",
    "SSEBuilder",
    "ThinkTagParser",
    "append_request_id",
    "build_base_request_body",
    "extract_text_from_content",
    "get_block_attr",
    "get_block_type",
    "get_token_count",
    "get_user_facing_error_message",
    "map_stop_reason",
    "set_if_not_none",
]
