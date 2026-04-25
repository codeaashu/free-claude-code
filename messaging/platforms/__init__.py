"""Messaging platform adapters (Telegram, Discord, etc.)."""

from .base import CLISession, MessagingPlatform, SessionManagerInterface
from .factory import create_messaging_platform

__all__ = [
    "CLISession",
    "MessagingPlatform",
    "SessionManagerInterface",
    "create_messaging_platform",
]
