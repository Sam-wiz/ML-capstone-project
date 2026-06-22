"""Runtime configuration helpers."""

import os


def has_openai_api_key() -> bool:
    """Return True when a real OpenAI API key appears to be configured."""
    key = os.getenv("OPENAI_API_KEY", "").strip()
    return bool(key and key != "sk-..." and not key.startswith("your_"))

