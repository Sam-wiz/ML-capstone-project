"""Helpers for parsing structured LLM responses."""

import json


def clean_json_response(raw: str) -> str:
    """
    Strip common Markdown/code-fence wrappers and recover the outer JSON object.
    Uses json.JSONDecoder.raw_decode to find proper JSON boundaries rather than
    naive brace-matching, which fails on strings containing '}' inside values.
    """
    text = (raw or "").strip()

    # Strip code fences: ```json ... ``` or ``` ... ```
    if text.startswith("```"):
        first_newline = text.find("\n")
        if first_newline != -1:
            text = text[first_newline + 1:]
        if text.endswith("```"):
            text = text[:-3]
        text = text.strip()

    # Use raw_decode to extract the first valid JSON object
    start = text.find("{")
    if start != -1:
        decoder = json.JSONDecoder()
        try:
            _, end_idx = decoder.raw_decode(text, start)
            return text[start:end_idx].strip()
        except json.JSONDecodeError:
            pass

    # Fallback: naive brace matching (handles truncated/partial responses)
    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1 and end > start:
        return text[start:end + 1].strip()

    return text.strip()

