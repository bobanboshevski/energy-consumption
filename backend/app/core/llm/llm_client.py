"""Thin, reusable HTTP transport for the OpenAI-compatible LLM endpoint.
Domain code never touches HTTP details directly — it calls chat_completion()."""
import requests

from app.core.config import settings


def chat_completion(
        messages: list[dict], *, temperature: float = 0.1, json_mode: bool = True, timeout: int = 240,
) -> dict:
    """
    Calls the chat-completions endpoint and returns the raw decoded JSON response.

    Raises:
        requests.RequestException — on any network/HTTP failure (caller maps it).
    """
    headers = {
        "Authorization": f"Bearer {settings.LLM_API_KEY}",
        "Content-Type": "application/json",
    }
    body = {
        "model": settings.LLM_MODEL_NAME,
        "temperature": temperature,
        # "max_tokens": max_tokens,
        "messages": messages,
    }
    if json_mode:
        body["response_format"] = {"type": "json_object"}

    response = requests.post(settings.LLM_ENDPOINT_URL, headers=headers, json=body, timeout=timeout)
    response.raise_for_status()
    return response.json()
