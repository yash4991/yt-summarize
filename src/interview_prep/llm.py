"""Build the LLM for a model setting: a cloud model, or the local LM Studio server.

``local`` / ``local/<model-id>`` is routed through crewai's OpenAI client to the
OpenAI-compatible server at ``LOCAL_BASE_URL`` (LM Studio, but Ollama, vLLM or
llama.cpp servers work the same way).
"""

from __future__ import annotations

import json
import logging
import urllib.error
import urllib.request
from functools import lru_cache

from .config import Settings, is_local

log = logging.getLogger(__name__)


class LocalModelError(RuntimeError):
    pass


@lru_cache(maxsize=8)
def _loaded_models(base_url: str, api_key: str) -> tuple[str, ...]:
    req = urllib.request.Request(base_url.rstrip("/") + "/models", headers={"Authorization": f"Bearer {api_key}"})
    try:
        with urllib.request.urlopen(req, timeout=5) as resp:
            data = json.load(resp)
    except (urllib.error.URLError, OSError, ValueError) as e:
        raise LocalModelError(
            f"Can't reach the local model server at {base_url} ({e}). Start it in LM Studio "
            "(Developer tab -> Start server, or run: lms server start) and check LOCAL_BASE_URL in .env."
        ) from e
    ids = [m.get("id", "") for m in data.get("data", [])]
    return tuple(i for i in ids if i and "embed" not in i.lower())


def local_model_id(settings: Settings, model: str | None = None) -> str:
    """The model id to send to the local server."""
    model = model or settings.model
    if model.startswith("local/"):
        return model.split("/", 1)[1]
    if settings.local_model:
        return settings.local_model
    ids = _loaded_models(settings.local_base_url, settings.local_api_key)
    if not ids:
        raise LocalModelError("The local server is running but has no chat model loaded. Load one in LM Studio.")
    return ids[0]


def local_server_problem(settings: Settings, model: str | None = None) -> str | None:
    """A readable problem description if the local model can't be used, else None."""
    _loaded_models.cache_clear()
    try:
        mid = local_model_id(settings, model)
    except LocalModelError as e:
        return str(e)
    ids = _loaded_models(settings.local_base_url, settings.local_api_key)
    if ids and mid not in ids:
        return f"Model '{mid}' is not available on the local server. Available: {', '.join(ids)}."
    return None


def model_key(settings: Settings, model: str | None = None) -> str:
    """Stable name for caching and reports, e.g. 'local/prism-ml/bonsai-27b'."""
    model = model or settings.model
    return f"local/{local_model_id(settings, model)}" if is_local(model) else model


def make_llm(settings: Settings, model: str | None = None, temperature: float = 0.2):
    from crewai import LLM

    model = model or settings.model
    max_tokens = settings.limits(model).max_output_tokens
    if not is_local(model):
        return LLM(model=model, temperature=temperature, max_tokens=max_tokens)
    kw = {}
    if settings.local_reasoning_effort:
        kw["reasoning_effort"] = settings.local_reasoning_effort
    return LLM(
        model=f"openai/{local_model_id(settings, model)}",
        base_url=settings.local_base_url,
        api_key=settings.local_api_key,
        temperature=temperature,
        max_tokens=max_tokens,
        timeout=settings.local_timeout_seconds,
        **kw,
    )
