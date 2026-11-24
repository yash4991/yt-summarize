"""Local model (LM Studio) support: routing, limits, and falling back to it when the cloud quota runs out."""

import pytest

from interview_prep import flow as flow_mod
from interview_prep import llm as llm_mod
from interview_prep.config import is_local
from interview_prep.flow import process_video
from interview_prep.models import RunStats
from interview_prep.retry import QuotaExceeded
from interview_prep.translate import strip_reasoning

from .conftest import make_pack


@pytest.fixture
def local_server(monkeypatch):
    """Pretend LM Studio is running with one chat model (and an embedding model) loaded."""
    llm_mod._loaded_models.cache_clear()
    monkeypatch.setattr(llm_mod, "_loaded_models", lambda url, key: ("prism-ml/bonsai-27b",))
    yield


def test_is_local():
    assert is_local("local") and is_local("local/qwen") and not is_local("gemini/gemini-2.5-flash")


def test_local_llm_points_at_lm_studio(settings, local_server):
    settings.local_base_url = "http://localhost:1235/v1"
    llm = llm_mod.make_llm(settings, "local", temperature=0.1)
    assert type(llm).__name__ == "OpenAICompletion"
    assert llm.model == "prism-ml/bonsai-27b"
    assert str(llm.base_url).startswith("http://localhost:1235/v1")
    assert llm.reasoning_effort == "none"  # hidden thinking is off
    assert llm.max_tokens == settings.local_context_tokens // 2


def test_explicit_local_model_id(settings, local_server):
    assert llm_mod.local_model_id(settings, "local/other-model") == "other-model"
    assert llm_mod.model_key(settings, "local") == "local/prism-ml/bonsai-27b"
    assert llm_mod.model_key(settings, "gemini/gemini-2.5-flash") == "gemini/gemini-2.5-flash"


def test_server_down_message(settings, monkeypatch):
    llm_mod._loaded_models.cache_clear()
    settings.local_base_url = "http://127.0.0.1:9/v1"  # nothing listens on port 9
    problem = llm_mod.local_server_problem(settings)
    assert "Can't reach the local model server" in problem and "lms server start" in problem


def test_local_limits_fit_small_context(settings):
    cloud, local = settings.limits("gemini/gemini-2.5-flash"), settings.limits("local")
    assert local.max_transcript_chars < cloud.max_transcript_chars
    assert local.parallel == 1 and cloud.parallel == 4
    settings.local_context_tokens = 32768
    assert settings.limits("local").max_transcript_chars == 2 * local.max_transcript_chars


def test_strip_reasoning():
    assert strip_reasoning("<think>hmm [00:01] no</think>\n[00:05] Real line") == "\n[00:05] Real line"


# --------------------------------------------------------------------------- #
# Flow: local as the main model, and as the fallback
# --------------------------------------------------------------------------- #


@pytest.fixture
def offline(monkeypatch, meta, transcript):
    monkeypatch.setattr(flow_mod, "fetch_metadata", lambda vid: meta)
    monkeypatch.setattr(flow_mod, "fetch_transcript", lambda vid, **kw: transcript)


def test_local_as_main_model_condenses_to_fit(offline, settings, local_server, fake_runner, monkeypatch):
    import interview_prep.crew as crew_pkg

    settings.model = "local"
    settings.local_context_tokens = 512  # tiny: forces condensing
    monkeypatch.setattr(crew_pkg, "condense_chunk", lambda cfg, model, chunk, i, n: f"[00:00] notes {i}")
    r = process_video("abcdefghijk", settings, crew_runner=fake_runner)
    assert fake_runner.calls[0].startswith("[00:00] notes 1")
    assert r.stats.model == "local/prism-ml/bonsai-27b"


def test_quota_falls_back_to_local_for_rest_of_run(offline, settings, local_server):
    settings.model, settings.fallback_model = "gemini/gemini-2.5-flash", "local"
    used = []

    def runner(cfg, meta, text, progress, transcript_note=""):
        used.append(cfg.model)
        if cfg.model.startswith("gemini"):
            raise QuotaExceeded("Your API plan's daily request limit is used up.")
        return make_pack(), RunStats(model=cfg.model, prompt_version="x")

    r = process_video("abcdefghijk", settings, crew_runner=runner)
    assert used == ["gemini/gemini-2.5-flash", "local"]
    assert r.stats.model == "local/prism-ml/bonsai-27b"
    assert settings.model == "local"  # sticky: the next video goes straight to local


def test_quota_without_fallback_still_fails(offline, settings):
    def runner(cfg, meta, text, progress, transcript_note=""):
        raise QuotaExceeded("used up")

    with pytest.raises(QuotaExceeded):
        process_video("abcdefghijk", settings, crew_runner=runner)
