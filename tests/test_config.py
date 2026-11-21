from interview_prep.config import DEFAULT_MODEL, Settings, missing_api_key


def test_default_model_is_gemini_flash(monkeypatch):
    monkeypatch.delenv("LLM_MODEL", raising=False)
    assert DEFAULT_MODEL == "gemini/gemini-2.5-flash"
    assert Settings().model == DEFAULT_MODEL


def test_gemini_accepts_either_key(monkeypatch):
    for var in ("GEMINI_API_KEY", "GOOGLE_API_KEY"):
        monkeypatch.delenv(var, raising=False)
    assert missing_api_key("gemini/gemini-2.5-flash") == "GEMINI_API_KEY or GOOGLE_API_KEY"
    monkeypatch.setenv("GOOGLE_API_KEY", "x")
    assert missing_api_key("gemini/gemini-2.5-flash") is None
    monkeypatch.delenv("GOOGLE_API_KEY")
    monkeypatch.setenv("GEMINI_API_KEY", "x")
    assert missing_api_key("gemini/gemini-2.5-flash") is None


def test_other_providers(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    assert missing_api_key("anthropic/claude-sonnet-5-5") == "ANTHROPIC_API_KEY"
    assert missing_api_key("ollama/llama3") is None  # local models need no key
