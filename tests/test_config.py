"""Settings: defaults, environment overrides and early, readable failures."""
import pytest

from app.config import DEFAULT_EMBEDDING_MODEL, Settings


def test_defaults():
    settings = Settings.from_env()
    assert settings.embedding_backend == "auto"
    assert settings.embedding_model == DEFAULT_EMBEDDING_MODEL
    assert settings.llm_provider == "auto"
    assert (settings.chunk_size, settings.chunk_overlap, settings.top_k) == (180, 30, 4)
    assert settings.max_upload_mb == 10
    assert settings.min_score is None
    assert settings.api_key is None
    assert settings.llm_model is None
    assert str(settings.data_dir) == "data"


def test_environment_variables_override_the_defaults(monkeypatch):
    monkeypatch.setenv("DOCQA_TOP_K", "7")
    monkeypatch.setenv("DOCQA_CHUNK_SIZE", "100")
    monkeypatch.setenv("DOCQA_CHUNK_OVERLAP", "10")
    monkeypatch.setenv("DOCQA_MIN_SCORE", "0.35")
    monkeypatch.setenv("DOCQA_API_KEY", "  secret  ")  # surrounding spaces are trimmed
    monkeypatch.setenv("DOCQA_EMBEDDING_BACKEND", "HASHING")  # case-insensitive
    monkeypatch.setenv("DOCQA_DATA_DIR", "/tmp/docqa-index")
    settings = Settings.from_env()
    assert settings.top_k == 7
    assert (settings.chunk_size, settings.chunk_overlap) == (100, 10)
    assert settings.min_score == pytest.approx(0.35)
    assert settings.api_key == "secret"
    assert settings.embedding_backend == "hashing"
    assert settings.data_dir.as_posix() == "/tmp/docqa-index"


def test_blank_values_count_as_unset(monkeypatch):
    monkeypatch.setenv("DOCQA_TOP_K", "   ")
    monkeypatch.setenv("DOCQA_API_KEY", "")
    settings = Settings.from_env()
    assert settings.top_k == 4
    assert settings.api_key is None


@pytest.mark.parametrize(
    ("name", "value"),
    [
        ("TOP_K", "abc"),
        ("TOP_K", "0"),
        ("TOP_K", "21"),
        ("CHUNK_SIZE", "5"),
        ("CHUNK_SIZE", "19"),
        ("CHUNK_OVERLAP", "999"),
        ("MAX_UPLOAD_MB", "0"),
        ("LLM_TIMEOUT", "-1"),
        ("EMBEDDING_BACKEND", "magic"),
        ("LLM_PROVIDER", "gpt"),
        ("MIN_SCORE", "high"),
    ],
)
def test_invalid_values_fail_and_name_the_offending_variable(monkeypatch, name, value):
    monkeypatch.setenv(f"DOCQA_{name}", value)
    with pytest.raises(ValueError, match=f"DOCQA_{name}"):
        Settings.from_env()


def test_overlap_must_be_smaller_than_the_chunk_size():
    with pytest.raises(ValueError, match="CHUNK_OVERLAP"):
        Settings(chunk_size=50, chunk_overlap=50)


def test_settings_are_immutable():
    settings = Settings()
    with pytest.raises(AttributeError):
        settings.top_k = 10  # type: ignore[misc]


def test_the_smallest_allowed_chunk_size_is_twenty_words():
    assert Settings(chunk_size=20, chunk_overlap=5).chunk_size == 20
