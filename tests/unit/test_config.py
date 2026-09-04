import textwrap

import pytest

from core.config.loader import load_settings
from core.errors.exceptions import ConfigurationError

DEFAULT_CONFIG = "config/default.yaml"


def write_config(tmp_path, body: str):
    path = tmp_path / "config.yaml"
    path.write_text(textwrap.dedent(body), encoding="utf-8")
    return path


def test_shipped_default_config_is_valid():
    """The config we ship must always load — it is the app's only entry state."""
    settings = load_settings(DEFAULT_CONFIG, local_path=None, apply_env=False)
    assert settings.application.name
    assert settings.video.input_type == "local_file"
    assert settings.buffer.recent_window_seconds > 0


def test_missing_file_raises_configuration_error():
    with pytest.raises(ConfigurationError, match="not found"):
        load_settings("config/does_not_exist.yaml", local_path=None, apply_env=False)


def test_invalid_value_raises_configuration_error(tmp_path, monkeypatch):
    base = (tmp_path / "base.yaml")
    original = open(DEFAULT_CONFIG, encoding="utf-8").read()
    base.write_text(original.replace('device: "cuda"', 'device: "tpu"'), encoding="utf-8")

    with pytest.raises(ConfigurationError):
        load_settings(base, local_path=None, apply_env=False)


def test_unknown_key_is_rejected(tmp_path):
    """A typo must fail loudly rather than silently using a default."""
    original = open(DEFAULT_CONFIG, encoding="utf-8").read()
    base = tmp_path / "base.yaml"
    base.write_text(original + "\nunexpected_section:\n  foo: 1\n", encoding="utf-8")

    with pytest.raises(ConfigurationError):
        load_settings(base, local_path=None, apply_env=False)


def test_local_file_overrides_defaults(tmp_path):
    original = open(DEFAULT_CONFIG, encoding="utf-8").read()
    base = tmp_path / "base.yaml"
    base.write_text(original, encoding="utf-8")

    local = write_config(
        tmp_path,
        """
        runtime:
          device: "cpu"
        """,
    )

    settings = load_settings(base, local_path=local, apply_env=False)
    assert settings.runtime.device == "cpu"
    # Untouched keys survive the merge.
    assert settings.runtime.inference_scheduler.max_queue_size > 0


def test_env_variable_overrides_yaml(tmp_path, monkeypatch):
    monkeypatch.setenv("SOCCER_RUNTIME__DEVICE", "cpu")
    monkeypatch.setenv("SOCCER_LOGGING__LEVEL", "DEBUG")

    settings = load_settings(DEFAULT_CONFIG, local_path=None, apply_env=True)
    assert settings.runtime.device == "cpu"
    assert settings.logging.level == "DEBUG"


def test_env_override_reaches_nested_keys(monkeypatch):
    monkeypatch.setenv("SOCCER_VIDEO__LOCAL_FILE__PATH", "D:/clips/match.mp4")

    settings = load_settings(DEFAULT_CONFIG, local_path=None, apply_env=True)
    assert settings.video.local_file.path == "D:/clips/match.mp4"


def test_env_override_parses_typed_scalars(monkeypatch):
    monkeypatch.setenv("SOCCER_VIDEO__LOCAL_FILE__LOOP", "false")
    monkeypatch.setenv("SOCCER_BUFFER__MAX_DECODED_FRAMES", "120")

    settings = load_settings(DEFAULT_CONFIG, local_path=None, apply_env=True)
    assert settings.video.local_file.loop is False
    assert settings.buffer.max_decoded_frames == 120


def test_encoded_buffer_must_cover_analysis_window(tmp_path):
    original = open(DEFAULT_CONFIG, encoding="utf-8").read()
    base = tmp_path / "base.yaml"
    base.write_text(
        original.replace("encoded_buffer_seconds: 30", "encoded_buffer_seconds: 5"),
        encoding="utf-8",
    )

    with pytest.raises(ConfigurationError, match="encoded_buffer_seconds"):
        load_settings(base, local_path=None, apply_env=False)


def test_action_spotter_provider_must_be_configured(tmp_path):
    original = open(DEFAULT_CONFIG, encoding="utf-8").read()
    base = tmp_path / "base.yaml"
    base.write_text(original.replace('provider: "tdeed"', 'provider: "nonexistent"'), encoding="utf-8")

    with pytest.raises(ConfigurationError, match="no entry under ai.providers"):
        load_settings(base, local_path=None, apply_env=False)
