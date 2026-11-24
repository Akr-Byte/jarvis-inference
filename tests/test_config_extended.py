from inference.config import Settings, _available_cores


def test_settings_parse_environment(monkeypatch) -> None:
    monkeypatch.setenv("INFERENCE_PORT", "9090")
    monkeypatch.setenv("MAX_CONCURRENT_REQUESTS", "3")
    monkeypatch.setenv("MAX_QUEUE_SIZE", "4")
    monkeypatch.setenv("MODEL_REGISTRY", "qwen3:1.7b, qwen3:4b")

    settings = Settings()

    assert settings.port == 9090
    assert settings.max_concurrent_requests == 3
    assert settings.max_queue_size == 4
    assert settings.models == ("qwen3:1.7b", "qwen3:4b")
    assert settings.inference_models == ("qwen3:1.7b", "qwen3:4b", "nomic-embed-text")
    assert settings.ollama_models == settings.inference_models


def test_settings_treat_empty_numeric_values_as_defaults(monkeypatch) -> None:
    """Templated .env files frequently leave numeric values blank."""
    monkeypatch.setenv("MAX_CONCURRENT_REQUESTS", "")
    monkeypatch.setenv("LLAMACPP_THREADS", "  ")
    monkeypatch.setenv("QUEUE_TIMEOUT_SECONDS", "")

    settings = Settings()

    assert settings.max_concurrent_requests == 1
    assert settings.llamacpp_threads >= 1
    assert settings.queue_timeout_seconds == 90.0


def test_settings_enforce_safe_minimums(monkeypatch) -> None:
    monkeypatch.setenv("MAX_CONCURRENT_REQUESTS", "0")
    monkeypatch.setenv("MAX_QUEUE_SIZE", "-1")
    monkeypatch.setenv("MAX_LOADED_MODELS", "0")
    monkeypatch.setenv("MEMORY_BUDGET_GB", "0.1")
    monkeypatch.setenv("MEMORY_HEADROOM_GB", "0.1")
    monkeypatch.setenv("BACKEND_HTTP_TIMEOUT_SECONDS", "0.1")
    monkeypatch.setenv("SHUTDOWN_TIMEOUT_SECONDS", "0.1")

    settings = Settings()

    assert settings.max_concurrent_requests == 1
    # maxsize=0 means unbounded for asyncio.Queue; the safe minimum is 1.
    assert settings.max_queue_size == 1
    assert settings.max_loaded_models == 1
    assert settings.memory_budget_gb == 0.5
    assert settings.memory_headroom_gb == 1.0
    assert settings.backend_http_timeout_seconds == 1
    assert settings.shutdown_timeout_seconds == 1


def test_managed_llamacpp_workstation_configuration(monkeypatch) -> None:
    monkeypatch.setenv("LLAMACPP_MANAGED", "true")
    monkeypatch.setenv("LLAMACPP_MODEL", "/models/model-00001-of-00004.gguf")
    monkeypatch.setenv("LLAMACPP_ALIAS", "workstation")
    monkeypatch.setenv("LLAMACPP_PORT", "9001")
    monkeypatch.setenv("MAX_CONTEXT_LENGTH", "131072")
    monkeypatch.delenv("MODEL_REGISTRY", raising=False)

    settings = Settings()

    assert settings.llamacpp_managed
    assert settings.llamacpp_base_url == "http://127.0.0.1:9001"
    assert settings.llamacpp_model.endswith("00001-of-00004.gguf")
    assert settings.models == ("workstation",)
    assert settings.max_context_length == 131072


def test_model_map_registers_multiple_runtime_profiles(monkeypatch) -> None:
    monkeypatch.setenv("LLAMACPP_MODEL_MAP", "coder=/models/coder.gguf;chat=/models/chat.gguf")
    monkeypatch.delenv("MODEL_REGISTRY", raising=False)
    monkeypatch.delenv("DEFAULT_MODEL", raising=False)

    settings = Settings()

    assert settings.llamacpp_model_map == {
        "coder": "/models/coder.gguf",
        "chat": "/models/chat.gguf",
    }
    assert settings.models == ("coder", "chat")
    assert settings.default_model == "coder"


def test_context_default_matches_documented_limit(monkeypatch) -> None:
    monkeypatch.delenv("MAX_CONTEXT_LENGTH", raising=False)

    settings = Settings()

    assert settings.max_context_length == 16384


def test_active_backend_reports_ollama_without_llamacpp(monkeypatch) -> None:
    monkeypatch.setenv("INFERENCE_BACKEND", "ollama")
    monkeypatch.setenv("LLAMACPP_MANAGED", "false")
    monkeypatch.delenv("LLAMACPP_URL", raising=False)

    settings = Settings()

    assert settings.active_backend == "ollama"


def test_available_cores_respects_cgroup_quota(monkeypatch) -> None:
    monkeypatch.setattr("inference.config._physical_cores", lambda: 8)
    monkeypatch.setattr("inference.config._cpu_quota", lambda: 2)

    assert _available_cores() == 2
