import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_production_deployment_layout_is_minimal() -> None:
    assert (ROOT / "docker-compose.yml").is_file()
    assert (ROOT / "scripts" / "install.sh").is_file()
    assert not (ROOT / "docker" / "docker-compose.yml").exists()
    assert not (ROOT / "scripts" / "install-optiplex.sh").exists()
    assert not (ROOT / "scripts" / "watchdog.sh").exists()
    assert not (ROOT / "scripts" / "backup.sh").exists()
    assert not (ROOT / "deploy").exists()


def test_install_script_has_valid_shell_syntax() -> None:
    result = subprocess.run(
        ["bash", "-n", str(ROOT / "scripts" / "install.sh")],
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr


def test_remote_gateway_configuration_is_documented() -> None:
    env_example = (ROOT / ".env.example").read_text(encoding="utf-8")
    compose = (ROOT / "docker-compose.yml").read_text(encoding="utf-8")
    assert "INFERENCE_BIND_ADDRESS=127.0.0.1" in env_example
    assert "INFERENCE_BIND_ADDRESS:-127.0.0.1" in compose
    assert "./models:/models:ro" in compose
    assert "LLAMACPP_URL: http://llama-server:8080" in compose
    assert "LLAMACPP_BIND_ADDRESS:-127.0.0.1" in compose
    assert "LLAMACPP_THREADS=${LLAMACPP_THREADS:-8}" in compose
    assert "LLAMACPP_BATCH_THREADS=${LLAMACPP_BATCH_THREADS:-8}" in compose
    assert "MEMORY_BUDGET_GB:-10" in compose
    assert "pids_limit: 256" in compose
    assert "start_period: 300s" in compose
    assert "MAX_LOADED_MODELS=1" in env_example
