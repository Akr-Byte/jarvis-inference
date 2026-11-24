from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
INSTALL = ROOT / "scripts" / "install.sh"


def test_install_script_removes_only_stale_application_containers() -> None:
    content = INSTALL.read_text(encoding="utf-8")
    assert "for container in jarvis-inference jarvis-llama-server; do" in content
    assert 'docker container inspect "$container"' in content
    assert 'docker rm -f "$container"' in content
    assert "docker compose down -v" not in content


def test_install_script_requires_a_mounted_gguf_model() -> None:
    content = INSTALL.read_text(encoding="utf-8")
    assert 'gguf_models=("$model_dir"/*.gguf)' in content
    assert 'model_dir="${MODEL_DIRECTORY:-$ROOT/models}"' in content
    assert "No GGUF model found in $model_dir" in content
    assert "ollama pull" not in content
