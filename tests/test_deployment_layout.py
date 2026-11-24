import re
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


def test_release_workflow_builds_an_existing_dockerfile() -> None:
    release = (ROOT / ".github" / "workflows" / "release.yml").read_text(encoding="utf-8")
    referenced = re.findall(r"^\s*file:\s*(\S+)\s*$", release, re.MULTILINE)
    assert referenced, "release workflow must build a container image"
    for dockerfile in referenced:
        assert (ROOT / dockerfile).is_file(), f"release references missing {dockerfile}"


def test_dockerfiles_run_as_non_root() -> None:
    dockerfiles = [
        ROOT / "Dockerfile",
        ROOT / "modules" / "llama" / "Dockerfile",
        ROOT / "modules" / "ollama" / "Dockerfile",
    ]
    for dockerfile in dockerfiles:
        directives = [
            line.strip()
            for line in dockerfile.read_text(encoding="utf-8").splitlines()
            if line.strip().upper().startswith("USER ")
        ]
        assert directives, f"{dockerfile.name} must declare USER for Trivy DS-0002"
        assert all("root" not in d.lower() for d in directives), directives


def test_gateway_image_has_no_pip_or_known_vulnerable_zlib() -> None:
    dockerfile = (ROOT / "Dockerfile").read_text(encoding="utf-8")
    assert "zlib1g" in dockerfile, "patched zlib1g must be installed at build time"
    assert "site-packages/pip" in dockerfile, "pip must be removed from the runtime image"
    assert "ensurepip" in dockerfile, "bundled pip wheel must not remain in the image"


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
