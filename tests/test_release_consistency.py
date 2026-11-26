"""Release metadata must stay aligned across pyproject, changelog and README."""

import shutil
from pathlib import Path

from scripts.check_release_consistency import check, read_version

ROOT = Path(__file__).resolve().parents[1]


def test_release_metadata_is_consistent() -> None:
    assert check(ROOT) == []


def test_release_workflow_requires_tag_to_be_on_main() -> None:
    workflow = (ROOT / ".github" / "workflows" / "release.yml").read_text(encoding="utf-8")
    assert "merge-base --is-ancestor HEAD" in workflow
    assert "refs/remotes/origin/main" in workflow


def test_consistency_check_detects_readme_drift(tmp_path: Path) -> None:
    for name in ("pyproject.toml", "README.md", "CHANGELOG.md"):
        shutil.copy2(ROOT / name, tmp_path / name)
    workflow = tmp_path / ".github" / "workflows"
    workflow.mkdir(parents=True)
    shutil.copy2(ROOT / ".github" / "workflows" / "release.yml", workflow / "release.yml")
    shutil.copy2(ROOT / "Dockerfile", tmp_path / "Dockerfile")

    assert check(tmp_path) == []

    version = read_version(tmp_path)
    readme = tmp_path / "README.md"
    readme.write_text(
        readme.read_text(encoding="utf-8").replace(
            f"**Current release: `{version}`**", "**Current release: `0.0.0`**"
        ),
        encoding="utf-8",
    )
    assert any("README.md" in problem for problem in check(tmp_path))
