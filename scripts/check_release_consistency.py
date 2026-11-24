"""Validate release metadata consistency for jarvis-inference.

Keeps the version in pyproject.toml, the top release entry in CHANGELOG.md and
the README status line in sync, so a tag cannot be cut from stale metadata.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

VERSION_RE = re.compile(r'^version\s*=\s*["\']([^"\']+)["\']', re.MULTILINE)
RELEASE_LINE_RE = re.compile(r"\*\*Current release:\s*`([^`]+)`\*\*")
CHANGELOG_ENTRY_RE = re.compile(r"^##\s+(\d+\.\d+\.\d+[^\s]*)\s", re.MULTILINE)
DOCKERFILE_REF_RE = re.compile(r"^\s*file:\s*(\S+)\s*$", re.MULTILINE)


def read_version(root: Path = ROOT) -> str:
    pyproject = root / "pyproject.toml"
    match = VERSION_RE.search(pyproject.read_text(encoding="utf-8"))
    if not match:
        raise SystemExit(f"Could not find project version in {pyproject}")
    return match.group(1)


def check(root: Path = ROOT) -> list[str]:
    """Return a list of inconsistencies; empty means everything is aligned."""
    version = read_version(root)
    problems: list[str] = []

    changelog = root / "CHANGELOG.md"
    entries = CHANGELOG_ENTRY_RE.findall(changelog.read_text(encoding="utf-8"))
    if not entries or entries[0] != version:
        top = entries[0] if entries else "missing"
        problems.append(f"CHANGELOG.md top release is {top}; expected {version}")

    readme_match = RELEASE_LINE_RE.search((root / "README.md").read_text(encoding="utf-8"))
    if not readme_match:
        problems.append("README.md is missing the '**Current release: `X.Y.Z`**' status line")
    elif readme_match.group(1) != version:
        problems.append(f"README.md current release is {readme_match.group(1)}; expected {version}")

    workflow = (root / ".github" / "workflows" / "release.yml").read_text(encoding="utf-8")
    for dockerfile in DOCKERFILE_REF_RE.findall(workflow):
        if not (root / dockerfile).is_file():
            problems.append(f"release workflow references missing {dockerfile}")

    return problems


def main() -> int:
    root = ROOT
    problems = check(root)
    if problems:
        for problem in problems:
            print(f"error: {problem}", file=sys.stderr)
        return 1
    print(f"release metadata consistent: {read_version(root)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
