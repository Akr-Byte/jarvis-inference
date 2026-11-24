# Releasing

This repository publishes three things from one immutable tag:

1. Python source/wheel artifacts on the GitHub Release,
2. SPDX SBOMs and provenance attestations on the GitHub Release,
3. Gateway and engine images on GHCR (`ghcr.io/akr-byte/jarvis-inference`).

There is no PyPI publication. The version in `pyproject.toml` is the single source of truth.

## Version policy

- `fix:` / `perf:` or documentation-only release corrections → patch release.
- `feat:` → minor release.
- Wire-contract or configuration changes that require operator action → minor release, with an
  explicit upgrade note in `CHANGELOG.md`.

Tags follow `vX.Y.Z` (for example `v0.6.0`). Once a tag is published it is immutable: corrections
require a new version, not a moved tag. Tags are global refs, so a tag pushed from a feature branch
still triggers `release.yml`; the workflow therefore refuses to publish unless the tagged commit is
already an ancestor of `origin/main`. Merge first, then tag.

## Pre-release checklist

- [ ] Update `pyproject.toml` `version`.
- [ ] Add a `## X.Y.Z` section at the top of `CHANGELOG.md` (below `## Unreleased`).
- [ ] Update the `Current release` status line in `README.md` to the new version.
- [ ] Run `python scripts/check_release_consistency.py` — it verifies all three files agree and
      that the release workflow points at an existing Dockerfile.
- [ ] Run the full CI gate locally (see below).
- [ ] Merge the release commit into `main` (the workflow verifies ancestry and will fail otherwise).
- [ ] Tag the commit on `main`, then push the annotated tag:
      `git tag -a vX.Y.Z -m "vX.Y.Z" && git push origin vX.Y.Z`.

## CI-equivalent local gate

```bash
python -m pip install -e '.[dev]' twine
python -m ruff check .
python -m ruff format --check .
bash -n scripts/*.sh
sh -n modules/llama/entrypoint.sh
python scripts/check_release_consistency.py
python -m pytest -q --cov=inference --cov-report=term-missing --cov-fail-under=70
docker compose build
python -m build
python -m twine check dist/*.whl dist/*.tar.gz
```

## What the tag-triggered workflow does

`.github/workflows/release.yml` runs on `push` of `v*.*.*` and can also be dispatched manually with a
`version` input to retry a publication for an existing immutable tag. It:

1. resolves the exact tag SHA and verifies the tag matches `pyproject.toml`,
2. verifies the tagged commit is on `main` (blocks releases from unmerged branches),
3. skips publication entirely if the GitHub Release already exists (idempotent),
4. re-runs every CI gate plus `twine check`, a clean-venv wheel import and `SHA256SUMS`,
5. builds and pushes `ghcr.io/akr-byte/jarvis-inference:{tag,latest,sha}`,
6. generates source and container SPDX SBOMs, scans the image with Trivy (`HIGH,CRITICAL`),
   attests provenance, and creates the GitHub Release with generated notes.

## Post-release verification

```bash
gh release view vX.Y.Z --json assets,tagName
curl -fsS "https://ghcr.io/v2/akr-byte/jarvis-inference/manifests/vX.Y.Z" -o /dev/null
docker compose config >/dev/null
```

Confirm the release assets include the wheel, sdist, `SHA256SUMS` and both SBOM files, and that the
GHCR manifest resolves for the new tag.

## Manual retry

If publication failed after the tag was pushed (for example a transient registry error), start the
workflow manually with the same `version`. The tag is immutable; no code or documentation fix may be
bundled into a retry. Corrections require a new patch release.
