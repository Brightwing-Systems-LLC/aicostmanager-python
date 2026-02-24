# CLAUDE.md

## Project Overview

AICostManager Python SDK — tracks usage/costs for LLM and non-LLM services.

## Build & Publish

- **Do NOT publish locally with twine.** Publishing is handled by GitHub Actions.
- Push a `v*` tag (e.g. `git tag v0.3.1 && git push origin v0.3.1`) to trigger `.github/workflows/publish.yml`.
- The workflow uses `secrets.PYPI_API_TOKEN` stored in the GitHub repo settings.
- PyPI package: https://pypi.org/project/aicostmanager/

## Version Bumping

Update version in **two** places:
1. `aicostmanager/__init__.py` (`__version__`)
2. `pyproject.toml` (`version` — appears twice, use `replace_all`)

## Testing

- `python3 -m pytest tests/ -v` — run unit/mock tests
- `RUN_NETWORK_TESTS=1` — enable real API/network tests
- `set -a && source .env && set +a` — load API keys before network tests
- `pytest-asyncio` is NOT installed; async tests use `asyncio.run()` pattern
- Skips for `boto3` (Bedrock), `fireworks` (Fireworks SDK), `HEYGEN_API_KEY` are expected when those deps/keys are absent

## Architecture

- `aicostmanager/usage_utils.py` — extractor functions, `_KNOWN_EXTRACTORS` registry, `VENDOR_TO_API`/`API_TO_VENDOR` mappings
- `aicostmanager/wrappers.py` — `ServiceWrapper` (generic), named wrappers are thin aliases
- `aicostmanager/tracker.py` — core `Tracker` class, imports vendor mappings from `usage_utils`
- `aicostmanager/delivery/` — immediate and persistent queue delivery strategies

## Local AICM Backend

- Default: `http://127.0.0.1:8890` (set via `AICM_API_BASE` in `.env`)
- API keys and provider keys are in `tests/.env` and project root `.env`
