import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import os

import pytest
from aicostmanager.triggered_limits_cache import triggered_limits_cache
try:
    from dotenv import load_dotenv
except ModuleNotFoundError:  # pragma: no cover - optional dependency
    def load_dotenv(*args, **kwargs):
        return None

# Load .env file from the current directory (sdks/python/tests)
ENV_PATH = os.path.join(os.path.dirname(__file__), ".env")
load_dotenv(ENV_PATH, override=True)

# Debug prints to confirm environment variable loading
print("AICM_API_KEY (pre-force):", os.environ.get("AICM_API_KEY"))
print("AICM_API_BASE:", os.environ.get("AICM_API_BASE"))
print("AICM_INI_PATH:", os.environ.get("AICM_INI_PATH"))
print("OPENAI_API_KEY (pre-force):", os.environ.get("OPENAI_API_KEY"))
print("ANTHROPIC_API_KEY (pre-force):", os.environ.get("ANTHROPIC_API_KEY"))
print("GOOGLE_API_KEY (pre-force):", os.environ.get("GOOGLE_API_KEY"))
print("DEEPSEEK_API_KEY (pre-force):", os.environ.get("DEEPSEEK_API_KEY"))
print("AWS_DEFAULT_REGION:", os.environ.get("AWS_DEFAULT_REGION"))

def pytest_collection_modifyitems(config, items):
    """Skip network-dependent tests unless explicitly enabled."""
    if os.environ.get("RUN_NETWORK_TESTS") == "1":
        return
    skip = pytest.mark.skip(reason="requires network access")
    for item in items:
        path = str(item.fspath)
        if "real" in path or "/tracker/" in path or "/tracker_async/" in path:
            item.add_marker(skip)


@pytest.fixture(scope="session", autouse=True)
def force_api_keys():
    # Always force the .env value for AICM_API_KEY and OPENAI_API_KEY
    ENV_PATH = os.path.join(os.path.dirname(__file__), ".env")
    load_dotenv(ENV_PATH, override=True)
    aicm_key = os.environ.get("AICM_API_KEY")
    openai_key = os.environ.get("OPENAI_API_KEY")
    anthropic_key = os.environ.get("ANTHROPIC_API_KEY")
    google_key = os.environ.get("GOOGLE_API_KEY")
    deepseek_key = os.environ.get("DEEPSEEK_API_KEY")
    aws_region = os.environ.get("AWS_DEFAULT_REGION")
    if not aicm_key:
        pytest.skip("AICM_API_KEY not set in .env file")
    if not openai_key:
        print("WARNING: OPENAI_API_KEY not set in .env file (some tests may skip)")
    os.environ["AICM_API_KEY"] = aicm_key
    os.environ["OPENAI_API_KEY"] = openai_key or ""
    if anthropic_key:
        os.environ["ANTHROPIC_API_KEY"] = anthropic_key
    if google_key:
        os.environ["GOOGLE_API_KEY"] = google_key
    if deepseek_key:
        os.environ["DEEPSEEK_API_KEY"] = deepseek_key
    if aws_region:
        os.environ["AWS_DEFAULT_REGION"] = aws_region
    print("AICM_API_KEY (forced):", os.environ.get("AICM_API_KEY"))
    print("OPENAI_API_KEY (forced):", os.environ.get("OPENAI_API_KEY"))
    yield
    # Optionally clear after tests
    # del os.environ["AICM_API_KEY"]
    # del os.environ["OPENAI_API_KEY"]


@pytest.fixture(scope="session")
def aicm_api_key():
    return os.environ.get("AICM_API_KEY")


@pytest.fixture(scope="session")
def openai_api_key():
    return os.environ.get("OPENAI_API_KEY")


@pytest.fixture(scope="session")
def anthropic_api_key():
    return os.environ.get("ANTHROPIC_API_KEY")


@pytest.fixture(scope="session")
def google_api_key():
    return os.environ.get("GOOGLE_API_KEY")


@pytest.fixture(scope="session")
def deepseek_api_key():
    return os.environ.get("DEEPSEEK_API_KEY")


@pytest.fixture(scope="session")
def aws_region():
    return os.environ.get("AWS_DEFAULT_REGION")


@pytest.fixture(scope="session")
def aicm_api_base():
    return os.environ.get("AICM_API_BASE", "http://127.0.0.1:8890")


@pytest.fixture(scope="session")
def aicm_ini_path(tmp_path_factory):
    ini_path = os.environ.get("AICM_INI_PATH")
    if ini_path:
        return ini_path
    # If not set, use a temp file
    tmp_dir = tmp_path_factory.mktemp("aicm_ini")
    return str(tmp_dir / "AICM.INI")


@pytest.fixture
def clear_triggered_limits(aicm_ini_path):
    """Clear triggered limits from the INI file to prevent test interference from usage limits."""
    import configparser

    # Read the current INI file
    cp = configparser.ConfigParser()
    try:
        cp.read(aicm_ini_path)
    except Exception:
        # File doesn't exist yet, that's fine
        pass

    # Remove triggered_limits section if it exists
    if "triggered_limits" in cp:
        cp.remove_section("triggered_limits")

    # Ensure triggered limits checks are enabled for tests that rely on them
    if "tracker" not in cp:
        cp.add_section("tracker")
    cp["tracker"]["AICM_LIMITS_ENABLED"] = "true"

    # Write the cleaned config back
    os.makedirs(os.path.dirname(aicm_ini_path), exist_ok=True)
    with open(aicm_ini_path, "w") as f:
        cp.write(f)

    yield

    # Cleanup: optionally restore original state or leave clean for next tests


@pytest.fixture(autouse=True)
def cleanup_ini_files():
    """Automatically clean up any test ini files created during testing."""
    import glob

    yield

    # Clean up any ini files in project root that might have been created accidentally
    for ini_file in glob.glob("ini*"):
        try:
            os.remove(ini_file)
        except OSError:
            pass

    # Clean up any test ini files in /tmp
    for ini_file in glob.glob("/tmp/test_*ini*"):
        try:
            os.remove(ini_file)
        except OSError:
            pass


@pytest.fixture(autouse=True)
def reset_triggered_limits_cache():
    triggered_limits_cache.clear()
    yield
    triggered_limits_cache.clear()
