import os
from collections.abc import AsyncIterator, Iterator
from typing import Optional

import pytest
import pytest_asyncio
from synthefy import SynthefyAPIClient, SynthefyAsyncAPIClient
from synthefy.api_client import BASE_URL


def pytest_configure(config: pytest.Config) -> None:
    # Register here too so the marker exists when a parent pytest config is active.
    config.addinivalue_line(
        "markers",
        "slow: marks tests as slow or optional, e.g. real local nori inference",
    )


def pytest_addoption(parser: pytest.Parser) -> None:
    parser.addoption(
        "--synthefy-api-target",
        action="store",
        default="prod",
        choices=("local", "dev", "prod"),
        help=(
            "Which Synthefy Forecasting API to hit: "
            "'prod' (default), 'dev' (dev.forecast.synthefy.com), or "
            "'local' (http://localhost:{FORECASTING_API_PORT})."
        ),
    )


@pytest.fixture(scope="session")
def synthefy_api_target(pytestconfig: pytest.Config) -> str:
    return str(pytestconfig.getoption("--synthefy-api-target"))


@pytest.fixture(scope="session")
def synthefy_base_url(synthefy_api_target: str) -> str:
    if synthefy_api_target == "local":
        port = os.getenv("FORECASTING_API_PORT", "8018")
        return f"http://localhost:{port}"
    if synthefy_api_target == "dev":
        return "https://dev.forecast.synthefy.com"
    return BASE_URL


@pytest.fixture(scope="session")
def synthefy_api_key(synthefy_api_target: str) -> Optional[str]:
    if synthefy_api_target in ("dev", "prod"):
        return os.getenv("SYNTHEFY_API_KEY")
    return None


@pytest.fixture(scope="session")
def require_synthefy_api_key(
    synthefy_api_target: str, synthefy_api_key: Optional[str]
) -> None:
    if synthefy_api_target in ("dev", "prod") and not synthefy_api_key:
        pytest.skip("SYNTHEFY_API_KEY environment variable not set")


@pytest.fixture
def synthefy_client(
    synthefy_base_url: str,
    synthefy_api_key: Optional[str],
    require_synthefy_api_key: None,
) -> Iterator[SynthefyAPIClient]:
    client = SynthefyAPIClient(
        api_key=synthefy_api_key, base_url=synthefy_base_url
    )
    try:
        yield client
    finally:
        client.close()


@pytest_asyncio.fixture
async def synthefy_async_client(
    synthefy_base_url: str,
    synthefy_api_key: Optional[str],
    require_synthefy_api_key: None,
) -> AsyncIterator[SynthefyAsyncAPIClient]:
    async with SynthefyAsyncAPIClient(
        api_key=synthefy_api_key,
        base_url=synthefy_base_url,
    ) as client:
        yield client
