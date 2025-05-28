"""Integration test configuration for mlflow-duckdb.

This module provides shared fixtures and utilities for integration tests,
including a health check to ensure the MLflow Tracking Server is running
before tests are executed.
"""

import os
import pytest
import requests


def is_mlflow_server_up(uri: str) -> bool:
    """Check if the MLflow Tracking Server is running and healthy.

    Args:
        uri (str): The URI of the MLflow Tracking Server.

    Returns:
        bool: True if the server responds with HTTP 200 to /health, otherwise False.

    """
    try:
        timeout: int = int(os.getenv("MLFLOW_HEALTH_TIMEOUT", "5"))
        response = requests.get(f"{uri}/health", timeout=timeout)
        return response.status_code == 200
    except requests.RequestException:
        return False


@pytest.fixture(scope="session", autouse=True)
def check_mlflow_server() -> None:
    """Fixture to ensure the MLflow server is reachable before integration tests run.

    This runs once per test session. If the server is unavailable, it skips all tests
    in the integration suite.

    Raises:
        pytest.skip: If the MLflow Tracking Server is unreachable.

    """
    uri: str = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
    if not is_mlflow_server_up(uri):
        pytest.skip(
            reason=(
                "Skipping integration tests — no MLflow server running at "
                f"{uri}"
            ),
            allow_module_level=True,
        )
