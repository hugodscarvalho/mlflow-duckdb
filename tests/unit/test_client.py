"""test_client.py.

Unit tests for the mlflow_duckdb.client module.
This module verifies that the get_client function returns a valid MlflowClient instance.
"""

from mlflow_duckdb.client import get_client
from mlflow.tracking import MlflowClient


def test_get_client_returns_mlflow_client():
    """Test that get_client returns an instance of MlflowClient."""
    client = get_client()
    assert isinstance(client, MlflowClient)
