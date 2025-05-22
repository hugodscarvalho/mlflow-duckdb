"""client.py.

Provides a utility function to create and return an MLflow client instance.
This can be used to interact with the MLflow tracking server programmatically.

Example:
    from mlflow_duckdb.client import get_client
    client = get_client()
    client.list_experiments()

"""

import mlflow


def get_client():
    """Return a default MLflow client instance."""
    return mlflow.tracking.MlflowClient()
