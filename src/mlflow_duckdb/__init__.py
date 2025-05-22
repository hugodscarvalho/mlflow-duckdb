"""mlflow_duckdb.

Top-level package for the MLflow-DuckDB plugin.
This module exposes the public interface for interacting with the plugin,
such as utility functions and plugin registration (if applicable).
"""

# Optional: Expose get_client at the package level
from .client import get_client as get_client
