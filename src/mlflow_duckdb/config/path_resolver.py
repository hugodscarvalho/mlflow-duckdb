"""Path-related utilities for resolving and preparing the DuckDB database file.

This module provides functions to:
- Parse and resolve the DuckDB path from a URI.
- Ensure the database directory exists on the local filesystem.
"""

import os
from mlflow.exceptions import MlflowException
from mlflow_duckdb.config.constants import (
    DEFAULT_DB_FILENAME,
    ERR_MSG_INVALID_URI,
)


def get_default_db_path() -> str:
    """Return the default DuckDB database file path.

    Returns:
        str: Expanded absolute path to the default DuckDB file.

    """
    return os.path.join(os.path.expanduser("~"), DEFAULT_DB_FILENAME)


def resolve_duckdb_path(store_uri: str, default_path: str | None = None) -> str:
    """Resolve a DuckDB file path from the provided URI.

    Args:
        store_uri (str): URI that may point to a DuckDB file.
        default_path (str | None): Fallback path if the URI does not include a path.
            Defaults to the user's home directory + default filename.

    Returns:
        str: Absolute file path to the DuckDB database.

    Raises:
        MlflowException: If the URI does not use a supported DuckDB scheme.

    """
    raw_path: str
    if store_uri.startswith("duckdb:///"):
        raw_path = store_uri[len("duckdb:///"):].strip()
    elif store_uri.startswith("duckdb://"):
        raw_path = store_uri[len("duckdb://"):].strip()
    elif store_uri.startswith("duckdb:"):
        raw_path = store_uri[len("duckdb:"):].strip()
    else:
        raise MlflowException(
            ERR_MSG_INVALID_URI.format(uri=store_uri),
        )

    if not default_path:
        default_path = get_default_db_path()

    return os.path.abspath(raw_path) if raw_path else default_path


def ensure_db_directory_exists(db_path: str) -> None:
    """Ensure that the directory for the DuckDB file exists.

    Args:
        db_path (str): Path to the DuckDB file.

    """
    db_dir = os.path.dirname(db_path)
    if db_dir and not os.path.exists(db_dir):
        os.makedirs(db_dir, exist_ok=True)
