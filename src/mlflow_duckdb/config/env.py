"""Environment variable handling and access validation for the DuckDB MLflow plugin.

This module provides helpers to:
- Check if MLflow is running in read-only mode.
- Validate that a DuckDB file can be accessed under current mode.
"""

import os
from mlflow.exceptions import MlflowException
from mlflow_duckdb.config.constants import (
    ENV_READ_ONLY,
    TRUTHY_STRINGS,
    ERR_MSG_READ_ONLY_FILE_MISSING,
)


def is_read_only() -> bool:
    """Determine whether the DuckDB store should be opened in read-only mode.

    Reads the environment variable defined by ENV_READ_ONLY and returns True if it is
    set to a truthy value (e.g., '1', 'true', 'yes'), case-insensitive.

    Returns:
        bool: True if read-only mode is enabled, otherwise False.

    """
    return os.environ.get(ENV_READ_ONLY, "").strip().lower() in TRUTHY_STRINGS


def validate_duckdb_access(db_path: str, read_only: bool) -> None:
    """Ensure that the DuckDB file exists if running in read-only mode.

    Args:
        db_path (str): Path to the DuckDB file.
        read_only (bool): Whether the system is in read-only mode.

    Raises:
        MlflowException: If the file does not exist but read-only mode is enabled.

    """
    if read_only and not os.path.exists(db_path):
        raise MlflowException(ERR_MSG_READ_ONLY_FILE_MISSING.format(db_path=db_path))
