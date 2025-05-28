"""Module for ensuring the default MLflow experiment exists in the DuckDB store.

This function creates a default experiment if it doesn't already exist. It is used
during initialization of the DuckDBTrackingStore.
"""

import time
import duckdb
from mlflow.entities.lifecycle_stage import LifecycleStage
from mlflow_duckdb.config.constants import (
    DEFAULT_EXPERIMENT_ID,
    DEFAULT_EXPERIMENT_NAME,
    DEFAULT_ARTIFACT_PATH_TEMPLATE,
    MS_IN_SECOND,
)


def ensure_default_experiment(
    cursor: duckdb.DuckDBPyConnection,
    db_path: str,
    default_id: str = DEFAULT_EXPERIMENT_ID,
) -> None:
    """Create the default MLflow experiment if it does not exist.

    Args:
        cursor (duckdb.DuckDBPyConnection): A DuckDB cursor to execute SQL commands.
        db_path (str): Filesystem path to DuckDB database, used to infer default
            artifact URI.
        default_id (str): Experiment ID to use for the default experiment.

    """
    exists = cursor.execute(
        "SELECT COUNT(*) FROM experiments WHERE experiment_id = ?", (default_id,),
    ).fetchone()[0]

    if not exists:
        timestamp = int(time.time() * MS_IN_SECOND)
        artifact_uri = DEFAULT_ARTIFACT_PATH_TEMPLATE.format(
            db_path=db_path,
            experiment_id=default_id,
        )
        cursor.execute(
            """
            INSERT INTO experiments (
                experiment_id,
                name,
                artifact_location,
                lifecycle_stage,
                creation_time,
                last_update_time
            ) VALUES (?, ?, ?, ?, ?, ?)
            """,
            (
                default_id,
                DEFAULT_EXPERIMENT_NAME,
                artifact_uri,
                LifecycleStage.ACTIVE,
                timestamp,
                timestamp,
            ),
        )
