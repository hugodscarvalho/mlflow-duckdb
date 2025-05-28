"""Schema definition for the 'experiments' table.

Defines creation logic for the 'experiments' table used by the DuckDB MLflow
tracking store.
"""
import duckdb


def create_experiments_table(cursor: duckdb.DuckDBPyConnection) -> None:
    """Create the 'experiments' table if it does not exist.

    Args:
        cursor (duckdb.DuckDBPyConnection): A DuckDB connection or cursor object.

    """
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS experiments (
            experiment_id TEXT PRIMARY KEY,
            name TEXT UNIQUE NOT NULL,
            artifact_location TEXT,
            lifecycle_stage TEXT,
            creation_time BIGINT,
            last_update_time BIGINT
        )
    """)
