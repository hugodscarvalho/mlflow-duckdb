"""Schema definition for the 'runs' table.

Defines creation logic for the 'runs' table used by the DuckDB MLflow
tracking store.
"""
import duckdb


def create_runs_table(cursor: duckdb.DuckDBPyConnection) -> None:
    """Create the 'runs' table if it does not exist.

    Args:
        cursor (duckdb.DuckDBPyConnection): A DuckDB connection or cursor object.

    """
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS runs (
            run_id TEXT PRIMARY KEY,
            experiment_id TEXT NOT NULL,
            name TEXT,
            user_id TEXT,
            status TEXT,
            start_time BIGINT,
            end_time BIGINT,
            artifact_uri TEXT,
            lifecycle_stage TEXT,
            creation_time BIGINT,
            last_update_time BIGINT
        )
    """)
