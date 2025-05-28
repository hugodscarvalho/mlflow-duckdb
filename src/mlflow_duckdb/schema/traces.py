"""Schema definition for the 'traces' table.

Defines creation logic for the 'traces' table used by the DuckDB MLflow
tracking store.
"""
import duckdb


def create_traces_table(cursor: duckdb.DuckDBPyConnection) -> None:
    """Create the 'traces' table if it does not exist.

    Args:
        cursor (duckdb.DuckDBPyConnection): A DuckDB connection or cursor object.

    """
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS traces (
            request_id TEXT PRIMARY KEY,
            experiment_id TEXT,
            timestamp_ms BIGINT,
            execution_time_ms BIGINT,
            status TEXT,
            request_metadata TEXT,
            tags TEXT
        )
    """)
