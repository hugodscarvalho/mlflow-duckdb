"""Schema definition for the 'metrics' table.

Defines creation logic for the 'metrics' table used by the DuckDB MLflow
tracking store.
"""
import duckdb


def create_metrics_table(cursor: duckdb.DuckDBPyConnection) -> None:
    """Create the 'metrics' table if it does not exist.

    Args:
        cursor (duckdb.DuckDBPyConnection): A DuckDB connection or cursor object.

    """
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS metrics (
            run_id TEXT,
            key TEXT,
            value DOUBLE,
            timestamp BIGINT,
            step BIGINT,
            PRIMARY KEY (run_id, key, timestamp, step)
        )
    """)
