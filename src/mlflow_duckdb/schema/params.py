"""Schema definition for the 'params' table.

Defines creation logic for the 'params' table used by the DuckDB MLflow
tracking store.
"""
import duckdb


def create_params_table(cursor: duckdb.DuckDBPyConnection) -> None:
    """Create the 'params' table if it does not exist.

    Args:
        cursor (duckdb.DuckDBPyConnection): A DuckDB connection or cursor object.

    """
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS params (
            run_id TEXT,
            key TEXT,
            value TEXT,
            PRIMARY KEY (run_id, key)
        )
    """)
