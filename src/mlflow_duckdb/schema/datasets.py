"""Schema definition for the 'datasets' table.

Defines creation logic for the 'datasets' table used by the DuckDB MLflow
tracking store.
"""
import duckdb


def create_datasets_table(cursor: duckdb.DuckDBPyConnection) -> None:
    """Create the 'datasets' table if it does not exist.

    Args:
        cursor (duckdb.DuckDBPyConnection): A DuckDB connection or cursor object.

    """
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS datasets (
            run_id TEXT,
            name TEXT,
            digest TEXT,
            source_type TEXT,
            source TEXT,
            schema TEXT,
            tags TEXT
        )
    """)
