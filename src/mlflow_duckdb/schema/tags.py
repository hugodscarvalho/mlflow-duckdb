"""Schema definition for the 'tags' table.

Defines creation logic for the 'tags' table used by the DuckDB MLflow
tracking store.
"""
import duckdb


def create_tags_table(cursor: duckdb.DuckDBPyConnection) -> None:
    """Create the 'tags' table if it does not exist.

    Args:
        cursor (duckdb.DuckDBPyConnection): A DuckDB connection or cursor object.

    """
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS tags (
            run_id TEXT,
            key TEXT,
            value TEXT,
            PRIMARY KEY (run_id, key)
        )
    """)
