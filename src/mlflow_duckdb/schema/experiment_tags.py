"""Schema definition for the 'experiment_tags' table.

Defines creation logic for the 'experiment_tags' table used by the DuckDB
MLflow tracking store.
"""
import duckdb


def create_experiment_tags_table(cursor: duckdb.DuckDBPyConnection) -> None:
    """Create the 'experiment_tags' table if it does not exist.

    Args:
        cursor (duckdb.DuckDBPyConnection): A DuckDB connection or cursor object.

    """
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS experiment_tags (
            experiment_id TEXT,
            key TEXT,
            value TEXT,
            PRIMARY KEY (experiment_id, key)
        )
    """)
