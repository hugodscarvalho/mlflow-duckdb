"""Schema manager for the MLflow DuckDB Tracking Store.

This module is responsible for initializing and maintaining the required schema
(tables and structures) for MLflow tracking within a DuckDB database.

It provides the `setup_schema` function, which ensures that all necessary tables
exist before MLflow operations are performed. This setup includes support for:

- Experiments and experiment tags
- Runs, metrics, parameters, and tags
- Dataset inputs and traces

Each table is created only if it does not already exist, ensuring idempotent
initialization suitable for both new and existing databases.
"""
import duckdb
from .experiments import create_experiments_table
from .experiment_tags import create_experiment_tags_table
from .runs import create_runs_table
from .metrics import create_metrics_table
from .params import create_params_table
from .tags import create_tags_table
from .datasets import create_datasets_table
from .traces import create_traces_table


def setup_schema(cursor: duckdb.DuckDBPyConnection) -> None:
    """Create all necessary MLflow tables in DuckDB if they do not exist.

    Args:
        cursor (duckdb.DuckDBPyConnection): A DuckDB connection or cursor object.

    """
    create_experiments_table(cursor)
    create_experiment_tags_table(cursor)
    create_runs_table(cursor)
    create_metrics_table(cursor)
    create_params_table(cursor)
    create_tags_table(cursor)
    create_datasets_table(cursor)
    create_traces_table(cursor)
